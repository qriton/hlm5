"""Phase-2 gates: inject novel facts into the FineWeb-trained trunk (65K custom BPE).

Port of run_real_vocab_kb_inject.py (which passed all G1 gates on the GPT-2-BPE trunk)
to the Leonardo trunk: editability floor, expressible targets, fact-set-mean-centered +
whitened keys, RAW unit(w_t) values (never center values), exact auto-boost, audit,
holdout + distribution-level collateral, edit/forget.

Tokenizer: the FineWeb 65K byte-level BPE (HuggingFace `tokenizers` JSON; specials
<pad>=0 <unk>=1 <bos>=2 <eos>=3; word-initial tokens carry the 'Ġ' prefix).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tokenizers import Tokenizer as HFTokenizer

from hlm5_lm import HLM5LM
from hlm5_memory import EditableHLM5Memory, unit

BOS, EOS = 2, 3

FINEWEB_VAL = "/leonardo_scratch/large/userexternal/mdima000/hlm3/data/fineweb/fineweb_22b_val.bin"
# scratch work/tokenizers was purged; the verified copy lives in the repo now
TOKENIZER_JSON = str(Path("~/HLM5/tokenizers/fineweb-65536.json").expanduser())


def novel_country_names():
    # Preserve the original first 64 names used by the registered G2/G3 gates,
    # then append a larger deterministic pool for stress tests beyond K=64.
    base_pre = ["Vor", "Zan", "Mar", "Tel", "Bru", "Kal", "Dor", "Fen"]
    base_suf = ["enia", "thia", "landia", "markia", "onia", "avia", "ostan", "oria"]
    names = [p + s for p, s in itertools.product(base_pre, base_suf)]

    extra_pre = [
        "Ari", "Bel", "Cor", "Eld", "Gal", "Har", "Ish", "Jor",
        "Len", "Mor", "Niv", "Orl", "Pav", "Qel", "Riv", "Sol",
        "Tur", "Ulm", "Val", "Wes", "Xan", "Yor", "Zen", "Ost",
    ]
    extra_suf = [
        "ara", "eria", "ovia", "istan", "mere", "vale", "dora", "mont",
        "grad", "port", "holm", "wick", "shire", "fjord", "crest", "haven",
        "polis", "mir", "nor", "tavia", "quay", "merea", "voss", "zar",
    ]
    seen = set(names)
    for p, s in itertools.product(extra_pre, extra_suf):
        name = p + s
        if name not in seen:
            names.append(name)
            seen.add(name)
    return names


def open_mmap(path):
    with open(path, "rb") as f:
        header = f.read(16)
    if header[:4] != b"HLM3":
        raise ValueError(f"{path}: bad magic")
    return np.memmap(path, dtype=np.uint16, mode="r", offset=16)


def encode_prompt(tokenizer, text):
    ids = tokenizer.encode(text).ids
    if ids and ids[0] == BOS:
        ids = ids[1:]
    if ids and ids[-1] == EOS:
        ids = ids[:-1]
    return torch.tensor(ids, dtype=torch.long)[None]


def layer_states_for_jacobian(model: HLM5LM, tokens: torch.Tensor):
    """Return residual states after embed and every transformer block."""
    T = tokens.shape[1]
    x = model.token_emb(tokens) + model.pos_emb[:, :T]
    names = ["embed"]
    states = [x]
    mask = torch.nn.Transformer.generate_square_subsequent_mask(T, device=tokens.device)
    for i, layer in enumerate(model.blocks.layers):
        x = layer(x, src_mask=mask, is_causal=True)
        names.append(f"block_{i}")
        states.append(x)
    if model.trained_memory is not None:
        x, _ = model.trained_memory(x)
    return names, states, model.norm(x)


def selected_token_jacobian_vectors(model: HLM5LM, tokenizer, token_ids: list[int],
                                    prompts: list[str], device):
    """Average d logit(token) / d residual_layer(last_pos) over prompts."""
    token_ids = [int(t) for t in dict.fromkeys(token_ids)]
    accum = None
    layer_names = None
    with torch.enable_grad():
        for prompt in prompts:
            ids = encode_prompt(tokenizer, prompt).to(device)
            names, states, final_hidden = layer_states_for_jacobian(model, ids)
            if accum is None:
                layer_names = names
                accum = {
                    tid: [torch.zeros(model.dim, device=device) for _ in names]
                    for tid in token_ids
                }
            logits = model.head(final_hidden)[0, -1]
            for tid in token_ids:
                grads = torch.autograd.grad(
                    logits[tid],
                    states,
                    retain_graph=True,
                    allow_unused=True,
                )
                for i, grad in enumerate(grads):
                    if grad is not None:
                        accum[tid][i] += grad[0, -1].detach()

    assert accum is not None and layer_names is not None
    denom = float(len(prompts))
    return layer_names, {
        tid: [unit(v / denom, dim=0).detach() for v in per_layer]
        for tid, per_layer in accum.items()
    }


def rare_wordlike_pool(model, tokenizer, freqs: np.ndarray,
                       pool_frac: float = 0.10,
                       min_pool: int = 1):
    """Low-frequency word-like self-NN-reachable tokens sorted by corpus freq."""
    W = model.head.weight.detach()
    n = W.shape[0]
    reachable = []
    for s in range(0, n, 4096):
        am = (W[s:s + 4096] @ W.T).argmax(-1)
        ok = am == torch.arange(s, min(s + 4096, n), device=W.device)
        reachable.extend((s + torch.nonzero(ok).flatten()).tolist())
    inv = {i: s for s, i in tokenizer.get_vocab().items()}
    wordlike = [t for t in reachable
                if isinstance(inv.get(t), str)
                and re.fullmatch(r"Ġ[A-Za-z]{3,}", inv[t])]
    wordlike = sorted(wordlike, key=lambda t: int(freqs[t]))
    cutoff = max(1, min_pool, int(np.ceil(len(wordlike) * pool_frac)))
    cutoff = min(cutoff, len(wordlike))
    pool = wordlike[:cutoff]
    fr = [int(freqs[t]) for t in pool]
    print(f"rare pool frac={pool_frac:.2f}: {len(pool)} word-like targets, "
          f"corpus freq {min(fr)}..{max(fr)}")
    return pool, inv, len(reachable) / n, len(wordlike)


@torch.no_grad()
def probe_flippable(model, h, tid: int, key_mean, whiten, dim: int, device) -> bool:
    """True iff finite boost can flip tid at fact hidden h (single-slot probe)."""
    mem = EditableHLM5Memory(dim=dim, memory_size=4, degree=5, temperature=0.10).to(device)
    key = (h - key_mean) @ whiten
    mem.inject(key, value=unit(model.head.weight[tid].detach(), 0), label="probe")
    q = key[None, None, :]
    g = torch.relu(mem.score(q)).max()
    ret, _ = mem(q)
    delta = model.head(g * ret[0, 0])
    base = model.head(h)
    gap = base - base[tid]
    slope = delta[tid] - delta
    bad = bool(((slope <= 1e-6) & (gap > 0)).any())
    return not bad


@torch.no_grad()
def multi_slot_unflippable_indices(model, fact_h, targets, key_mean, whiten,
                                   dim: int, device) -> list[int]:
    """Return selected fact indices that cannot flip after multi-slot retrieval.

    The single-slot probe proves the raw value can beat the trunk head in
    isolation. Rare targets can still fail when neighboring memory slots blend
    into the retrieved vector. This check mirrors the later exact-boost
    feasibility test while the target set is being built.
    """
    if not targets:
        return []
    mem = EditableHLM5Memory(
        dim=dim,
        memory_size=max(4, len(targets) + 4),
        degree=5,
        temperature=0.10,
    ).to(device)
    for i, tid in enumerate(targets):
        key = (fact_h[i] - key_mean) @ whiten
        mem.inject(key, value=unit(model.head.weight[tid].detach(), 0), label=f"probe-{i}")

    bad_indices: list[int] = []
    for i, tid in enumerate(targets):
        q = ((fact_h[i] - key_mean) @ whiten)[None, None, :]
        g = torch.relu(mem.score(q)).max()
        ret, _ = mem(q)
        delta = model.head(g * ret[0, 0])
        base = model.head(fact_h[i])
        gap = base - base[tid]
        slope = delta[tid] - delta
        if bool(((slope <= 1e-6) & (gap > 0)).any()):
            bad_indices.append(i)
    return bad_indices


def pick_flippable_rare_targets(model, fact_h, countries, pool, inv, key_mean, whiten,
                                want: int, dim: int, device,
                                require_multislot: bool = True):
    """Assign rare targets that remain flippable after multi-slot retrieval."""
    used: set[int] = set()
    picked: list[int] = []
    skipped: list[dict] = []
    for i in range(want):
        h = fact_h[i]
        found = None
        for tid in pool:
            if tid in used:
                continue
            if probe_flippable(model, h, tid, key_mean, whiten, dim, device):
                if require_multislot:
                    bad = multi_slot_unflippable_indices(
                        model, fact_h, [*picked, tid], key_mean, whiten, dim, device)
                    if bad:
                        if len(skipped) < 32:
                            skipped.append({"index": i, "country": countries[i],
                                            "target_id": int(tid),
                                            "target_token": inv.get(tid, "?"),
                                            "reason": f"multi_slot_bad_indices={bad}"})
                        continue
                found = tid
                break
            if len(skipped) < 32:
                skipped.append({"index": i, "country": countries[i], "target_id": int(tid),
                                "target_token": inv.get(tid, "?"),
                                "reason": "single_slot_unflippable"})
        if found is None:
            raise RuntimeError(
                f"flippability filter: could not fill fact {i} ({countries[i]}) "
                f"from {len(pool)} rare-decile candidates")
        used.add(found)
        picked.append(found)
    mode = "multi-slot" if require_multislot else "single-slot"
    print(f"flippability filter ({mode}): picked {want}/{want} rare targets "
          f"(skipped {len(skipped)} failed probes in scan)")
    print(f"sample targets: {[inv[t][1:] for t in picked[:8]]}")
    return picked, skipped


def expressible_targets(model, tokenizer, want: int, mode: str = "spread",
                        freqs: np.ndarray | None = None):
    """Targets from the trunk's expressible vocabulary: self-NN argmax-reachable
    (the per-fact guarantee that raw unit(w_t) beats every competitor) AND a full
    word start (byte-level BPE marks word-initial tokens with the 'Ġ' prefix).

    mode='rare' (gate G2c): restrict to the rarest decile of the word-like
    expressible set by measured train-corpus frequency -- the HLM3 rare-token wall,
    served by injection instead of training."""
    W = model.head.weight.detach()
    n = W.shape[0]
    reachable = []
    for s in range(0, n, 4096):
        am = (W[s:s + 4096] @ W.T).argmax(-1)
        ok = am == torch.arange(s, min(s + 4096, n), device=W.device)
        reachable.extend((s + torch.nonzero(ok).flatten()).tolist())
    inv = {i: s for s, i in tokenizer.get_vocab().items()}
    wordlike = [t for t in reachable
                if isinstance(inv.get(t), str)
                and re.fullmatch(r"Ġ[A-Za-z]{3,}", inv[t])]
    frac = len(reachable) / n
    print(f"expressible vocab: {len(reachable)}/{n} ({100 * frac:.1f}%) argmax-reachable, "
          f"{len(wordlike)} word-like")
    if len(wordlike) < want:
        raise RuntimeError(f"trunk can express only {len(wordlike)} word-like targets; "
                           f"need {want} -- trunk below the editability floor")
    if mode == "rare":
        if freqs is None:
            raise ValueError("rare mode needs corpus frequencies")
        wordlike = sorted(wordlike, key=lambda t: int(freqs[t]))
        wordlike = wordlike[: max(want, len(wordlike) // 10)]  # rarest decile
        fr = [int(freqs[t]) for t in wordlike]
        print(f"rare decile: {len(wordlike)} targets, corpus freq {min(fr)}..{max(fr)}")
    step = max(1, len(wordlike) // want)
    picked = wordlike[::step][:want]
    print(f"sample targets: {[inv[t][1:] for t in picked[:8]]}")
    return picked, frac, len(wordlike), inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/hlm5_base_fineweb/best.pt")
    ap.add_argument("--val-bin", default=FINEWEB_VAL)
    ap.add_argument("--tokenizer", default=TOKENIZER_JSON)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--memory-size", type=int, default=256)
    ap.add_argument("--gate-thresh", type=float, default=0.95)
    ap.add_argument("--hook-mode", choices=["prehead", "residual"], default="prehead",
                    help="prehead = legacy final-hidden hook; residual = internal stream hook")
    ap.add_argument("--hook-layer-index", type=int, default=None,
                    help="residual mode: apply memory after this many transformer blocks")
    ap.add_argument("--calibration-prompt", action="append", default=None,
                    help="residual mode: prompt for selected-token Jacobian averaging")
    ap.add_argument("--target-mode", choices=["spread", "rare"], default="spread",
                    help="rare = gate G2c: rarest decile of expressible targets")
    ap.add_argument("--train-bin", default=None,
                    help="train bin for the rare-mode frequency sample (first 100M tokens)")
    ap.add_argument("--out", default="reports/fineweb_kb_inject.json")
    ap.add_argument("--flippability-filter", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="rare mode: skip unflippable targets when picking K facts (G3c close)")
    ap.add_argument("--rare-picker", choices=["single", "greedy"], default="greedy",
                    help="rare mode: greedy also rejects targets that fail after multi-slot retrieval")
    ap.add_argument("--rare-pool-frac", type=float, default=0.10,
                    help="rare mode: primary low-frequency candidate fraction")
    ap.add_argument("--rare-pool-fallback-frac", type=float, default=0.25,
                    help="rare mode: retry this wider low-frequency fraction if the primary pool cannot fill")
    ap.add_argument("--rare-pool-fallback-min-multiple", type=int, default=12,
                    help="rare mode: fallback pool has at least this many candidates per requested fact")
    args = ap.parse_args()
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = HFTokenizer.from_file(args.tokenizer)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = HLM5LM(vocab=ck["vocab"], **ck["cfg"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    ctx = ck["cfg"]["max_len"]
    print(f"loaded {args.checkpoint}: val PPL {ck['val_ppl']:.2f} (step {ck['step']})")
    if args.hook_mode == "residual":
        if args.hook_layer_index is None:
            raise ValueError("--hook-mode residual requires --hook-layer-index")
        n_layers = len(model.blocks.layers)
        if not 0 <= args.hook_layer_index <= n_layers:
            raise ValueError(f"--hook-layer-index must be in [0, {n_layers}]")
        print(f"residual hook mode: layer index {args.hook_layer_index}/{n_layers}")

    def feature_hidden(ids: torch.Tensor) -> torch.Tensor:
        if args.hook_mode == "residual":
            return model.residual_state(ids, args.hook_layer_index)[0]
        return model.hidden(ids)

    def logits_from_feature(index: int, h: torch.Tensor) -> torch.Tensor:
        if args.hook_mode == "prehead":
            return model.head(h)
        x, mask = model.residual_state(fact_ids[index], args.hook_layer_index)
        x = x.clone()
        x[:, -1, :] = h
        return model.head(model.continue_from_residual_state(
            x, mask, args.hook_layer_index))[0, -1]

    val_buf = open_mmap(args.val_bin)
    windows = [torch.from_numpy(val_buf[s * ctx:(s + 1) * ctx + 1].astype(np.int64))
               for s in range(164)]
    ref_windows, holdout = windows[:100], windows[100:]
    with torch.no_grad():
        means = [feature_hidden(w[None, :-1].to(device))[0].mean(0) for w in ref_windows]
    text_mean = torch.stack(means).mean(0)

    K = args.k
    countries = novel_country_names()[:K]
    prompts = [f"The capital of {c} is" for c in countries]
    freqs = None
    picker_meta: dict = {}
    if args.target_mode == "rare":
        train_buf = open_mmap(args.train_bin or
                              args.val_bin.replace("_val.bin", "_train.bin"))
        sample = np.asarray(train_buf[:100_000_000])
        freqs = np.bincount(sample, minlength=ck["vocab"])
        print(f"frequency sample: {sample.shape[0]:,} tokens")

    fact_h, fact_ids = [], []
    with torch.no_grad():
        for p in prompts:
            ids = encode_prompt(tokenizer, p).to(device)
            fact_h.append(feature_hidden(ids)[0, -1])
            fact_ids.append(ids)

    fact_mean = torch.stack(fact_h).mean(0)

    def coherence(mean_vec, transform=None):
        kk = torch.stack(fact_h) - mean_vec
        if transform is not None:
            kk = kk @ transform
        kk = unit(kk, -1)
        return float((kk @ kk.T).fill_diagonal_(0).abs().max())

    with torch.no_grad():
        text_h = torch.cat([feature_hidden(w[None, :-1].to(device))[0]
                            for w in ref_windows[:20]])
        resid = torch.cat([torch.stack(fact_h), text_h]) - fact_mean
        cov = (resid.T @ resid) / resid.shape[0]
        eigval, eigvec = torch.linalg.eigh(cov.float())
        floor = 0.01 * float(eigval.mean())
        whiten = eigvec @ torch.diag((eigval + floor).rsqrt()) @ eigvec.T

    rho_text, rho_fact = coherence(text_mean), coherence(fact_mean)
    rho_white = coherence(fact_mean, whiten)
    key_mean = fact_mean
    print(f"key coherence: text-centered {rho_text:.3f}, fact-centered {rho_fact:.3f}, "
          f"+whitened {rho_white:.3f}")

    if args.hook_mode == "residual" and args.target_mode == "rare" and args.flippability_filter:
        print("rare flippability filter is pre-head-specific; skipping for residual hook mode")
        picker_meta["flippability_filter_skipped_reason"] = (
            "pre-head unit(w_t) flippability filter does not apply to residual "
            "selected-token Jacobian values"
        )
        args.flippability_filter = False

    if args.target_mode == "rare" and args.flippability_filter:
        pool, inv, reachable_frac, n_wordlike = rare_wordlike_pool(
            model, tokenizer, freqs, pool_frac=args.rare_pool_frac)
        picker_error = None
        try:
            targets, picker_skipped = pick_flippable_rare_targets(
                model, fact_h, countries, pool, inv, key_mean, whiten, K,
                ck["cfg"]["dim"], device,
                require_multislot=args.rare_picker == "greedy")
            picker_pool_frac = args.rare_pool_frac
        except RuntimeError as exc:
            fallback_frac = max(args.rare_pool_fallback_frac, args.rare_pool_frac)
            fallback_min = K * max(1, args.rare_pool_fallback_min_multiple)
            if fallback_frac <= args.rare_pool_frac and fallback_min <= len(pool):
                raise
            picker_error = str(exc)
            print(f"rare picker primary pool failed: {picker_error}")
            print(f"retrying with wider low-frequency pool "
                  f"frac={fallback_frac:.2f}, min_pool={fallback_min}")
            pool, inv, reachable_frac, n_wordlike = rare_wordlike_pool(
                model, tokenizer, freqs,
                pool_frac=fallback_frac,
                min_pool=fallback_min)
            targets, picker_skipped = pick_flippable_rare_targets(
                model, fact_h, countries, pool, inv, key_mean, whiten, K,
                ck["cfg"]["dim"], device,
                require_multislot=args.rare_picker == "greedy")
            picker_pool_frac = fallback_frac
        picker_meta = {
            "picker": (
                "flippability_filtered_multislot_rare_v3"
                if args.rare_picker == "greedy"
                else "flippability_filtered_single_slot_rare_v1"
            ),
            "pool_size": len(pool),
            "pool_fraction": picker_pool_frac,
            "primary_pool_fraction": args.rare_pool_frac,
            "primary_picker_error": picker_error,
            "picker_skipped_sample": picker_skipped[:16],
        }
    else:
        targets, reachable_frac, n_wordlike, inv = expressible_targets(
            model, tokenizer, K, mode=args.target_mode, freqs=freqs)

    base_pred = []
    with torch.no_grad():
        for i, h in enumerate(fact_h):
            base_pred.append(int(logits_from_feature(i, h).argmax()))
    already = sum(int(b == t) for b, t in zip(base_pred, targets))
    print(f"K={K} facts; {already} already predicted at baseline (should be ~0)")

    calibration_prompts = args.calibration_prompt or [
        "The answer is",
        "In summary, the result is",
        "The capital city is",
        "The correct word is",
    ]
    layer_names = None
    jacobian_vectors = None
    if args.hook_mode == "residual":
        print(f"building selected-token Jacobian values for {len(set(targets))} targets")
        layer_names, jacobian_vectors = selected_token_jacobian_vectors(
            model, tokenizer, targets, calibration_prompts, device)

    def memory_value(target_id: int) -> torch.Tensor:
        if args.hook_mode == "residual":
            return jacobian_vectors[int(target_id)][args.hook_layer_index]
        return unit(model.head.weight[target_id].detach(), 0)

    mem = EditableHLM5Memory(dim=ck["cfg"]["dim"], memory_size=args.memory_size,
                             degree=5, temperature=0.10).to(device)
    slots = [mem.inject((fact_h[i] - key_mean) @ whiten,
                        value=memory_value(targets[i]),
                        label=f"{countries[i]}->{inv[targets[i]][1:]}")
             for i in range(K)]

    unflippable_detail = []
    boost_validation_flips = None
    with torch.no_grad():
        need, unflippable = 0.0, 0
        for i in range(K):
            q = ((fact_h[i] - key_mean) @ whiten)[None, None, :]
            g = torch.relu(mem.score(q)).amax(-1, keepdim=True)
            g = torch.where(g >= args.gate_thresh, g, torch.zeros_like(g))
            ret, _ = mem(q)
            base = logits_from_feature(i, fact_h[i])
            plus = logits_from_feature(i, fact_h[i] + g[0, 0, 0] * ret[0, 0])
            delta = plus - base
            tid = targets[i]
            gap = base - base[tid]
            slope = delta[tid] - delta
            flippable = slope > 1e-6
            bad = bool(((slope <= 1e-6) & (gap > 0)).any())
            if bad:
                unflippable += 1
                worst_j = int(gap.argmax())
                unflippable_detail.append({
                    "index": i,
                    "country": countries[i],
                    "target_id": int(tid),
                    "target_token": inv.get(tid, "?"),
                    "base_argmax_token": inv.get(int(base.argmax()), "?"),
                    "max_gap_token": inv.get(worst_j, "?"),
                    "max_gap": float(gap.max()),
                    "corpus_freq": int(freqs[tid]) if freqs is not None else None,
                })
            if bool(flippable.any()):
                need = max(need, float((gap[flippable] / slope[flippable]).clamp_min(0.0).max()))
        boost = 1.5 * need + 1.0
        if args.hook_mode == "residual":
            for _ in range(8):
                ok = 0
                for i in range(K):
                    q = ((fact_h[i] - key_mean) @ whiten)[None, None, :]
                    g = torch.relu(mem.score(q)).amax(-1, keepdim=True)
                    g = torch.where(g >= args.gate_thresh, g, torch.zeros_like(g))
                    ret, _ = mem(q)
                    h_after = fact_h[i] + boost * g[0, 0, 0] * ret[0, 0]
                    ok += int(int(logits_from_feature(i, h_after).argmax()) == targets[i])
                boost_validation_flips = ok
                if ok == K:
                    break
                boost *= 1.5
    print(f"auto boost = {boost:.1f}" + (f"  WARNING: {unflippable}/{K} unflippable"
                                         if unflippable else ""))
    if boost_validation_flips is not None:
        print(f"residual boost validation flips {boost_validation_flips}/{K}")
    for row in unflippable_detail:
        print(f"  unflippable [{row['index']}] {row['country']} -> {row['target_token']!r} "
              f"(base wants {row['base_argmax_token']!r}, worst gap {row['max_gap']:.3f} "
              f"via {row['max_gap_token']!r})")

    def attach_current_memory() -> None:
        if args.hook_mode == "residual":
            model.attach_residual_memory(
                mem,
                key_mean,
                boost,
                args.hook_layer_index,
                args.gate_thresh,
                key_transform=whiten,
            )
        else:
            model.attach_memory(mem, key_mean, boost, args.gate_thresh, key_transform=whiten)

    attach_current_memory()

    @torch.no_grad()
    def predict(ids):
        logits, audit = model(ids, return_audit=True)
        return int(logits[0, -1].argmax()), audit

    flips = audits = 0
    failed_inject = []
    for i in range(K):
        p, audit = predict(fact_ids[i])
        ok = p == targets[i]
        flips += int(ok)
        audits += int(int(audit.top_slots.reshape(-1, audit.top_slots.shape[-1])[-1][0]) == slots[i])
        if not ok:
            failed_inject.append({
                "index": i,
                "country": countries[i],
                "target_token": inv[targets[i]],
                "got_token": inv.get(p, "?"),
                "unflippable_precheck": any(u["index"] == i for u in unflippable_detail),
            })

    flips_holdout = 0
    for w in holdout:
        x = w[None, :-1].to(device)
        with torch.no_grad():
            after, _ = model(x)
            model.detach_memory()
            before, _ = model(x)
            attach_current_memory()
        flips_holdout += int((after.argmax(-1) != before.argmax(-1)).sum())

    def ppl_over(ws):
        losses = []
        with torch.no_grad():
            for w in ws:
                x, y = w[None, :-1].to(device), w[None, 1:].to(device)
                logits, _ = model(x)
                losses.append(float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                    y.reshape(-1))))
        return math.exp(sum(losses) / len(losses))

    ppl_attached = ppl_over(ref_windows)
    model.detach_memory()
    ppl_base = ppl_over(ref_windows)
    attach_current_memory()
    ppl_delta_pct = 100.0 * (ppl_attached - ppl_base) / ppl_base

    print(f"G2: inject_flip {flips}/{K}  audit {audits}/{K}  "
          f"holdout flipped positions {flips_holdout}  "
          f"val PPL {ppl_base:.2f} -> {ppl_attached:.2f} ({ppl_delta_pct:+.4f}%)")

    new_target = targets[1]
    mem.values.data[slots[0]] = unit(memory_value(new_target), 0)
    p_edit, audit_edit = predict(fact_ids[0])
    edit_ok = p_edit == new_target
    edit_audit_ok = int(audit_edit.top_slots.reshape(-1, audit_edit.top_slots.shape[-1])[-1][0]) == slots[0]
    mem.remove(slots[2])
    forget_ok = predict(fact_ids[2])[0] == base_pred[2]
    others_broken = []
    others_after = 0
    for i in range(K):
        if i in (0, 2):
            continue
        ok = int(predict(fact_ids[i])[0] == targets[i])
        others_after += ok
        if not ok:
            others_broken.append({
                "index": i,
                "country": countries[i],
                "target_token": inv[targets[i]],
                "got_token": inv.get(int(predict(fact_ids[i])[0]), "?"),
            })
    print(f"edit {'OK' if edit_ok else 'FAIL'} (audit {'OK' if edit_audit_ok else 'FAIL'}), "
          f"forget {'OK' if forget_ok else 'FAIL'}, others intact {others_after}/{K - 2}")
    if others_broken:
        for row in others_broken[:8]:
            print(f"  collateral [{row['index']}] {row['country']} wanted {row['target_token']!r} "
                  f"got {row['got_token']!r}")

    report = {
        "checkpoint": args.checkpoint, "trunk_val_ppl": ck["val_ppl"], "K": K,
        "hook_mode": args.hook_mode,
        "hook_layer_index": args.hook_layer_index,
        "hook_layer_name": (
            None if layer_names is None else layer_names[args.hook_layer_index]
        ),
        "value_mode": (
            "selected_token_jacobian"
            if args.hook_mode == "residual"
            else "head_weight_unit"
        ),
        "calibration_prompts": calibration_prompts if args.hook_mode == "residual" else None,
        "target_mode": args.target_mode,
        **picker_meta,
        "target_freqs": [int(freqs[t]) for t in targets] if freqs is not None else None,
        "boost": boost, "already_predicted": already,
        "boost_validation_flips": boost_validation_flips,
        "key_coherence_text_centered": rho_text,
        "key_coherence_fact_centered": rho_fact,
        "key_coherence_whitened": rho_white,
        "reachable_fraction": reachable_frac,
        "wordlike_expressible": n_wordlike,
        "unflippable_facts": unflippable,
        "unflippable_detail": unflippable_detail,
        "failed_inject": failed_inject,
        "inject_flip": flips / K, "audit_correct": audits / K,
        "holdout_flipped_positions": int(flips_holdout),
        "val_ppl_base": ppl_base, "val_ppl_attached": ppl_attached,
        "val_ppl_delta_pct": ppl_delta_pct,
        "edit_ok": bool(edit_ok), "edit_audit_ok": bool(edit_audit_ok),
        "forget_ok": bool(forget_ok),
        "others_intact_after_edit_forget": others_after / (K - 2),
        "others_broken_after_edit_forget": others_broken,
        "gates": {
            "G2_flip>=0.95": flips / K >= 0.95,
            "G2_audit==1": audits == K,
            "G2_holdout_zero_flips": flips_holdout == 0,
            "G2_ppl_delta<0.1pct": abs(ppl_delta_pct) < 0.1,
            "G2_edit_forget": bool(edit_ok and edit_audit_ok and forget_ok
                                   and others_after == K - 2),
        },
    }
    report["all_gates_pass"] = all(report["gates"].values())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("=" * 72)
    print(f"gates: {report['gates']}")
    print(f"ALL GATES {'PASS' if report['all_gates_pass'] else 'FAIL'} -> {args.out}")


if __name__ == "__main__":
    main()

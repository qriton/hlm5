"""Phase A: HLM5 reachability-certificate features on CounterFact records (gpt2-xl).

Replicates conventions of scripts/run_1b_certificate.py (EPS, risk labels,
hard-blocker clause) and scripts/run_1b_betastar.py (beta* grid over the
capped interval). Per record:
  h = post-ln_f last hidden at final prompt token, W = tied lm_head weight,
  y = first BPE token of " "+target_new, v = unit(W_y),
  a_j = (W_y - W_j)^T h, b_j = (W_y - W_j)^T v over the FULL vocab.
Features: L, U, slack, risk in {safe,narrow,brittle,unreachable}, hard-blocker
flag + token, worst-case margin at beta* (grid), head_reachable(y),
base argmax, already_correct, single_token_target.

Data: scripts/data/counterfact.json (fetch with scripts/fetch_counterfact.py).

Run: python scripts/run_cert_counterfact_gpt2xl.py
Out: results/cert_counterfact_gpt2xl.jsonl (incremental, resumable)
     results/cert_counterfact_gpt2xl_summary.json
"""
import json
import time
from pathlib import Path

try:
    import torchaudio  # noqa: F401
except Exception:
    import sys
    sys.modules["torchaudio"] = None  # keep transformers from importing a broken torchaudio

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from hlm5.io import RESULTS_DIR

DATA = Path(__file__).resolve().parent / "data" / "counterfact.json"
OUT = RESULTS_DIR / "cert_counterfact_gpt2xl.jsonl"
SUMMARY = RESULTS_DIR / "cert_counterfact_gpt2xl_summary.json"
N_TARGET = 1000
EPS = 1e-4          # same as run_1b_certificate.py
GRID_N = 120        # same as run_1b_betastar.py


def load_model(dev):
    """fp32 first (fits 21GB); on OOM retry once after 60s; then fp16."""
    def _load(dt):
        m = GPT2LMHeadModel.from_pretrained("gpt2-xl", dtype=dt).to(dev).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m
    try:
        return _load(torch.float32), "fp32"
    except torch.cuda.OutOfMemoryError:
        print("OOM loading fp32; retrying in 60s ...", flush=True)
        torch.cuda.empty_cache(); time.sleep(60)
        try:
            return _load(torch.float32), "fp32"
        except torch.cuda.OutOfMemoryError:
            print("OOM again; falling back to fp16.", flush=True)
            torch.cuda.empty_cache()
            return _load(torch.float16), "fp16"


def main():
    torch.manual_seed(0)

    if not DATA.exists():
        raise FileNotFoundError(
            f"{DATA} not found. Run `python scripts/fetch_counterfact.py` first "
            "to download CounterFact.")

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("loading gpt2-xl ...", flush=True)
    tok = AutoTokenizer.from_pretrained("gpt2-xl")
    model, dtype = load_model(dev)
    print(f"gpt2-xl loaded on {dev} in {dtype}", flush=True)

    W = model.lm_head.weight.detach().float()   # (V, d) tied head, fp32 for cert math
    V, d = W.shape
    Wn = W.norm(dim=1)
    print(f"head: vocab {V}, dim {d}", flush=True)

    records = json.load(open(DATA, encoding="utf-8"))
    print(f"counterfact records: {len(records)}", flush=True)

    # ---- resume support: keep only valid existing lines, remember case_ids ----
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if OUT.exists():
        valid = []
        for line in open(OUT, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done[row["case_id"]] = row
                valid.append(line)
            except json.JSONDecodeError:
                print("dropping malformed trailing line in existing output", flush=True)
        with open(OUT, "w", encoding="utf-8") as f:
            f.write("\n".join(valid) + ("\n" if valid else ""))
        print(f"resuming: {len(done)} records already present", flush=True)

    @torch.no_grad()
    def forward_last(prompt):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        out = model(ids, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1].float()       # post-ln_f last hidden
        logits_last = out.logits[0, -1].float()
        return h, logits_last

    @torch.no_grad()
    def certificate(base, y):
        """base = W @ h (V,), y = target token id. Conventions of run_1b_certificate.py."""
        a = base[y] - base                              # a_j (a[y]=0)
        v = W[y] / (Wn[y] + 1e-8)
        Wv = W @ v
        b = Wv[y] - Wv                                  # b_j (b[y]=0)
        mask = torch.ones(V, dtype=torch.bool, device=dev); mask[y] = False
        aj, bj = a[mask], b[mask]
        idx = torch.arange(V, device=dev)[mask]
        bpos, bneg, bzero = bj > EPS, bj < -EPS, bj.abs() <= EPS
        hard = (bzero | (bj < 0)) & (aj <= 0)           # hard blocker: a_j<=0 AND b_j<=EPS
        ratio = -aj / bj
        L = torch.clamp(ratio[bpos].max(), min=0.0) if bpos.any() else torch.tensor(0.0, device=dev)
        U = ratio[bneg].min() if bneg.any() else torch.tensor(float("inf"), device=dev)
        reach = (not bool(hard.any())) and float(L) < float(U)

        out = {
            "reachable": reach,
            "L": float(L),
            "U": float(U) if float(U) != float("inf") else None,
            "slack": float(U - L) if float(U) != float("inf") else None,
            "hard_blocker": bool(hard.any()),
            "blocker_token_id": None, "blocker_token_str": None,
            "blocker_a": None, "blocker_b": None,
            "beta_heur": None, "margin_after_heur": None,
            "beta_star": None, "margin_at_beta_star": None,
        }

        if reach:
            beta = float(L) * 1.05 + 1.0
            marg = aj + beta * bj
            jb = int(idx[marg.argmin()])
            out["beta_heur"] = beta
            out["margin_after_heur"] = float(marg.min())
            out["blocker_token_id"] = jb
            out["blocker_token_str"] = tok.decode([jb])
            s = float(U - L)
            out["risk"] = "safe" if s > 5 * beta else ("narrow" if s > beta else "brittle")
            # beta* grid over capped interval (run_1b_betastar.py conventions)
            Lf, Uf = float(L), float(U)
            hi = min(Uf, Lf + max(50.0, 5.0 * Lf)) if Uf != float("inf") else Lf + max(50.0, 5.0 * Lf)
            grid = Lf + torch.linspace(0, 1, GRID_N, device=dev) * (hi - Lf)
            margs = aj[:, None] + grid[None, :] * bj[:, None]     # (V-1, GRID_N)
            worst = margs.min(0).values
            k = int(worst.argmax())
            out["beta_star"] = float(grid[k])
            out["margin_at_beta_star"] = float(worst[k])
        else:
            out["risk"] = "unreachable"
            if bool(hard.any()):
                hard_idx = idx[hard]
                jb = int(hard_idx[base[hard_idx].argmax()])       # worst hard blocker by base logit
                out["blocker_token_id"] = jb
                out["blocker_token_str"] = tok.decode([jb])
                out["blocker_a"] = float(base[y] - base[jb])
                out["blocker_b"] = float(Wv[y] - Wv[jb])
        return out

    @torch.no_grad()
    def head_reachable(y):
        return bool(int((W @ W[y]).argmax()) == y)

    n_done_new = 0
    usable_count = 0
    identity_checked = False
    identity_info = None
    t0 = time.time()
    fout = open(OUT, "a", encoding="utf-8")

    for rec in records:
        if usable_count >= N_TARGET:
            break
        rr = rec["requested_rewrite"]
        enc_new = tok.encode(" " + rr["target_new"]["str"])
        enc_true = tok.encode(" " + rr["target_true"]["str"])
        if len(enc_new) < 1 or len(enc_true) < 1:
            continue                                    # target tokens not well-defined
        usable_idx = usable_count
        usable_count += 1
        cid = rec["case_id"]
        if cid in done:
            continue                                    # already computed (resume)

        y = enc_new[0]
        y_true = enc_true[0]
        prompt = rr["prompt"].format(rr["subject"])
        h, logits_last = forward_last(prompt)
        base = W @ h

        if not identity_checked:
            diff = float((base - logits_last).abs().max())
            same_argmax = bool(int(base.argmax()) == int(logits_last.argmax()))
            identity_info = {"max_abs_diff_lmhead_vs_logits": diff, "argmax_match": same_argmax,
                             "case_id": cid, "dtype": dtype}
            print(f"[identity check, first record] max|W@h - model logits| = {diff:.3e}, "
                  f"argmax match = {same_argmax} (h = hidden_states[-1] post-ln_f, {dtype})", flush=True)
            assert same_argmax, "logit identity check failed: h is not the pre-head hidden state"
            identity_checked = True

        cert = certificate(base, y)
        base_arg = int(base.argmax())
        row = {
            "case_id": cid,
            "usable_index": usable_idx,
            "subject": rr["subject"],
            "prompt": prompt,
            "target_new": rr["target_new"]["str"],
            "target_true": rr["target_true"]["str"],
            "y_token_id": int(y),
            "y_token_str": tok.decode([y]),
            "y_true_token_id": int(y_true),
            "y_true_token_str": tok.decode([y_true]),
            **cert,
            "head_reachable": head_reachable(y),
            "base_argmax_id": base_arg,
            "base_argmax_str": tok.decode([base_arg]),
            "already_correct": bool(base_arg == y),
            "single_token_target": bool(len(enc_new) == 1),
        }
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        fout.flush()
        n_done_new += 1
        if usable_count % 100 == 0:
            print(f"  {usable_count}/{N_TARGET} usable processed "
                  f"({n_done_new} new, {time.time()-t0:.0f}s)", flush=True)

    fout.close()
    print(f"done: {usable_count} usable records, {n_done_new} newly computed, "
          f"{time.time()-t0:.0f}s", flush=True)

    # ---- summary over the full output file ----
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: r["usable_index"])
    n = len(rows)
    label_counts = {}
    for r in rows:
        label_counts[r["risk"]] = label_counts.get(r["risk"], 0) + 1
    finite_slacks = sorted(r["slack"] for r in rows if r["slack"] is not None)
    reach_slacks = sorted(r["slack"] for r in rows if r["reachable"] and r["slack"] is not None)

    def med(x):
        return x[len(x) // 2] if x else None

    summary = {
        "model": "gpt2-xl", "dtype": dtype, "device": dev,
        "n_records": n,
        "identity_check_first_record": identity_info if identity_info is not None else "resumed; checked in earlier run",
        "risk_label_counts": label_counts,
        "fraction_reachable": round(sum(r["reachable"] for r in rows) / n, 4),
        "fraction_hard_blocker": round(sum(r["hard_blocker"] for r in rows) / n, 4),
        "fraction_infinite_slack": round(sum(r["slack"] is None for r in rows) / n, 4),
        "median_slack_finite_all": round(med(finite_slacks), 4) if finite_slacks else None,
        "median_slack_reachable_finite": round(med(reach_slacks), 4) if reach_slacks else None,
        "fraction_head_reachable": round(sum(r["head_reachable"] for r in rows) / n, 4),
        "fraction_already_correct": round(sum(r["already_correct"] for r in rows) / n, 4),
        "fraction_single_token_target": round(sum(r["single_token_target"] for r in rows) / n, 4),
        "case_id_first": rows[0]["case_id"] if rows else None,
        "case_id_last": rows[-1]["case_id"] if rows else None,
    }
    json.dump(summary, open(SUMMARY, "w", encoding="utf-8"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved -> {OUT}\nsaved -> {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()

"""Phase A: HLM5 reachability-certificate features on CounterFact records (gpt2-xl).

Replicates conventions of scripts/run_1b_certificate.py (exact signs, risk labels,
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
Out: results/certificate_refresh_staging/cert_counterfact_gpt2xl.jsonl
     results/certificate_refresh_staging/cert_counterfact_gpt2xl_summary.json
"""
# ruff: noqa: E402 -- direct script execution bootstraps the repository root.
import json
import sys
import time
from pathlib import Path

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

try:
    import torchaudio  # noqa: F401
except Exception:
    sys.modules["torchaudio"] = None  # keep transformers from importing a broken torchaudio

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from hlm5.artifact_contract import (
    COUNTERFACT_SAMPLE_CONTRACT,
    COUNTERFACT_PATH,
    GPT2_XL_MODEL_ID,
    GPT2_XL_REVISION,
    STAGING_DIR,
    atomic_write_json,
    atomic_write_jsonl,
    counterfact_contract,
    float64_boundary_record,
    load_contract_jsonl,
    require_preflight,
    sha256_file,
    stable_json_sha256,
)

DATA = COUNTERFACT_PATH
OUT = STAGING_DIR / "cert_counterfact_gpt2xl.jsonl"
SUMMARY = STAGING_DIR / "cert_counterfact_gpt2xl_summary.json"
N_TARGET = 1000
CHECKPOINT_EVERY = 10
EPS = 0.0           # exact certificate: no slope dead-zone
CERTIFICATE_ARITHMETIC_DTYPE = "float64"
GRID_N = 120        # same as run_1b_betastar.py
IDENTITY_RTOL = 1e-5
IDENTITY_ATOL = 2e-5


def load_model(dev):
    """Load the registered local GPT-2 XL revision in float32 only."""
    model = GPT2LMHeadModel.from_pretrained(
        GPT2_XL_MODEL_ID,
        revision=GPT2_XL_REVISION,
        dtype=torch.float32,
        local_files_only=True,
    ).to(dev).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main():
    torch.manual_seed(0)
    _, preflight_sha256 = require_preflight(__file__, "counterfact")
    contract = counterfact_contract(
        __file__,
        preflight_sha256,
        COUNTERFACT_SAMPLE_CONTRACT,
    )
    contract_sha256 = stable_json_sha256(contract)

    if not DATA.exists():
        raise FileNotFoundError(
            f"{DATA} not found. Run `python scripts/fetch_counterfact.py` first "
            "to download CounterFact.")

    resume_rows = load_contract_jsonl(
        OUT,
        contract_sha256,
        require_prefix=True,
    )
    if any(
        not isinstance(row.get("certificate_boundary_dtypes"), dict)
        or not row["certificate_boundary_dtypes"]
        or any(
            dtype != "float64"
            for dtype in row["certificate_boundary_dtypes"].values()
        )
        for row in resume_rows
    ):
        raise RuntimeError("staged resume row has a non-float64 certificate boundary")
    print(f"resume prefix: {len(resume_rows)} records", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("loading gpt2-xl ...", flush=True)
    tok = AutoTokenizer.from_pretrained(
        GPT2_XL_MODEL_ID,
        revision=GPT2_XL_REVISION,
        local_files_only=True,
    )
    model = load_model(dev)
    dtype = "float32"
    print(f"gpt2-xl loaded on {dev} in {dtype}", flush=True)

    W_native = model.lm_head.weight.detach()
    W = W_native.to(torch.float64)
    V, d = W.shape
    Wn = W.norm(dim=1)
    print(f"head: vocab {V}, dim {d}", flush=True)

    records = json.load(open(DATA, encoding="utf-8"))
    print(f"counterfact records: {len(records)}", flush=True)

    @torch.no_grad()
    def forward_last(prompt):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        out = model(ids, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1].detach()       # native post-ln_f hidden
        logits_last = out.logits[0, -1].detach()
        return h, logits_last

    @torch.no_grad()
    def certificate(base, y):
        """base = W @ h (V,), y = target token id. Conventions of run_1b_certificate.py."""
        a = base[y] - base                              # a_j (a[y]=0)
        v = W[y] / (Wn[y] + 1e-12)
        Wv = W @ v
        b = Wv[y] - Wv                                  # b_j (b[y]=0)
        mask = torch.ones(V, dtype=torch.bool, device=dev)
        mask[y] = False
        aj, bj = a[mask], b[mask]
        idx = torch.arange(V, device=dev)[mask]
        bpos, bneg = bj > EPS, bj < 0
        hard = (bj <= EPS) & (aj <= 0)                  # hard blocker: a_j<=0 AND b_j<=0
        ratio = -aj / bj
        L = torch.clamp(ratio[bpos].max(), min=0.0) if bpos.any() else torch.tensor(0.0, device=dev, dtype=W.dtype)
        U = ratio[bneg].min() if bneg.any() else torch.tensor(float("inf"), device=dev, dtype=W.dtype)
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

        grid = None
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
            grid = Lf + torch.linspace(
                0,
                1,
                GRID_N + 2,
                device=dev,
                dtype=W.dtype,
            )[1:-1] * (hi - Lf)
            margs = aj[:, None] + grid[None, :] * bj[:, None]     # (V-1, GRID_N)
            worst = margs.min(0).values
            k = int(worst.argmax())
            if float(worst[k]) > 0:
                out["beta_star"] = float(grid[k])
                out["margin_at_beta_star"] = float(worst[k])
            else:
                raise RuntimeError(
                    "dose grid found no positive-margin point inside a "
                    "reachable certificate interval"
                )
        else:
            out["risk"] = "unreachable"
            if bool(hard.any()):
                hard_idx = idx[hard]
                jb = int(hard_idx[base[hard_idx].argmax()])       # worst hard blocker by base logit
                out["blocker_token_id"] = jb
                out["blocker_token_str"] = tok.decode([jb])
                out["blocker_a"] = float(base[y] - base[jb])
                out["blocker_b"] = float(Wv[y] - Wv[jb])
        boundary_tensors = {
            "head": W,
            "base_logits": base,
            "direction": v,
            "ratios": ratio,
        }
        if grid is not None:
            boundary_tensors["grid"] = grid
        return out, float64_boundary_record(**boundary_tensors)

    @torch.no_grad()
    def head_reachable(y):
        return bool(int((W @ W[y]).argmax()) == y)

    n_done_new = 0
    usable_count = 0
    identity_info = None
    t0 = time.time()
    rows = list(resume_rows)

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
        y = enc_new[0]
        y_true = enc_true[0]
        prompt = rr["prompt"].format(rr["subject"])
        if usable_idx < len(resume_rows):
            retained = resume_rows[usable_idx]
            expected_identity = {
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
                "single_token_target": bool(len(enc_new) == 1),
            }
            mismatches = {
                key: {"retained": retained.get(key), "expected": value}
                for key, value in expected_identity.items()
                if retained.get(key) != value
            }
            if mismatches:
                raise RuntimeError(
                    "staged resume prefix does not match the registered data "
                    f"at usable index {usable_idx}: {mismatches}"
                )
            if usable_idx != 0:
                continue

        h_native, logits_last = forward_last(prompt)
        if usable_idx == 0:
            base_native = W_native @ h_native
            diff = float((base_native - logits_last).abs().max())
            same_argmax = bool(
                int(base_native.argmax()) == int(logits_last.argmax())
            )
            identity_allclose = bool(
                torch.allclose(
                    base_native,
                    logits_last,
                    rtol=IDENTITY_RTOL,
                    atol=IDENTITY_ATOL,
                )
            )
            identity_info = {
                "max_abs_diff_lmhead_vs_logits": diff,
                "argmax_match": same_argmax,
                "allclose": identity_allclose,
                "rtol": IDENTITY_RTOL,
                "atol": IDENTITY_ATOL,
                "case_id": cid,
                "dtype": dtype,
            }
            print(
                f"[identity check, first record] max|W@h - model logits| = "
                f"{diff:.3e}, allclose = {identity_allclose}, argmax match = "
                f"{same_argmax} (h = hidden_states[-1] post-ln_f, {dtype})",
                flush=True,
            )
            if not same_argmax or not identity_allclose:
                raise RuntimeError(
                    "logit identity check failed: h is not the pre-head hidden state"
                )
            if resume_rows:
                continue

        h = h_native.to(torch.float64)
        base = W @ h

        cert, boundary = certificate(base, y)
        boundary.update(float64_boundary_record(hidden=h))
        if cert["reachable"]:
            beta_star = cert["beta_star"]
            upper = cert["U"]
            if (
                beta_star is None
                or not cert["L"] < beta_star
                or (upper is not None and not beta_star < upper)
                or not cert["margin_at_beta_star"] > 0.0
            ):
                raise RuntimeError(
                    f"invalid CounterFact certified dose at usable index {usable_idx}"
                )
        base_arg = int(logits_last.argmax())
        row = {
            "certificate_contract_sha256": contract_sha256,
            "certificate_boundary_dtypes": boundary,
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
        rows.append(row)
        n_done_new += 1
        if n_done_new % CHECKPOINT_EVERY == 0:
            atomic_write_jsonl(OUT, rows)
        if usable_count % 100 == 0:
            print(f"  {usable_count}/{N_TARGET} usable processed "
                  f"({n_done_new} new, {time.time()-t0:.0f}s)", flush=True)

    atomic_write_jsonl(OUT, rows)
    print(f"done: {usable_count} usable records, {n_done_new} newly computed, "
          f"{time.time()-t0:.0f}s", flush=True)

    # ---- summary over the full output file ----
    rows = load_contract_jsonl(
        OUT,
        contract_sha256,
        require_prefix=True,
    )
    n = len(rows)
    if n != N_TARGET:
        raise RuntimeError(f"incomplete CounterFact staging output: {n}/{N_TARGET}")
    if [row["usable_index"] for row in rows] != list(range(N_TARGET)):
        raise RuntimeError("CounterFact usable indices are not exactly 0..999")
    if len({row["case_id"] for row in rows}) != N_TARGET:
        raise RuntimeError("CounterFact case IDs are not unique")
    if any(
        row.get("certificate_contract_sha256") != contract_sha256 for row in rows
    ):
        raise RuntimeError("CounterFact row contract hash mismatch after completion")
    if any(
        not isinstance(row.get("certificate_boundary_dtypes"), dict)
        or not row["certificate_boundary_dtypes"]
        or any(
            dtype != "float64"
            for dtype in row["certificate_boundary_dtypes"].values()
        )
        for row in rows
    ):
        raise RuntimeError("CounterFact row certificate boundary is not float64")
    if (
        identity_info is None
        or identity_info.get("argmax_match") is not True
        or identity_info.get("allclose") is not True
    ):
        raise RuntimeError("missing or failed first-record native identity check")
    atomic_write_jsonl(OUT, rows)
    jsonl_sha256 = sha256_file(OUT)
    label_counts = {}
    for r in rows:
        label_counts[r["risk"]] = label_counts.get(r["risk"], 0) + 1
    finite_slacks = sorted(r["slack"] for r in rows if r["slack"] is not None)
    reach_slacks = sorted(r["slack"] for r in rows if r["reachable"] and r["slack"] is not None)

    def med(x):
        return x[len(x) // 2] if x else None

    summary_contract = {**contract, "jsonl_sha256": jsonl_sha256}
    summary = {
        "certificate_contract": summary_contract,
        "certificate_boundary_dtypes": rows[0]["certificate_boundary_dtypes"],
        "model": "gpt2-xl", "dtype": dtype, "device": dev,
        "n_records": n,
        "identity_check_first_record": identity_info,
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
    atomic_write_json(SUMMARY, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)
    print(f"saved -> {OUT}\nsaved -> {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()

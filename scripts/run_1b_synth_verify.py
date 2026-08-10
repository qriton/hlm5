"""End-to-end check of residual synthesis: does r* ACTUALLY flip the token at the
key, with a usable beta? (m*>0 only proves a direction exists.) For unreachable
targets (under v=unit(W_y)) at a novel key, synthesize r*, then search for the
smallest beta with argmax(head(h + beta r*)) == target. Reports the real flip rate
and the beta magnitude needed.

Run: python scripts/run_1b_synth_verify.py
Out: results/certificate_refresh_staging/synth_verify.json
"""
# ruff: noqa: E402 -- direct script execution bootstraps the repository root.
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

import torch

from hlm5.artifact_contract import (
    SYNTHESIS_FLIP_SAMPLE_CONTRACT,
    STAGING_DIR,
    atomic_write_json,
    float64_boundary_record,
    hlm5_contract,
    require_preflight,
)
from hlm5.io import load_trunk, load_tokenizer


EPS = 0.0
CERTIFICATE_ARITHMETIC_DTYPE = "float64"
OUTPUT = STAGING_DIR / "synth_verify.json"


def native_dosed_argmax(W_native, h_native, direction, beta):
    """Score a float64 certificate direction at the native injection boundary."""
    injected = (beta * direction).to(h_native.dtype)
    return int((W_native @ (h_native + injected)).argmax())


def main():
    random.seed(0)
    torch.manual_seed(0)
    _, preflight_sha256 = require_preflight(__file__, "hlm5")

    TOK = load_tokenizer()
    model, ck = load_trunk("baseline")
    DEV = next(model.parameters()).device.type
    model_dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    contract = hlm5_contract(
        __file__,
        preflight_sha256,
        model_dtype,
        SYNTHESIS_FLIP_SAMPLE_CONTRACT,
    )
    for p in model.parameters():
        p.requires_grad_(False)
    W_native = model.head.weight.detach()
    W = W_native.to(torch.float64)
    V, d = W.shape
    Wn = W.norm(dim=1)

    @torch.no_grad()
    def hidlast_native(t):
        return model.hidden(
            torch.tensor([TOK.encode(t).ids], device=DEV)
        )[0, -1].detach()

    h_native = hidlast_native("The capital of Vorenia is")
    h = h_native.to(torch.float64)
    base = W @ h
    pool = []
    for s, i in TOK.get_vocab().items():
        t = s.replace("Ġ", " ")
        if t.startswith(" ") and re.fullmatch(r"[A-Za-z]{3,}", t.strip()):
            pool.append(i)
    pool = sorted(pool)[:1200]
    def is_unreachable(tid):
        a = base[tid] - base
        v = W[tid] / (Wn[tid] + 1e-12)
        Wv = W @ v
        b = Wv[tid] - Wv
        m = torch.ones(V, dtype=torch.bool, device=DEV)
        m[tid] = False
        aj, bj = a[m], b[m]
        hard = (bj <= EPS) & (aj <= 0)
        bpos, bneg = bj > EPS, bj < 0
        L = torch.clamp(((-aj / bj)[bpos]).max(), min=0.0) if bpos.any() else torch.tensor(0.0, device=DEV, dtype=W.dtype)
        U = ((-aj / bj)[bneg]).min() if bneg.any() else torch.tensor(float("inf"), device=DEV, dtype=W.dtype)
        return bool(hard.any()) or float(L) >= float(U)

    def synth(tid, steps=300):
        r = (W[tid] / (Wn[tid] + 1e-12)).clone()
        tau = 8.0
        for _ in range(steps):
            sc = W @ r
            sc[tid] = -1e9
            w = torch.softmax(tau * (sc - sc.max()), 0)
            grad = W[tid] - (w[:, None] * W).sum(0)
            r = r + 0.5 * grad
            r = r / (r.norm() + 1e-12)
            tau = min(40.0, tau * 1.01)
        scores = W @ r
        competitors = scores.clone()
        competitors[tid] = -1e9
        return r, float(scores[tid] - competitors.max())

    BETAS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000]
    unreachable = [t for t in pool if is_unreachable(t)][:40]
    flipped_emb = flipped_synth = 0
    betas_used = []
    synthesis_margins = []
    first_synth_direction = None
    case_rows = []
    with torch.no_grad():
        for tid in unreachable:
            v = W[tid] / (Wn[tid] + 1e-12)
            # baseline value unit(W_y): can any beta flip?
            fe = any(
                native_dosed_argmax(W_native, h_native, v, bb) == tid
                for bb in BETAS
            )
            flipped_emb += int(fe)
            # synthesized residual
            r, synthesis_margin = synth(tid)
            synthesis_margins.append(synthesis_margin)
            if first_synth_direction is None:
                first_synth_direction = r
            bmin = None
            for bb in BETAS:
                if native_dosed_argmax(W_native, h_native, r, bb) == tid:
                    bmin = bb
                    break
            if bmin is not None:
                flipped_synth += 1
                betas_used.append(bmin)
            case_rows.append(
                {
                    "target_id": tid,
                    "unit_value_flipped": fe,
                    "synthesis_min_margin": synthesis_margin,
                    "synth_value_flipped": bmin is not None,
                    "first_flip_beta": bmin,
                }
            )

    n = len(unreachable)
    if first_synth_direction is None:
        raise RuntimeError("registered synthesis sample is empty")
    betas_used_sorted = sorted(betas_used)
    out = {"certificate_contract": contract,
           "certificate_boundary_dtypes": float64_boundary_record(
               head=W,
               hidden=h,
               direction=first_synth_direction,
           ),
           "model": "HLM5-1B-trunk", "n_unreachable_tested": n,
           "flip_with_unit_Wy": flipped_emb,
           "flip_with_synth_residual": flipped_synth,
           "rescue_flip_rate": round(flipped_synth / n, 3) if n else None,
           "median_beta_needed": betas_used_sorted[len(betas_used_sorted) // 2] if betas_used_sorted else None,
           "max_beta_needed": max(betas_used) if betas_used else None,
           "min_beta_needed": min(betas_used) if betas_used else None,
           "per_target": case_rows,
           "validity": {
               "synthesis_directions_checked": len(synthesis_margins),
               "all_synthesis_margins_positive": all(
                   margin > 0.0 for margin in synthesis_margins
               ),
               "minimum_synthesis_margin": min(synthesis_margins),
           }}
    atomic_write_json(OUTPUT, out)
    print(json.dumps(out, indent=2, sort_keys=True, allow_nan=False))
    print(f"saved -> {OUTPUT}")


if __name__ == "__main__":
    main()

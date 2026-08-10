"""beta* = argmax_{beta in [L,U)} min_j (a_j + beta b_j): the principled deployment
strength that maximizes the worst-case competitor margin (1-D concave piecewise-
linear, solved by scanning). Compares the worst-case margin M(beta*) to the
heuristic beta = 1.05*L + 1 used elsewhere, on reachable targets at a novel key.

Run: python scripts/run_1b_betastar.py
Out: results/certificate_refresh_staging/betastar.json
"""
# ruff: noqa: E402 -- direct script execution bootstraps the repository root.
import json
import math
import random
import re
import sys
from pathlib import Path

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

import torch

from hlm5.artifact_contract import (
    BETA_STAR_SAMPLE_CONTRACT,
    STAGING_DIR,
    atomic_write_json,
    float64_boundary_record,
    hlm5_contract,
    require_preflight,
)
from hlm5.io import load_trunk, load_tokenizer


EPS = 0.0
CERTIFICATE_ARITHMETIC_DTYPE = "float64"
OUTPUT = STAGING_DIR / "betastar.json"


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
        BETA_STAR_SAMPLE_CONTRACT,
    )
    W = model.head.weight.detach().to(torch.float64)
    V, d = W.shape
    Wn = W.norm(dim=1)

    @torch.no_grad()
    def hidlast(text):
        return model.hidden(
            torch.tensor([TOK.encode(text).ids], device=DEV)
        )[0, -1].detach().to(torch.float64)

    h = hidlast("The capital of Vorenia is")
    base = W @ h
    pool = []
    for s, i in TOK.get_vocab().items():
        t = s.replace("Ġ", " ")
        if t.startswith(" ") and re.fullmatch(r"[A-Za-z]{3,}", t.strip()):
            pool.append(i)
    pool = sorted(pool)[:1200]
    betas = torch.linspace(0, 1, 122, device=DEV, dtype=W.dtype)[1:-1]

    mstar, mheur, bstar = [], [], []
    target_rows = []
    inside_open_interval = []
    envelope_reachable = 0
    grid_failures = 0
    with torch.no_grad():
        for tid in pool:
            a = base[tid] - base
            v = W[tid] / (Wn[tid] + 1e-12)
            Wv = W @ v
            b = Wv[tid] - Wv
            m = torch.ones(V, dtype=torch.bool, device=DEV)
            m[tid] = False
            aj, bj = a[m], b[m]
            bpos, bneg = bj > EPS, bj < 0
            L = torch.clamp(((-aj / bj)[bpos]).max(), min=0.0) if bpos.any() else torch.tensor(0.0, device=DEV, dtype=W.dtype)
            U = ((-aj / bj)[bneg]).min() if bneg.any() else torch.tensor(float("inf"), device=DEV, dtype=W.dtype)
            hard = (bj <= EPS) & (aj <= 0)
            if bool(hard.any()) or float(L) >= float(U):
                continue
            envelope_reachable += 1
            Lf, Uf = float(L), float(U)
            hi = min(Uf, Lf + max(50.0, 5.0 * Lf)) if Uf != float("inf") else Lf + max(50.0, 5.0 * Lf)
            grid = Lf + betas * (hi - Lf)                      # (120,)
            marg = aj[:, None] + grid[None, :] * bj[:, None]   # (V-1,120)
            worst = marg.min(0).values                         # (120,)
            k = int(worst.argmax())
            if float(worst[k]) <= 0:
                grid_failures += 1
                continue
            chosen = float(grid[k])
            bstar.append(chosen)
            chosen_margin = float(worst[k])
            mstar.append(chosen_margin)
            inside_open_interval.append(
                Lf < chosen and (not math.isfinite(Uf) or chosen < Uf)
            )
            bh = 1.05 * Lf + 1.0
            heuristic_margin = float((aj + bh * bj).min())
            mheur.append(heuristic_margin)
            target_rows.append(
                {
                    "target_id": tid,
                    "L": Lf,
                    "U": Uf if math.isfinite(Uf) else None,
                    "beta_star": chosen,
                    "worst_margin_beta_star": chosen_margin,
                    "heuristic_beta": bh,
                    "worst_margin_heuristic": heuristic_margin,
                }
            )

    def med(x):
        x = sorted(x)
        return round(x[len(x) // 2], 3) if x else None
    if (
        grid_failures
        or len(bstar) != envelope_reachable
        or not all(inside_open_interval)
        or not all(margin > 0.0 for margin in mstar)
    ):
        raise RuntimeError("beta-star producer emitted an invalid certified dose")
    sample_tid = pool[0]
    sample_direction = W[sample_tid] / (Wn[sample_tid] + 1e-12)
    sample_a = base[sample_tid] - base
    sample_wv = W @ sample_direction
    sample_ratios = -sample_a / (sample_wv[sample_tid] - sample_wv)
    out = {"certificate_contract": contract,
           "certificate_boundary_dtypes": float64_boundary_record(
               head=W,
               hidden=h,
               direction=sample_direction,
               ratios=sample_ratios,
               grid=betas,
           ),
           "model": "HLM5-1B-trunk", "n_reachable": len(bstar),
           "median_beta_star": med(bstar),
           "median_worst_margin_betastar": med(mstar),
           "median_worst_margin_heuristic": med(mheur),
           "median_margin_gain": round((med(mstar) or 0) - (med(mheur) or 0), 3),
           "per_target": target_rows,
           "validity": {
               "candidate_doses_checked": len(bstar),
               "envelope_reachable": envelope_reachable,
               "grid_failures": grid_failures,
               "all_candidate_doses_inside_open_interval": all(
                   inside_open_interval
               ),
               "all_candidate_margins_positive": all(
                   margin > 0.0 for margin in mstar
               ),
               "minimum_candidate_margin": min(mstar),
           }}
    atomic_write_json(OUTPUT, out)
    print(json.dumps(out, indent=2, sort_keys=True, allow_nan=False))
    print(f"saved -> {OUTPUT}")


if __name__ == "__main__":
    main()

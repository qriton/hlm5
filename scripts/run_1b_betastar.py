"""beta* = argmax_{beta in [L,U)} min_j (a_j + beta b_j): the principled deployment
strength that maximizes the worst-case competitor margin (1-D concave piecewise-
linear, solved by scanning). Compares the worst-case margin M(beta*) to the
heuristic beta = 1.05*L + 1 used elsewhere, on reachable targets at a novel key.

Run: python scripts/run_1b_betastar.py
Out: results/betastar.json
"""
import json
import random
import re

import torch

from hlm5.io import load_trunk, load_tokenizer, RESULTS_DIR


def main():
    random.seed(0)
    torch.manual_seed(0)

    TOK = load_tokenizer()
    model, ck = load_trunk("baseline")
    DEV = next(model.parameters()).device.type
    W = model.head.weight.detach().float(); V, d = W.shape
    Wn = W.norm(dim=1)

    @torch.no_grad()
    def hidlast(text):
        return model.hidden(torch.tensor([TOK.encode(text).ids], device=DEV))[0, -1].float()

    h = hidlast("The capital of Vorenia is")
    base = W @ h
    pool = []
    for s, i in TOK.get_vocab().items():
        t = s.replace("Ġ", " ")
        if t.startswith(" ") and re.fullmatch(r"[A-Za-z]{3,}", t.strip()):
            pool.append(i)
    pool = sorted(pool)[:1200]
    EPS = 1e-4
    betas = torch.linspace(0, 1, 120, device=DEV)

    mstar, mheur, bstar = [], [], []
    with torch.no_grad():
        for tid in pool:
            a = base[tid] - base
            v = W[tid] / (Wn[tid] + 1e-8)
            Wv = W @ v
            b = Wv[tid] - Wv
            m = torch.ones(V, dtype=torch.bool, device=DEV); m[tid] = False
            aj, bj = a[m], b[m]
            bpos, bneg = bj > EPS, bj < -EPS
            L = torch.clamp(((-aj / bj)[bpos]).max(), min=0.0) if bpos.any() else torch.tensor(0.0, device=DEV)
            U = ((-aj / bj)[bneg]).min() if bneg.any() else torch.tensor(float("inf"), device=DEV)
            hard = ((bj.abs() <= EPS) | (bj < 0)) & (aj <= 0)
            if bool(hard.any()) or float(L) >= float(U):
                continue
            Lf, Uf = float(L), float(U)
            hi = min(Uf, Lf + max(50.0, 5.0 * Lf)) if Uf != float("inf") else Lf + max(50.0, 5.0 * Lf)
            grid = Lf + betas * (hi - Lf)                      # (120,)
            marg = aj[:, None] + grid[None, :] * bj[:, None]   # (V-1,120)
            worst = marg.min(0).values                         # (120,)
            k = int(worst.argmax())
            bstar.append(float(grid[k])); mstar.append(float(worst[k]))
            bh = 1.05 * Lf + 1.0
            mheur.append(float((aj + bh * bj).min()))

    def med(x):
        x = sorted(x); return round(x[len(x) // 2], 3) if x else None
    out = {"model": "HLM5-1B-trunk", "n_reachable": len(bstar),
           "median_beta_star": med(bstar),
           "median_worst_margin_betastar": med(mstar),
           "median_worst_margin_heuristic": med(mheur),
           "median_margin_gain": round((med(mstar) or 0) - (med(mheur) or 0), 3)}
    out_path = RESULTS_DIR / "betastar.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()

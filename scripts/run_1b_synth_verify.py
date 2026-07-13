"""End-to-end check of residual synthesis: does r* ACTUALLY flip the token at the
key, with a usable beta? (m*>0 only proves a direction exists.) For unreachable
targets (under v=unit(W_y)) at a novel key, synthesize r*, then search for the
smallest beta with argmax(head(h + beta r*)) == target. Reports the real flip rate
and the beta magnitude needed.

Run: python scripts/run_1b_synth_verify.py
Out: results/synth_verify.json
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
    for p in model.parameters():
        p.requires_grad_(False)
    W = model.head.weight.detach().float(); V, d = W.shape
    Wn = W.norm(dim=1)

    @torch.no_grad()
    def hidlast(t):
        return model.hidden(torch.tensor([TOK.encode(t).ids], device=DEV))[0, -1].float()

    h = hidlast("The capital of Vorenia is")
    base = W @ h
    pool = []
    for s, i in TOK.get_vocab().items():
        t = s.replace("Ġ", " ")
        if t.startswith(" ") and re.fullmatch(r"[A-Za-z]{3,}", t.strip()):
            pool.append(i)
    pool = sorted(pool)[:1200]
    EPS = 0.0

    def is_unreachable(tid):
        a = base[tid] - base
        v = W[tid] / (Wn[tid] + 1e-8); Wv = W @ v; b = Wv[tid] - Wv
        m = torch.ones(V, dtype=torch.bool, device=DEV); m[tid] = False
        aj, bj = a[m], b[m]
        hard = (bj <= EPS) & (aj <= 0)
        bpos, bneg = bj > EPS, bj < 0
        L = torch.clamp(((-aj / bj)[bpos]).max(), min=0.0) if bpos.any() else torch.tensor(0.0, device=DEV)
        U = ((-aj / bj)[bneg]).min() if bneg.any() else torch.tensor(float("inf"), device=DEV)
        return bool(hard.any()) or float(L) >= float(U)

    def synth(tid, steps=300):
        r = (W[tid] / (Wn[tid] + 1e-8)).clone()
        tau = 8.0
        for _ in range(steps):
            sc = W @ r; sc[tid] = -1e9
            w = torch.softmax(tau * (sc - sc.max()), 0)
            grad = W[tid] - (w[:, None] * W).sum(0)
            r = r + 0.5 * grad; r = r / (r.norm() + 1e-8)
            tau = min(40.0, tau * 1.01)
        return r

    BETAS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000]
    unreachable = [t for t in pool if is_unreachable(t)][:40]
    flipped_emb = flipped_synth = 0
    betas_used = []
    with torch.no_grad():
        for tid in unreachable:
            v = W[tid] / (Wn[tid] + 1e-8)
            # baseline value unit(W_y): can any beta flip?
            fe = any(int((W @ (h + bb * v)).argmax()) == tid for bb in BETAS)
            flipped_emb += int(fe)
            # synthesized residual
            r = synth(tid)
            bmin = None
            for bb in BETAS:
                if int((W @ (h + bb * r)).argmax()) == tid:
                    bmin = bb; break
            if bmin is not None:
                flipped_synth += 1; betas_used.append(bmin)

    n = len(unreachable)
    betas_used_sorted = sorted(betas_used)
    out = {"model": "HLM5-1B-trunk", "n_unreachable_tested": n,
           "flip_with_unit_Wy": flipped_emb,
           "flip_with_synth_residual": flipped_synth,
           "rescue_flip_rate": round(flipped_synth / n, 3) if n else None,
           "median_beta_needed": betas_used_sorted[len(betas_used_sorted) // 2] if betas_used_sorted else None,
           "max_beta_needed": max(betas_used) if betas_used else None,
           "min_beta_needed": min(betas_used) if betas_used else None}
    out_path = RESULTS_DIR / "synth_verify.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()

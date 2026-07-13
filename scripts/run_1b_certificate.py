"""Certificate envelope + residual synthesis on the frozen 1B trunk.

(1) ENVELOPE: for a fact hidden h and target y, the post-edit margin to competitor
    j is affine in edit strength beta:  m_j = a_j + beta*b_j,  a_j = (W_y-W_j).h,
    b_j = (W_y-W_j).v  with the deployed value v = unit(W_y). Feasible interval:
       L = max(0, max_{b_j>0} -a_j/b_j),  U = min_{b_j<0} -a_j/b_j.
    Reachable iff L<U and no competitor has a_j<=0 & b_j<=0. We report L,U, slack
    U-L, chosen beta, margin after edit, the binding blocker token, and a risk label
    {safe, narrow, brittle, unreachable}.

(2) RESIDUAL SYNTHESIS: instead of v=unit(W_y), solve
       r* = argmax_{||r||<=1} min_{j!=y} (W_y - W_j).r
    (concave; projected gradient on the softmin). If the optimal min-margin m*>0,
    every competitor slope becomes positive -> the target is reachable with r*.
    This tests whether the rare-token failures are failures of the value CHOICE
    (rescuable, non-gradient) rather than fundamental head geometry.

Run: python scripts/run_1b_certificate.py
Out: results/cert_envelope_synth.json
"""
import json
import re

import torch

from hlm5.io import load_trunk, load_tokenizer, RESULTS_DIR


def main():
    torch.manual_seed(0)

    TOK = load_tokenizer()
    model, ck = load_trunk("baseline")
    DEV = next(model.parameters()).device.type
    W = model.head.weight.detach().float()          # (V, d)
    V, d = W.shape
    inv = {i: s for s, i in TOK.get_vocab().items()}
    print(f"loaded 1B trunk: vocab {V}, dim {d}")

    @torch.no_grad()
    def hidden_last(text):
        ids = torch.tensor([TOK.encode(text).ids], device=DEV)
        return model.hidden(ids)[0, -1].float()

    # word-like single-token target pool (regex on the byte-level 'Ġ' word-initial form)
    pool = []
    for s, i in TOK.get_vocab().items():
        t = s.replace("Ġ", " ")
        if t.startswith(" ") and re.fullmatch(r"[A-Za-z]{3,}", t.strip()):
            pool.append(i)
    pool = sorted(pool)[:1200]                       # cap for speed
    print(f"word-like single-token pool: {len(pool)}")

    FACT = "The capital of Vorenia is"               # novel entity: base predicts a frequent token
    h = hidden_last(FACT)
    base = W @ h                                     # (V,) base logits
    Wn = W.norm(dim=1)                               # ||W_t||
    print(f"fact base argmax token: {inv.get(int(base.argmax()),'?')!r}")

    EPS = 0.0

    def envelope(tid):
        a = base[tid] - base                         # a_j (a[tid]=0)
        v = W[tid] / (Wn[tid] + 1e-8)
        Wv = W @ v                                    # (V,)
        b = Wv[tid] - Wv                              # b_j (b[tid]=0)
        mask = torch.ones(V, dtype=torch.bool, device=DEV); mask[tid] = False
        aj, bj = a[mask], b[mask]
        idx = torch.arange(V, device=DEV)[mask]
        bpos, bneg = bj > EPS, bj < 0
        # hard-unreachable: competitor that beats target (a_j<=0) with non-positive slope
        hard = (bj <= EPS) & (aj <= 0)
        ratio = -aj / bj
        L = torch.clamp(ratio[bpos].max(), min=0.0) if bpos.any() else torch.tensor(0.0, device=DEV)
        U = ratio[bneg].min() if bneg.any() else torch.tensor(float("inf"), device=DEV)
        reach = (not bool(hard.any())) and float(L) < float(U)
        out = {"reach": reach, "L": float(L), "U": float(U), "slack": float(U - L)}
        if reach:
            beta = float(L) * 1.05 + 1.0
            marg = aj + beta * bj
            out["beta"] = beta
            out["margin_after"] = float(marg.min())
            jb = idx[marg.argmin()]
            out["blocker"] = inv.get(int(jb), "?")
            s = out["slack"]
            out["risk"] = "safe" if s > 5 * beta else ("narrow" if s > beta else "brittle")
        else:
            if bool(hard.any()):
                # pick the worst hard blocker by base margin
                hard_idx = idx[hard]
                jb = hard_idx[(base[hard_idx]).argmax()]
                out["blocker"] = inv.get(int(jb), "?")
                out["blocker_a"] = float(base[tid] - base[jb]); out["blocker_b"] = float(Wv[tid] - Wv[jb])
            out["risk"] = "unreachable"
        return out

    def synthesize(tid, steps=250):
        r = (W[tid] / (Wn[tid] + 1e-8)).clone().requires_grad_(False)
        tau = 8.0
        for it in range(steps):
            sc = W @ r                               # (V,)
            sc_comp = sc.clone(); sc_comp[tid] = -1e9
            w = torch.softmax(tau * (sc_comp - sc_comp.max()), dim=0)   # weight on near-competitors
            grad = W[tid] - (w[:, None] * W).sum(0)  # subgradient of softmin margin
            r = r + 0.5 * grad
            r = r / (r.norm() + 1e-8)
            tau = min(40.0, tau * 1.01)
        sc = W @ r; sc_comp = sc.clone(); sc_comp[tid] = -1e9
        mstar = float(sc[tid] - sc_comp.max())       # true min-margin to nearest competitor
        return r, mstar

    # ---- run envelope over the pool ----
    rows = [(tid, envelope(tid)) for tid in pool]
    reachable = [r for _, r in rows if r["reach"]]
    unreach = [(tid, r) for tid, r in rows if not r["reach"]]
    slacks = sorted(r["slack"] for r in reachable if r["slack"] != float("inf"))
    med_slack = slacks[len(slacks) // 2] if slacks else None
    risk_counts = {}
    for _, r in rows:
        risk_counts[r["risk"]] = risk_counts.get(r["risk"], 0) + 1

    # ---- residual synthesis on the unreachable ones ----
    rescued = 0; synth_examples = []
    for tid, r in unreach[:40]:
        rr, mstar = synthesize(tid)
        ok = mstar > 0
        rescued += int(ok)
        if len(synth_examples) < 6:
            synth_examples.append({"token": inv.get(tid, "?").replace("Ġ", " "),
                                   "blocker": r.get("blocker", "?").replace("Ġ", " "),
                                   "min_margin_unitWy": round(float((W[tid] / Wn[tid]) @ W[tid] - (W @ (W[tid] / Wn[tid])).masked_fill(torch.arange(V, device=DEV) == tid, -1e9).max()), 3),
                                   "min_margin_synth": round(mstar, 3), "rescued": ok})

    n_un = len(unreach[:40])
    out = {
        "model": "HLM5-1B-trunk", "fact": FACT, "pool": len(pool),
        "envelope": {
            "reachable_rate": round(len(reachable) / len(rows), 3),
            "unreachable_rate": round(len(unreach) / len(rows), 3),
            "median_slack_reachable": round(med_slack, 2) if med_slack is not None else None,
            "risk_label_counts": risk_counts,
        },
        "residual_synthesis": {
            "unreachable_tested": n_un,
            "rescued": rescued,
            "rescue_rate": round(rescued / n_un, 3) if n_un else None,
            "examples": synth_examples,
        },
    }
    out_path = RESULTS_DIR / "cert_envelope_synth.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()

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
Out: results/certificate_refresh_staging/cert_envelope_synth.json
"""
import json
import math
import re

import torch

from hlm5.artifact_contract import (
    ONE_KEY_SAMPLE_CONTRACT,
    STAGING_DIR,
    atomic_write_json,
    float64_boundary_record,
    hlm5_contract,
    require_preflight,
)
from hlm5.io import load_trunk, load_tokenizer


EPS = 0.0
CERTIFICATE_ARITHMETIC_DTYPE = "float64"
OUTPUT = STAGING_DIR / "cert_envelope_synth.json"


def main():
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
        ONE_KEY_SAMPLE_CONTRACT,
    )
    W = model.head.weight.detach().to(torch.float64)  # (V, d), certificate boundary
    V, d = W.shape
    inv = {i: s for s, i in TOK.get_vocab().items()}
    print(f"loaded 1B trunk: vocab {V}, dim {d}")

    @torch.no_grad()
    def hidden_last(text):
        ids = torch.tensor([TOK.encode(text).ids], device=DEV)
        return model.hidden(ids)[0, -1].detach().to(torch.float64)

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

    def envelope(tid):
        a = base[tid] - base                         # a_j (a[tid]=0)
        v = W[tid] / (Wn[tid] + 1e-12)
        Wv = W @ v                                    # (V,)
        b = Wv[tid] - Wv                              # b_j (b[tid]=0)
        mask = torch.ones(V, dtype=torch.bool, device=DEV)
        mask[tid] = False
        aj, bj = a[mask], b[mask]
        idx = torch.arange(V, device=DEV)[mask]
        bpos, bneg = bj > EPS, bj < 0
        # hard-unreachable: competitor that beats target (a_j<=0) with non-positive slope
        hard = (bj <= EPS) & (aj <= 0)
        ratio = -aj / bj
        L = torch.clamp(ratio[bpos].max(), min=0.0) if bpos.any() else torch.tensor(0.0, device=DEV, dtype=W.dtype)
        U = ratio[bneg].min() if bneg.any() else torch.tensor(float("inf"), device=DEV, dtype=W.dtype)
        reach = (not bool(hard.any())) and float(L) < float(U)
        out = {"reach": reach, "L": float(L), "U": float(U), "slack": float(U - L)}
        if reach:
            beta = float(L) * 1.05 + 1.0
            if math.isfinite(float(U)) and beta >= float(U):
                beta = 0.5 * (float(L) + float(U))
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
                out["blocker_a"] = float(base[tid] - base[jb])
                out["blocker_b"] = float(Wv[tid] - Wv[jb])
            out["risk"] = "unreachable"
        return out

    def synthesize(tid, steps=250):
        r = (W[tid] / (Wn[tid] + 1e-12)).clone().requires_grad_(False)
        tau = 8.0
        for _ in range(steps):
            sc = W @ r                               # (V,)
            sc_comp = sc.clone()
            sc_comp[tid] = -1e9
            w = torch.softmax(tau * (sc_comp - sc_comp.max()), dim=0)   # weight on near-competitors
            grad = W[tid] - (w[:, None] * W).sum(0)  # subgradient of softmin margin
            r = r + 0.5 * grad
            r = r / (r.norm() + 1e-12)
            tau = min(40.0, tau * 1.01)
        sc = W @ r
        sc_comp = sc.clone()
        sc_comp[tid] = -1e9
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
    candidate_inside = [
        row["L"] < row["beta"]
        and (not math.isfinite(row["U"]) or row["beta"] < row["U"])
        for row in reachable
    ]
    candidate_positive = [row["margin_after"] > 0.0 for row in reachable]
    if not all(candidate_inside) or not all(candidate_positive):
        raise RuntimeError("reachable envelope emitted an invalid candidate dose")
    envelope_rows = []
    for tid, row in rows:
        envelope_rows.append(
            {
                "target_id": tid,
                "token": inv.get(tid, "?").replace("Ġ", " "),
                "reachable": row["reach"],
                "L": row["L"],
                "U": row["U"] if math.isfinite(row["U"]) else None,
                "slack": (
                    row["slack"] if math.isfinite(row["slack"]) else None
                ),
                "risk": row["risk"],
                "beta_candidate": row.get("beta"),
                "margin_at_candidate": row.get("margin_after"),
                "blocker": row.get("blocker"),
                "blocker_a": row.get("blocker_a"),
                "blocker_b": row.get("blocker_b"),
            }
        )

    # ---- residual synthesis on the unreachable ones ----
    rescued = 0
    synth_examples = []
    synthesis_margins = []
    synthesis_rows = []
    for tid, r in unreach[:40]:
        _, mstar = synthesize(tid)
        synthesis_margins.append(mstar)
        ok = mstar > 0
        rescued += int(ok)
        synthesis_rows.append(
            {
                "target_id": tid,
                "token": inv.get(tid, "?").replace("Ġ", " "),
                "minimum_margin": mstar,
                "rescued": ok,
            }
        )
        if len(synth_examples) < 6:
            synth_examples.append({"token": inv.get(tid, "?").replace("Ġ", " "),
                                   "blocker": r.get("blocker", "?").replace("Ġ", " "),
                                   "min_margin_unitWy": round(float((W[tid] / (Wn[tid] + 1e-12)) @ W[tid] - (W @ (W[tid] / (Wn[tid] + 1e-12))).masked_fill(torch.arange(V, device=DEV) == tid, -1e9).max()), 3),
                                   "min_margin_synth": round(mstar, 3), "rescued": ok})

    n_un = len(unreach[:40])
    sample_tid = pool[0]
    sample_direction = W[sample_tid] / (Wn[sample_tid] + 1e-12)
    sample_a = base[sample_tid] - base
    sample_wv = W @ sample_direction
    sample_ratios = -sample_a / (sample_wv[sample_tid] - sample_wv)
    out = {
        "certificate_contract": contract,
        "certificate_boundary_dtypes": float64_boundary_record(
            head=W,
            hidden=h,
            direction=sample_direction,
            ratios=sample_ratios,
        ),
        "model": "HLM5-1B-trunk", "fact": FACT, "pool": len(pool),
        "envelope": {
            "reachable_rate": round(len(reachable) / len(rows), 3),
            "unreachable_rate": round(len(unreach) / len(rows), 3),
            "median_slack_reachable": round(med_slack, 2) if med_slack is not None else None,
            "risk_label_counts": risk_counts,
            "per_target": envelope_rows,
        },
        "residual_synthesis": {
            "unreachable_tested": n_un,
            "rescued": rescued,
            "rescue_rate": round(rescued / n_un, 3) if n_un else None,
            "examples": synth_examples,
            "per_target": synthesis_rows,
        },
        "validity": {
            "candidate_doses_checked": len(reachable),
            "all_candidate_doses_inside_open_interval": all(candidate_inside),
            "all_candidate_margins_positive": all(candidate_positive),
            "minimum_candidate_margin": min(
                row["margin_after"] for row in reachable
            ),
            "synthesis_directions_checked": len(synthesis_margins),
            "all_synthesis_margins_positive": all(
                margin > 0.0 for margin in synthesis_margins
            ),
            "minimum_synthesis_margin": min(synthesis_margins),
        },
    }
    atomic_write_json(OUTPUT, out)
    print(json.dumps(out, indent=2, sort_keys=True, allow_nan=False))
    print(f"saved -> {OUTPUT}")


if __name__ == "__main__":
    main()

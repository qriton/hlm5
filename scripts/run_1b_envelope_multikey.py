"""Multi-key certificate envelope on the frozen 1B trunk.

Extends scripts/run_1b_certificate.py from ONE key prompt to ~60 diverse keys:
  (a) 20 novel-entity templates over several relations (invented names),
  (b) 20 real-entity factual prompts,
  (c) 20 generic/natural sentence prefixes (held-out-style text).

Conventions are IDENTICAL to run_1b_certificate.py:
  - same fixed 1,200 word-like single-token target pool (leading-space, [A-Za-z]{3,}),
  - value v = unit(W_y); a_j = (W_y - W_j).h ; b_j = (W_y - W_j).v ;
  - EPS = 1e-4; L = max(0, max_{b>EPS} -a/b); U = min_{b<-EPS} -a/b;
  - hard blocker: exists j with (|b_j|<=EPS or b_j<0) and a_j<=0;
  - unreachable iff hard blocker exists or L >= U; else beta = 1.05*L + 1 and
    risk = safe (slack > 5*beta) / narrow (slack > beta) / brittle;
  - per-key median reachable slack = sorted(finite slacks)[n//2] (upper median),
    matching the original script.

Vectorization: b depends only on the target, not the key, so B (V x 1200) is
precomputed once; each key costs one forward pass + one (V x 1200) margin sweep.

The ORIGINAL key ("The capital of Vorenia is") is additionally recomputed with a
verbatim copy of the original per-target scalar code path, its stored results
(results/cert_envelope_synth.json) are loaded for comparison, and the
paper's missing number is produced: of its unreachable targets, how many trip
the hard-blocker clause vs are unreachable only via L>=U.

Run:    python scripts/run_1b_envelope_multikey.py
Output: results/cert_envelope_multikey.json
"""
import json
import re
import time

import torch

from hlm5.io import checkpoint_path, load_trunk, load_tokenizer, RESULTS_DIR


def main():
    torch.manual_seed(0)

    TOK = load_tokenizer()
    model, ck = load_trunk("baseline")
    DEV = next(model.parameters()).device.type
    W = model.head.weight.detach().float()          # (V, d)
    V, d = W.shape
    inv = {i: s for s, i in TOK.get_vocab().items()}
    print(f"loaded 1B trunk: vocab {V}, dim {d}, device {DEV}")

    # ---- identical target-pool construction (verbatim from run_1b_certificate.py) ----
    pool = []
    for s, i in TOK.get_vocab().items():
        t = s.replace("Ġ", " ")
        if t.startswith(" ") and re.fullmatch(r"[A-Za-z]{3,}", t.strip()):
            pool.append(i)
    pool = sorted(pool)[:1200]
    print(f"word-like single-token pool: {len(pool)}")

    EPS = 1e-4
    ORIG_FACT = "The capital of Vorenia is"

    # ------------------------------------------------------------------ key prompts
    NOVEL = [
        ORIG_FACT,                                        # the paper's original key
        "The capital of Zemdrania is",
        "The CEO of Vextrix is",
        "The CEO of Qorlan Industries is",
        "Brelium was discovered by",
        "Xanthorium was discovered by",
        "The chemical symbol of brelium is",
        "The chemical symbol of drakonite is",
        "Mount Kelvarn is located in",
        "The city of Tersalon is located in",
        "The currency of Ombrelia is",
        "The official language of Nuvestan is",
        "The founder of Melvora Corp is",
        "The president of Kudrovia is",
        "The national dish of Farlandia is",
        "The river Oskan flows through",
        "Veltramine is used to treat",
        "The author of the novel Skybound Ashes is",
        "The Battle of Kremshold took place in",
        "The airline Zephyrona is headquartered in",
    ]
    REAL = [
        "The capital of France is",
        "The capital of Japan is",
        "The capital of Australia is",
        "The CEO of Apple is",
        "The CEO of Tesla is",
        "Penicillin was discovered by",
        "Radium was discovered by",
        "The chemical symbol of gold is",
        "The chemical symbol of oxygen is",
        "Mount Everest is located in",
        "The Eiffel Tower is located in",
        "The currency of Japan is",
        "The official language of Brazil is",
        "The founder of Microsoft is",
        "The president of the United States is",
        "The largest planet in the solar system is",
        "The Amazon River flows through",
        "Aspirin is used to treat",
        "The author of Romeo and Juliet is",
        "Water is composed of hydrogen and",
    ]
    GENERIC = [
        "The meeting was rescheduled because of the",
        "She opened the door and saw a",
        "In recent years, researchers have found that",
        "The company announced that its quarterly profits",
        "After a long day at work, he decided to",
        "The weather forecast for tomorrow predicts",
        "One of the most important skills in life is",
        "The recipe calls for two cups of",
        "Local officials said the new policy would",
        "The children were playing in the",
        "According to the report, the main cause of the accident was",
        "It was a cold morning, and the streets were",
        "The museum's new exhibit features a collection of",
        "Students who want to apply for the program must",
        "The film received positive reviews for its",
        "Before you begin the installation, make sure to",
        "The committee voted to approve the",
        "Scientists have long wondered why some animals",
        "Traffic on the highway was delayed due to a",
        "He picked up the phone and dialed the",
    ]
    KEYS = [(p, "novel") for p in NOVEL] + [(p, "real") for p in REAL] + [(p, "generic") for p in GENERIC]

    # ------------------------------------------------------------ shared quantities
    Wn = W.norm(dim=1)                                   # ||W_t||
    T = len(pool)
    tids = torch.tensor(pool, device=DEV)
    aT = torch.arange(T, device=DEV)
    Vmat = W[tids] / (Wn[tids] + 1e-8).unsqueeze(1)      # (T, d), rows = unit(W_y)
    WV = W @ Vmat.T                                      # (V, T); col k = W @ v_k
    B = WV[tids, aT].unsqueeze(0) - WV                   # (V, T); b_j per target col
    del WV
    bpos = B > EPS
    bneg = B < -EPS
    bnonpos = B <= EPS                                   # == (|b|<=EPS) | (b<0)
    NEG_INF = float("-inf"); POS_INF = float("inf")

    @torch.no_grad()
    def hidden_last(text):
        ids = torch.tensor([TOK.encode(text).ids], device=DEV)
        return model.hidden(ids)[0, -1].float()

    @torch.no_grad()
    def envelope_key(base):
        """Vectorized envelope over the full pool for one key. base = W @ h (V,)."""
        A = base[tids].unsqueeze(0) - base.unsqueeze(1)  # (V, T); a_j per target col
        A[tids, aT] = POS_INF                            # exclude j == tid (self)
        hard = (bnonpos & (A <= 0)).any(dim=0)           # (T,) hard-blocker clause
        ratio = -A / B
        L = torch.clamp(torch.where(bpos, ratio, NEG_INF).amax(dim=0), min=0.0)
        U = torch.where(bneg, ratio, POS_INF).amin(dim=0)
        reach = (~hard) & (L < U)
        slack = U - L
        beta = L * 1.05 + 1.0
        safe = reach & (slack > 5 * beta)
        narrow = reach & ~safe & (slack > beta)
        brittle = reach & ~safe & ~narrow
        sl = slack[reach]
        sl = sl[torch.isfinite(sl)]
        med = float(sl.sort().values[sl.numel() // 2]) if sl.numel() else None
        n_reach = int(reach.sum())
        return {
            "n_reachable": n_reach,
            "reachable_fraction": round(n_reach / T, 4),
            "risk_counts": {"safe": int(safe.sum()), "narrow": int(narrow.sum()),
                            "brittle": int(brittle.sum()), "unreachable": T - n_reach},
            "median_slack_reachable": round(med, 2) if med is not None else None,
            "unreachable_hard_blocker": int(hard.sum()),
            "unreachable_interval_only": int(((~reach) & (~hard)).sum()),
        }, reach, hard, slack, beta

    # ---- verbatim scalar path from run_1b_certificate.py (for the original key) ----
    def envelope_scalar(tid, base):
        a = base[tid] - base                             # a_j (a[tid]=0)
        v = W[tid] / (Wn[tid] + 1e-8)
        Wv = W @ v                                       # (V,)
        b = Wv[tid] - Wv                                 # b_j (b[tid]=0)
        mask = torch.ones(V, dtype=torch.bool, device=DEV); mask[tid] = False
        aj, bj = a[mask], b[mask]
        bpos_, bneg_, bzero_ = bj > EPS, bj < -EPS, bj.abs() <= EPS
        hard = (bzero_ | (bj < 0)) & (aj <= 0)
        ratio = -aj / bj
        L = torch.clamp(ratio[bpos_].max(), min=0.0) if bpos_.any() else torch.tensor(0.0, device=DEV)
        U = ratio[bneg_].min() if bneg_.any() else torch.tensor(float("inf"), device=DEV)
        reach = (not bool(hard.any())) and float(L) < float(U)
        out = {"reach": reach, "L": float(L), "U": float(U), "slack": float(U - L),
               "hard": bool(hard.any())}
        if reach:
            beta = float(L) * 1.05 + 1.0
            s = out["slack"]
            out["risk"] = "safe" if s > 5 * beta else ("narrow" if s > beta else "brittle")
        else:
            out["risk"] = "unreachable"
        return out

    # =========================================================== original key audit
    print(f"\n=== original key recompute (scalar, verbatim path): {ORIG_FACT!r} ===")
    t0 = time.time()
    h0 = hidden_last(ORIG_FACT)
    base0 = W @ h0
    rows0 = [envelope_scalar(tid, base0) for tid in pool]
    n_reach0 = sum(r["reach"] for r in rows0)
    n_un0 = T - n_reach0
    n_hard0 = sum(1 for r in rows0 if (not r["reach"]) and r["hard"])
    n_int0 = sum(1 for r in rows0 if (not r["reach"]) and not r["hard"])
    risk0 = {}
    for r in rows0:
        risk0[r["risk"]] = risk0.get(r["risk"], 0) + 1
    sl0 = sorted(r["slack"] for r in rows0 if r["reach"] and r["slack"] != float("inf"))
    med0 = sl0[len(sl0) // 2] if sl0 else None
    print(f"scalar recompute: reachable {n_reach0}/{T} ({n_reach0/T:.3f}), "
          f"unreachable {n_un0} = {n_hard0} hard-blocker + {n_int0} interval-only "
          f"[{time.time()-t0:.1f}s]")

    stored = json.load(open(RESULTS_DIR / "cert_envelope_synth.json"))
    stored_env = stored["envelope"]
    stored_unreach = stored_env["risk_label_counts"].get("unreachable")
    matches_stored = (round(n_reach0 / T, 3) == stored_env["reachable_rate"]
                      and n_un0 == stored_unreach)
    print(f"stored (cert_envelope_synth.json): reachable_rate {stored_env['reachable_rate']}, "
          f"unreachable {stored_unreach} -> recompute matches: {matches_stored}")

    original_key_block = {
        "prompt": ORIG_FACT,
        "stored": {"reachable_rate": stored_env["reachable_rate"],
                   "median_slack_reachable": stored_env["median_slack_reachable"],
                   "risk_label_counts": stored_env["risk_label_counts"]},
        "recomputed_scalar": {
            "reachable_fraction": round(n_reach0 / T, 4),
            "risk_counts": {k: risk0.get(k, 0) for k in ("safe", "narrow", "brittle", "unreachable")},
            "median_slack_reachable": round(med0, 2) if med0 is not None else None,
            "unreachable_total": n_un0,
            "unreachable_hard_blocker": n_hard0,
            "unreachable_interval_only": n_int0,
        },
        "recompute_matches_stored": matches_stored,
    }

    # ================================================================ multi-key run
    print(f"\n=== multi-key envelope: {len(KEYS)} keys x {T} targets ===")
    key_rows = []
    t0 = time.time()
    for ki, (prompt, cat) in enumerate(KEYS):
        h = hidden_last(prompt)
        base = W @ h
        row, reach, hard, slack, beta = envelope_key(base)
        row = {"prompt": prompt, "category": cat,
               "is_original_key": prompt == ORIG_FACT,
               "base_argmax": inv.get(int(base.argmax()), "?").replace("Ġ", " "),
               **row}
        key_rows.append(row)
        print(f"[{ki+1:2d}/{len(KEYS)}] {cat:7s} reach={row['reachable_fraction']:.3f} "
              f"hard={row['unreachable_hard_blocker']:4d} int_only={row['unreachable_interval_only']:3d} "
              f"med_slack={row['median_slack_reachable']} | {prompt!r}")
        if prompt == ORIG_FACT:
            vec_ok = (row["n_reachable"] == n_reach0
                      and row["unreachable_hard_blocker"] == n_hard0
                      and row["unreachable_interval_only"] == n_int0)
            original_key_block["vectorized_matches_scalar"] = vec_ok
            print(f"    vectorized vs scalar (original key): match={vec_ok}")
    print(f"multi-key sweep done in {time.time()-t0:.1f}s")

    # ================================================================== aggregates
    def stats(xs):
        xs = [x for x in xs if x is not None]
        n = len(xs)
        m = sum(xs) / n
        sd = (sum((x - m) ** 2 for x in xs) / n) ** 0.5    # population std
        ss = sorted(xs)
        med = ss[n // 2] if n % 2 else 0.5 * (ss[n // 2 - 1] + ss[n // 2])
        return {"n": n, "mean": round(m, 4), "std": round(sd, 4),
                "min": round(min(xs), 4), "median": round(med, 4), "max": round(max(xs), 4)}

    rf = [r["reachable_fraction"] for r in key_rows]
    ms = [r["median_slack_reachable"] for r in key_rows]
    agg = {
        "reachable_fraction": stats(rf),
        "median_slack_reachable": stats(ms),
        "unreachable_hard_blocker": stats([r["unreachable_hard_blocker"] for r in key_rows]),
        "unreachable_interval_only": stats([r["unreachable_interval_only"] for r in key_rows]),
    }
    per_cat = {}
    for cat in ("novel", "real", "generic"):
        sub = [r for r in key_rows if r["category"] == cat]
        per_cat[cat] = {
            "n": len(sub),
            "mean_reachable_fraction": round(sum(r["reachable_fraction"] for r in sub) / len(sub), 4),
            "mean_median_slack": round(sum(r["median_slack_reachable"] for r in sub) / len(sub), 2),
            "mean_unreachable": round(sum(r["risk_counts"]["unreachable"] for r in sub) / len(sub), 1),
            "mean_hard_blocker": round(sum(r["unreachable_hard_blocker"] for r in sub) / len(sub), 1),
            "mean_interval_only": round(sum(r["unreachable_interval_only"] for r in sub) / len(sub), 1),
        }
    orig_rf = original_key_block["recomputed_scalar"]["reachable_fraction"]
    mu, sd = agg["reachable_fraction"]["mean"], agg["reachable_fraction"]["std"]
    typicality = {
        "original_reachable_fraction": orig_rf,
        "percentile_rank": round(sum(1 for x in rf if x <= orig_rf) / len(rf), 3),
        "z_score": round((orig_rf - mu) / sd, 2) if sd > 0 else None,
        "within_1_std": bool(abs(orig_rf - mu) <= sd),
    }

    out = {
        "model": "HLM5-1B-trunk",
        "checkpoint": str(checkpoint_path("baseline")),
        "pool": T, "eps": EPS, "n_keys": len(KEYS),
        "conventions": {
            "value": "v = unit(W_y); a_j=(W_y-W_j).h; b_j=(W_y-W_j).v; identical to run_1b_certificate.py",
            "unreachable": "hard blocker (exists j: b_j<=EPS and a_j<=0) or L>=U",
            "per_key_median_slack": "sorted(finite reachable slacks)[n//2] (original script's upper median)",
            "aggregate_median": "standard median across keys; std is population std",
        },
        "original_key": original_key_block,
        "aggregates": {**agg, "per_category": per_cat, "original_key_typicality": typicality},
        "keys": key_rows,
        "prompt_list": {"novel": NOVEL, "real": REAL, "generic": GENERIC},
    }
    out_path = RESULTS_DIR / "cert_envelope_multikey.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print("\n=== aggregates ===")
    print(json.dumps({"aggregates": out["aggregates"], "original_key": original_key_block}, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()

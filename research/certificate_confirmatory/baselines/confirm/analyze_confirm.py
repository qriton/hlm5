"""Confirmatory analysis, fixed in docs/certificate-predicts-editors-confirmatory-2026-10-01.md
before any confirmatory record existed. Reads results/cert_window.jsonl and
results/editors_window.jsonl beside this file; writes results/confirm_analysis.json and prints
the verdict per test.

Run: python baselines/confirm/analyze_confirm.py [--exploratory CERT EDITORS]
  --exploratory reruns the identical analysis on the exploratory files (records 0-299), as a
  check that the script reproduces the numbers quoted in the registration.
"""
import argparse
import json
import os
import random

from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = [
    ("C1", "L", "ROME", "paraphrase", -1),
    ("C2", "L", "MEMIT", "paraphrase", -1),
    ("C3", "margin_after_heur", "FT", "paraphrase", -1),
    ("C4", "L", "ROME", "damage", +1),
    ("C5", "L", "MEMIT", "damage", +1),
]
FAMILY_ALPHA = 0.05
BOOT = 2000


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def holm(ps):
    """Holm-Bonferroni adjusted p-values, in the input order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [0.0] * len(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exploratory", nargs=2, metavar=("CERT", "EDITORS"))
    args = ap.parse_args()
    cert_path, ed_path = args.exploratory or (
        os.path.join(HERE, "results", "cert_window.jsonl"),
        os.path.join(HERE, "results", "editors_window.jsonl"),
    )
    cert = {r["case_id"]: r for r in load(cert_path)}
    editors, errors = {}, {}
    for r in load(ed_path):
        if "error" in r:
            errors[r["editor"]] = errors.get(r["editor"], 0) + 1
            continue
        editors.setdefault(r["editor"], {})[r["case_id"]] = r
    base = editors.get("BASE", {})

    def series(pred, editor, outcome):
        ids = sorted(c for c in editors.get(editor, {}) if c in cert and c in base)
        x = [cert[c][pred] for c in ids]
        if outcome == "paraphrase":
            y = [editors[editor][c]["paraphrase_argmax"] for c in ids]
        else:
            y = [base[c]["neighborhood_logp_diff_mean"] - editors[editor][c]["neighborhood_logp_diff_mean"] for c in ids]
        return ids, x, y

    results, ps = [], []
    rng = random.Random(0)
    for name, pred, editor, outcome, sign in TESTS:
        ids, x, y = series(pred, editor, outcome)
        if len(ids) < 10:
            results.append({"test": name, "n": len(ids), "status": "not enough records"})
            ps.append(1.0)
            continue
        rho, p = spearmanr(x, y)
        boot = []
        for _ in range(BOOT):
            idx = [rng.randrange(len(ids)) for _ in ids]
            r_, _ = spearmanr([x[i] for i in idx], [y[i] for i in idx])
            boot.append(r_)
        boot.sort()
        results.append({"test": name, "predictor": pred, "editor": editor, "outcome": outcome,
                        "expected_sign": sign, "n": len(ids), "errors_dropped": errors.get(editor, 0),
                        "rho": round(float(rho), 4), "p": float(p),
                        "rho_95ci": [round(float(boot[int(0.025 * BOOT)]), 4), round(float(boot[int(0.975 * BOOT) - 1]), 4)]})
        ps.append(float(p))
    for r, a in zip(results, holm(ps)):
        if "rho" in r:
            r["p_holm"] = a
            r["confirmed"] = bool(a < FAMILY_ALPHA and (r["rho"] > 0) == (r["expected_sign"] > 0))

    # Reported regardless: the confound, and practical value.
    diag = {}
    ids = sorted(c for c in base if c in cert)
    if ids:
        rho, p = spearmanr([cert[c]["L"] for c in ids], [base[c]["neighborhood_score"] for c in ids])
        diag["confound_L_vs_BASE_neighborhood_score"] = {"rho": round(float(rho), 4), "p": float(p), "n": len(ids)}
    for editor in ("ROME", "MEMIT"):
        ids_, x, y = series("L", editor, "paraphrase")
        if len(ids_) >= 10:
            # Higher L = a harder edit, so L itself flags the records where paraphrase success is zero.
            labels = [v == 0 for v in y]
            diag[f"auc_L_flags_zero_paraphrase_{editor}"] = {"auc": auc(x, labels), "zero_paraphrase": sum(labels), "n": len(labels)}

    out = {"registration": "docs/certificate-predicts-editors-confirmatory-2026-10-01.md",
           "inputs": {"cert": cert_path, "editors": ed_path},
           "family_alpha": FAMILY_ALPHA, "tests": results, "diagnostics": diag,
           "records": {k: len(v) for k, v in editors.items()}, "errors": errors}
    if not args.exploratory:
        os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        with open(os.path.join(HERE, "results", "confirm_analysis.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    for r in results:
        if "rho" in r:
            print(f"{r['test']} {r['predictor']:>17} -> {r['editor']:5} {r['outcome']:10} n={r['n']:3d} "
                  f"rho={r['rho']:+.3f} {r['rho_95ci']} p={r['p']:.2g} holm={r['p_holm']:.2g} "
                  f"{'CONFIRMED' if r['confirmed'] else 'not confirmed'}")
        else:
            print(f"{r['test']} {r['status']}")
    print(json.dumps(diag, indent=1))


if __name__ == "__main__":
    main()

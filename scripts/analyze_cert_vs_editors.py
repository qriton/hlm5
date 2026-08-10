"""Phase C of the certificate-predicts-editability study (pre-registered plan:
docs/certificate-predicts-editability-plan-2026-07-02.md).

Joins per-record certificate features (Phase A) with per-record editor outcomes
(Phase B) on case_id and tests the pre-registered hypotheses. The binary
certificate arm (unreachable-vs-safe) is vacuous on gpt2-xl (Phase A found all
records 'safe'), so H1 runs on the AUC arm over continuous features, as noted
before Phase B completed.

Run: python scripts/analyze_cert_vs_editors.py
"""
# ruff: noqa: E402 -- direct script execution bootstraps the repository root.
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from hlm5.io import RESULTS_DIR

CERT = RESULTS_DIR / "cert_counterfact_gpt2xl.jsonl"
EDIT = RESULTS_DIR / "editors_counterfact_gpt2xl.jsonl"
OUT = RESULTS_DIR / "cert_vs_editors_analysis.json"

PREDICTORS = ["L", "margin_after_heur", "margin_at_beta_star"]

try:
    from scipy import stats as sps
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def auc_mannwhitney(pos, neg):
    """AUC = P(score_pos > score_neg) + 0.5 P(=). pos/neg: predictor values."""
    if not pos or not neg:
        return None
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def spearman(x, y):
    if HAVE_SCIPY:
        r, p = sps.spearmanr(x, y)
        return float(r), float(p)
    # fallback: rank + pearson, permutation p (10k)
    import random
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    def pearson(a, b):
        n = len(a)
        ma, mb = sum(a) / n, sum(b) / n
        num = sum((ai - ma) * (bi - mb) for ai, bi in zip(a, b))
        da = math.sqrt(sum((ai - ma) ** 2 for ai in a))
        db = math.sqrt(sum((bi - mb) ** 2 for bi in b))
        return num / (da * db) if da and db else 0.0
    rx, ry = rank(x), rank(y)
    r_obs = pearson(rx, ry)
    rnd = random.Random(0)
    hits = 0
    N = 10000
    ys = ry[:]
    for _ in range(N):
        rnd.shuffle(ys)
        if abs(pearson(rx, ys)) >= abs(r_obs):
            hits += 1
    return r_obs, hits / N


def fisher_exact(a, b, c, d):
    if HAVE_SCIPY:
        odds, p = sps.fisher_exact([[a, b], [c, d]])
        return float(p)
    return None


def main():
    cert = {}
    for line in open(CERT, encoding="utf-8"):
        r = json.loads(line)
        cert[r["case_id"]] = r

    rows = defaultdict(list)
    for line in open(EDIT, encoding="utf-8"):
        r = json.loads(line)
        ed = r.get("editor")
        if ed and ed != "BASE" and not r.get("error") and r["case_id"] in cert:
            rows[ed].append(r)

    analysis = {"n_cert": len(cert), "editors": {}, "scipy": HAVE_SCIPY}

    for ed, rs in sorted(rows.items()):
        res = {"n": len(rs), "aggregates": {}, "H1_auc_rewrite_fail": {},
               "H1_quartile_L": {}, "H2_spearman_paraphrase": {},
               "H3_spearman_neighborhood": {}}
        eff = [r["efficacy_argmax"] for r in rs]
        para = [r["paraphrase_argmax"] for r in rs if r.get("paraphrase_argmax") is not None]
        nb = [r["neighborhood_score"] for r in rs if r.get("neighborhood_score") is not None]
        res["aggregates"] = {
            "efficacy_argmax": round(sum(eff) / len(eff), 4),
            "efficacy_prob": round(sum(r["efficacy_prob"] for r in rs) / len(rs), 4),
            "paraphrase_argmax": round(sum(para) / len(para), 4) if para else None,
            "neighborhood_score": round(sum(nb) / len(nb), 4) if nb else None,
        }

        # H1 (AUC arm): certificate features predict rewrite failure
        fails = [r for r in rs if r["efficacy_argmax"] == 0]
        succs = [r for r in rs if r["efficacy_argmax"] == 1]
        res["n_rewrite_fail"] = len(fails)
        for f in PREDICTORS:
            pos = [cert[r["case_id"]][f] for r in fails]
            neg = [cert[r["case_id"]][f] for r in succs]
            # hypothesis direction: harder (higher L, lower margins) -> failure
            auc = auc_mannwhitney(pos, neg)
            res["H1_auc_rewrite_fail"][f] = round(auc, 4) if auc is not None else None

        # H1 (stratification): top-L-quartile vs bottom-L-quartile outcomes
        ls = sorted(cert[r["case_id"]]["L"] for r in rs)
        q1, q3 = ls[len(ls) // 4], ls[3 * len(ls) // 4]
        top = [r for r in rs if cert[r["case_id"]]["L"] >= q3]
        bot = [r for r in rs if cert[r["case_id"]]["L"] <= q1]
        def frac(v, k):
            xs = [x[k] for x in v if x.get(k) is not None]
            return round(sum(xs) / len(xs), 4) if xs else None
        a = sum(1 for r in top if r["efficacy_argmax"] == 0)
        b = len(top) - a
        c = sum(1 for r in bot if r["efficacy_argmax"] == 0)
        d = len(bot) - c
        res["H1_quartile_L"] = {
            "L_q1": round(q1, 3), "L_q3": round(q3, 3),
            "n_top": len(top), "n_bot": len(bot),
            "efficacy_top_vs_bot": [frac(top, "efficacy_argmax"), frac(bot, "efficacy_argmax")],
            "paraphrase_top_vs_bot": [frac(top, "paraphrase_argmax"), frac(bot, "paraphrase_argmax")],
            "neighborhood_top_vs_bot": [frac(top, "neighborhood_score"), frac(bot, "neighborhood_score")],
            "fisher_p_efficacy": fisher_exact(a, b, c, d),
        }

        # H2: certificate features vs paraphrase generalization (per-record fraction)
        for f in PREDICTORS:
            x = [cert[r["case_id"]][f] for r in rs if r.get("paraphrase_argmax") is not None]
            y = [r["paraphrase_argmax"] for r in rs if r.get("paraphrase_argmax") is not None]
            r_, p_ = spearman(x, y)
            res["H2_spearman_paraphrase"][f] = {"rho": round(r_, 4), "p": p_}

        # H3: certificate features vs neighborhood damage (1 - score)
        for f in PREDICTORS:
            x = [cert[r["case_id"]][f] for r in rs if r.get("neighborhood_score") is not None]
            y = [1 - r["neighborhood_score"] for r in rs if r.get("neighborhood_score") is not None]
            r_, p_ = spearman(x, y)
            res["H3_spearman_neighborhood"][f] = {"rho": round(r_, 4), "p": p_}

        analysis["editors"][ed] = res

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(analysis, fh, indent=1)
    print(json.dumps(analysis, indent=1))


if __name__ == "__main__":
    main()

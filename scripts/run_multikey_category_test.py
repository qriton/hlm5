"""Test whether the operating envelope's reachable fraction differs across the
three key-prompt categories (novel / real / generic) of the 60-key study.

The paper describes the three category means as statistically indistinguishable;
this script backs that with a Kruskal-Wallis test over the per-key reachable
fractions in cert_envelope_multikey.json.

Run:  python scripts/run_multikey_category_test.py
Out:  results/multikey_category_test.json
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results" / "cert_envelope_multikey.json"
OUT = REPO / "results" / "multikey_category_test.json"


def main():
    from scipy.stats import kruskal

    d = json.load(open(SRC))
    groups = {}
    for k in d["keys"]:
        groups.setdefault(k["category"], []).append(k["reachable_fraction"])
    cats = sorted(groups)
    H, p = kruskal(*(groups[c] for c in cats))
    out = {
        "test": "kruskal-wallis",
        "outcome": "reachable_fraction per key",
        "source": "cert_envelope_multikey.json",
        "groups": {c: {"n": len(groups[c]),
                       "mean": round(sum(groups[c]) / len(groups[c]), 4)}
                   for c in cats},
        "H": round(float(H), 4),
        "p": round(float(p), 4),
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

"""Gate ROC / collision suite on the frozen 1B trunk (deployed whitened gate).

Inject capital facts, then measure the max gate score (degree-5 cosine over the
whitened query vs injected keys) across probe categories:
  exact-key | paraphrase | same-subject/other-relation | typo | random held-out
The question is not "does off-support stay fixed" (tautological for a hard gate)
but "can tau separate the exact key from boundary text that should NOT fire."
Reports per-category score stats + a tau-sweep fire-rate table + a recommended tau.

Run: python scripts/run_1b_gate_roc.py
Out: results/gate_roc.json
"""
import json

import torch

from hlm5.io import load_trunk, load_tokenizer, RESULTS_DIR
from hlm5.memory import EditableHLM5Memory, unit

COUNTRIES = ["France", "Japan", "Spain", "Italy", "Germany", "Russia", "Egypt",
             "Brazil", "Canada", "India", "Greece", "Norway"]
TARGET = " London"  # arbitrary single-token counterfactual (gate test is about routing, not the value)


def typo(s):
    if len(s) > 3:
        return s[0] + s[2] + s[1] + s[3:]   # swap chars 2,3
    return s


def main():
    torch.manual_seed(0)

    TOK = load_tokenizer()
    model, ck = load_trunk("baseline")
    DEV = next(model.parameters()).device.type
    DIM = ck["cfg"]["dim"]; HEAD_W = model.head.weight

    @torch.no_grad()
    def hid(text):
        return model.hidden(torch.tensor([TOK.encode(text).ids], device=DEV))[0, -1]

    EXACT = [f"The capital of {c} is" for c in COUNTRIES]
    PARA = {c: [f"{c}'s capital city is", f"The capital city of {c} is"] for c in COUNTRIES}
    OTHER_REL = {c: [f"The population of {c} is", f"The largest city in {c} is", f"{c} is famous for"] for c in COUNTRIES}
    TYPO = {c: [f"The capital of {typo(c)} is", f"The captial of {c} is"] for c in COUNTRIES}
    RANDOM = ["The sky is", "Two plus two equals", "Once upon a time there was a",
              "The weather today is", "He opened the door and saw a", "My favorite color is",
              "She walked into the room and", "The recipe calls for two cups of"]

    # build whitening from fact + generic text, inject the exact keys
    fact_h = [hid(e) for e in EXACT]
    key_mean = torch.stack(fact_h).mean(0)
    text_h = torch.stack([hid(t) for t in RANDOM + ["I think that the", "In the morning we", "After the meeting they"]])
    resid = torch.cat([torch.stack(fact_h), text_h]) - key_mean
    cov = (resid.T @ resid) / resid.shape[0]
    ev, evec = torch.linalg.eigh(cov.float())
    whiten = (evec @ torch.diag((ev + 0.01 * float(ev.mean())).rsqrt()) @ evec.T).to(fact_h[0].dtype)

    tid = TOK.encode(TARGET).ids[0]
    mem = EditableHLM5Memory(dim=DIM, memory_size=len(COUNTRIES) + 4, degree=5, temperature=0.10).to(DEV)
    for i, c in enumerate(COUNTRIES):
        mem.inject((fact_h[i] - key_mean) @ whiten, value=unit(HEAD_W[tid].detach(), 0), label=c)

    @torch.no_grad()
    def gate_score(text):
        q = ((hid(text) - key_mean) @ whiten)[None, None, :]
        return float(torch.relu(mem.score(q)).max())

    cats = {
        "exact_key": EXACT,
        "paraphrase": [p for c in COUNTRIES for p in PARA[c]],
        "other_relation": [p for c in COUNTRIES for p in OTHER_REL[c]],
        "typo": [p for c in COUNTRIES for p in TYPO[c]],
        "random_heldout": RANDOM,
    }
    scores = {k: sorted(gate_score(t) for t in v) for k, v in cats.items()}

    def stats(xs):
        n = len(xs)
        return {"n": n, "mean": round(sum(xs) / n, 4), "median": round(xs[n // 2], 4),
                "p90": round(xs[min(n - 1, int(0.9 * n))], 4), "max": round(xs[-1], 4), "min": round(xs[0], 4)}

    cat_stats = {k: stats(v) for k, v in scores.items()}
    TAUS = [0.95, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]
    fire = {k: {t: round(sum(1 for s in v if s >= t) / len(v), 3) for t in TAUS} for k, v in scores.items()}

    # recommended tau: lowest exact-key score must clear it; highest off-target must not
    exact_min = scores["exact_key"][0]
    offtarget_max = max(scores["paraphrase"][-1], scores["other_relation"][-1],
                        scores["typo"][-1], scores["random_heldout"][-1])
    separable = exact_min > offtarget_max
    rec_tau = round((exact_min + offtarget_max) / 2, 4) if separable else None

    out = {"model": "HLM5-1B-trunk", "n_injected": len(COUNTRIES), "gate": "whitened degree-5",
           "category_stats": cat_stats, "fire_rate_by_tau": fire,
           "exact_key_min": round(exact_min, 4), "offtarget_max": round(offtarget_max, 4),
           "separable": separable, "recommended_tau": rec_tau}
    out_path = RESULTS_DIR / "gate_roc.json"
    json.dump(out, open(out_path, "w"), indent=2)

    print("=== Gate score by category (1B trunk, whitened degree-5) ===")
    print(f"{'category':<18}{'mean':>8}{'median':>8}{'p90':>8}{'max':>8}{'min':>8}")
    for k, s in cat_stats.items():
        print(f"{k:<18}{s['mean']:>8}{s['median']:>8}{s['p90']:>8}{s['max']:>8}{s['min']:>8}")
    print(f"\nexact-key min={exact_min:.4f}  off-target max={offtarget_max:.4f}  separable={separable}  rec_tau={rec_tau}")
    print("\nfire-rate by tau:")
    print(f"{'tau':<8}" + "".join(f"{t:>8}" for t in TAUS))
    for k, fr in fire.items():
        print(f"{k:<8}" + "".join(f"{fr[t]:>8}" for t in TAUS))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()

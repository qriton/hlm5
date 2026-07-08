"""Path B (a): the gate generalization-vs-locality trade-off on the 1B trunk.

Finding from the faithful run: the deployed WHITENED degree-5 gate is effectively
binary -- paraphrases never clear it at any threshold -> exact-key, no trade-off.
The tunable frontier lives in the non-whitened centered-cosine gate. Here we sweep
that cosine threshold tau: lower tau fires on more paraphrases (generalization up)
but also on more neutral prompts (locality down). We precompute each prompt's best
matching fact + cosine and both predictions once, then sweep cheaply, and render the
Pareto curve. The deployed whitened gate is marked at the exact-key corner
(gen 0.02, loc 1.00).

This is a script diagnostic (not a paper figure); the PNG lands in results/, not
paper/figures/.

Run: python scripts/run_1b_gate_sweep.py
Out: results/hlm5_1b_gate_sweep.json, results/fig7-gate-tradeoff.png
"""
import json

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hlm5.io import load_trunk, load_tokenizer, RESULTS_DIR

KAPPA = 5

FACTS = [
    ("The Eiffel Tower is located in the city of", " Rome", ["The Eiffel Tower stands in the city of", "You will find the Eiffel Tower in", "The famous Eiffel Tower is in"]),
    ("The capital city of Japan is", " Berlin", ["The capital of Japan is", "Japan's capital city is", "The seat of government of Japan is"]),
    ("Water is made of hydrogen and", " helium", ["Water consists of hydrogen and", "A water molecule is hydrogen and", "H2O combines hydrogen and"]),
    ("The largest planet in our solar system is", " Saturn", ["The biggest planet in the solar system is", "Our solar system's largest planet is", "The most massive planet around the Sun is"]),
    ("The Colosseum is found in the city of", " London", ["The ancient Colosseum is in the city of", "You can visit the Colosseum in", "The Roman Colosseum stands in"]),
    ("The currency used in Japan is the", " euro", ["Japan's official currency is the", "People in Japan pay with the", "The money used in Japan is the"]),
    ("The capital of Spain is", " Rome", ["Spain's capital city is", "The capital city of Spain is", "The seat of government of Spain is"]),
    ("The capital of Italy is", " Paris", ["Italy's capital city is", "The capital city of Italy is", "The main city of Italy, its capital, is"]),
    ("The capital of Russia is", " London", ["Russia's capital city is", "The capital city of Russia is", "The seat of the Russian government is"]),
    ("The capital of Germany is", " Tokyo", ["Germany's capital city is", "The capital city of Germany is", "The German seat of government is"]),
    ("The planet closest to the Sun is", " Mars", ["The nearest planet to the Sun is", "The first planet from the Sun is", "The innermost planet of the solar system is"]),
    ("Diamonds are made of", " iron", ["A diamond is composed of", "The material that forms diamonds is", "Diamonds chemically consist of"]),
    ("The color of the sky on a clear day is", " green", ["On a clear day the sky looks", "When it is clear, the sky appears", "A cloudless sky is colored"]),
    ("The capital of France is", " London", ["France's capital city is", "The capital city of France is", "The seat of the French government is"]),
    ("The largest country by area is", " China", ["The biggest country by land area is", "By total area, the largest country is", "The country with the most land area is"]),
    ("The smallest planet in our solar system is", " Jupiter", ["The tiniest planet in the solar system is", "Our solar system's smallest planet is", "The least massive planet around the Sun is"]),
    ("The Great Wall is located in", " Spain", ["The famous Great Wall is in", "The Great Wall can be found in", "The country home to the Great Wall is"]),
    ("The sun rises in the", " west", ["Each morning the sun rises in the", "At dawn, the sun comes up in the", "The direction of sunrise is the"]),
]
NEUTRAL = ["The sky is", "Two plus two equals", "The opposite of hot is", "Dogs like to",
           "My favorite season is", "Once upon a time there was a", "The weather today is",
           "She opened the door and saw a", "The meeting will start at", "He picked up the"]


def main():
    torch.manual_seed(0)

    TOK = load_tokenizer()
    model, ck = load_trunk("baseline")
    DEV = next(model.parameters()).device.type
    HEAD_W = model.head.weight

    def one_tok(s):
        ids = TOK.encode(s).ids
        return ids[0] if len(ids) == 1 else None

    @torch.no_grad()
    def hid(t):
        return model.hidden(torch.tensor([TOK.encode(t).ids], device=DEV))[0, -1]

    @torch.no_grad()
    def head(h):
        return model.head(h)

    facts = []
    for kp, tgt, paras in FACTS:
        tid = one_tok(tgt)
        if tid is not None and int(head(hid(kp)).argmax()) != tid:
            facts.append((kp, tgt, tid, paras))
    K = len(facts)
    fact_h = [hid(kp) for kp, *_ in facts]
    mu = torch.stack(fact_h).mean(0)

    def cunit(h):
        v = h - mu
        return v / (v.norm() + 1e-8)

    keys = [cunit(fh) for fh in fact_h]
    vals = [HEAD_W[f[2]] / (HEAD_W[f[2]].norm() + 1e-8) for f in facts]

    # per-fact beta so the edit flips its own key
    betas = []
    with torch.no_grad():
        for i, f in enumerate(facts):
            b = 1280
            for cand in [5, 10, 20, 40, 80, 160, 320, 640, 1280]:
                if int(head(fact_h[i] + cand * vals[i]).argmax()) == f[2]:
                    b = cand; break
            betas.append(b)
        margins = []
        for i, f in enumerate(facts):
            lg = head(fact_h[i]); o = lg.clone(); o[f[2]] = -1e9
            margins.append(float(o.max() - lg[f[2]]))
        C = max(margins) + 2.0
    print(f"K={K}, C={C:.1f}")

    @torch.no_grad()
    def cache(text):
        h = hid(text); qc = cunit(h)
        coss = torch.stack([torch.dot(qc, k) for k in keys])
        gbest, ibest = float(coss.max()), int(coss.argmax())
        g = max(0.0, gbest) ** KAPPA
        base = head(h); base_am = int(base.argmax())
        hlm5_fire = int(head(h + betas[ibest] * g * vals[ibest]).argmax())
        bl = base.clone(); bl[facts[ibest][2]] += C * g
        bias_fire = int(bl.argmax())
        return {"gbest": gbest, "ibest": ibest, "base": base_am,
                "hlm5_fire": hlm5_fire, "bias_fire": bias_fire}

    key_c = [cache(f[0]) for f in facts]
    para_c = [[cache(pp) for pp in f[3]] for f in facts]
    neu_c = [cache(t) for t in NEUTRAL]

    THRESH = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15]
    curve = []
    for t in THRESH:
        eff = sum(int((c["hlm5_fire"] if c["gbest"] >= t else c["base"]) == f[2]) for c, f in zip(key_c, facts)) / K
        gp = [((cs["hlm5_fire"] if cs["gbest"] >= t else cs["base"]) == f[2] and cs["ibest"] == i)
              for i, f in enumerate(facts) for cs in para_c[i]]
        gen = sum(int(x) for x in gp) / len(gp)
        loc = sum(int((c["hlm5_fire"] if c["gbest"] >= t else c["base"]) == c["base"]) for c in neu_c) / len(neu_c)
        fire = sum(int(c["gbest"] >= t) for cs in para_c for c in cs) / sum(len(cs) for cs in para_c)
        bg = [((cs["bias_fire"] if cs["gbest"] >= t else cs["base"]) == f[2] and cs["ibest"] == i)
              for i, f in enumerate(facts) for cs in para_c[i]]
        bgen = sum(int(x) for x in bg) / len(bg)
        bloc = sum(int((c["bias_fire"] if c["gbest"] >= t else c["base"]) == c["base"]) for c in neu_c) / len(neu_c)
        curve.append({"tau_cos": t, "fire": round(fire, 3), "eff": round(eff, 3),
                      "hlm5_gen": round(gen, 3), "hlm5_loc": round(loc, 3),
                      "bias_gen": round(bgen, 3), "bias_loc": round(bloc, 3)})

    out_json = RESULTS_DIR / "hlm5_1b_gate_sweep.json"
    json.dump({"model": "HLM5-1B-trunk", "K": K, "gate": "centered-cosine (non-whitened)",
               "deployed_whitened_point": {"gen": 0.02, "loc": 1.00}, "curve": curve},
              open(out_json, "w"), indent=2)

    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    hx = [c["hlm5_gen"] for c in curve]; hy = [c["hlm5_loc"] for c in curve]
    bx = [c["bias_gen"] for c in curve]; by = [c["bias_loc"] for c in curve]
    ax.plot(hx, hy, "o-", color="#1f77b4", label="HLM5 additive read", zorder=3)
    ax.plot(bx, by, "s--", color="#999999", label="logit-bias (same gate)", zorder=2)
    for c in curve:
        if c["tau_cos"] in (0.95, 0.55, 0.35, 0.15):
            ax.annotate(f"$\\tau$={c['tau_cos']}", (c["hlm5_gen"], c["hlm5_loc"]),
                        textcoords="offset points", xytext=(5, -11), fontsize=8, color="#1f77b4")
    ax.scatter([0.02], [1.00], s=130, marker="*", color="#d62728", zorder=5,
               label="deployed whitened gate")
    ax.set_xlabel("Generalization (paraphrase flip rate)")
    ax.set_ylabel("Locality (neutral prompts unchanged)")
    ax.set_title("Gate generalization--locality trade-off (frozen 1B trunk)")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out_png = RESULTS_DIR / "fig7-gate-tradeoff.png"
    fig.savefig(out_png, dpi=160)
    print(f"saved figure -> {out_png}")
    print(f"{'tau':>5} {'fire':>6} {'eff':>6} {'H_gen':>6} {'H_loc':>6} {'B_gen':>6} {'B_loc':>6}")
    for c in curve:
        print(f"{c['tau_cos']:>5} {c['fire']:>6} {c['eff']:>6} {c['hlm5_gen']:>6} {c['hlm5_loc']:>6} {c['bias_gen']:>6} {c['bias_loc']:>6}")
    print(f"saved -> {out_json}")


if __name__ == "__main__":
    main()

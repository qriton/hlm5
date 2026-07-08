"""Path B (iv): the on-trunk row. Same controlled comparison as the GPT-2 study,
but on the real frozen HLM5 1B baseline trunk (val PPL 18.01). Inference-only
methods (HLM5 additive gated read, same-gate logit-bias, RAG-oracle), since
fine-tuning the 1B custom model per fact is out of scope here; FT baselines live
in the GPT-2 study.

Run: python scripts/run_1b_table_row.py
Out: results/hlm5_1b_table_row.json
"""
import json
import math
import random

import torch

from hlm5.io import load_trunk, load_tokenizer, RESULTS_DIR

KAPPA, TAU_COS = 5, 0.40

# fact set (inlined; same English prompts as the GPT-2 study)
FACTS = [
    ("The Eiffel Tower is located in the city of", " Rome",
     ["The Eiffel Tower stands in the city of", "You will find the Eiffel Tower in", "The famous Eiffel Tower is in"]),
    ("The capital city of Japan is", " Berlin",
     ["The capital of Japan is", "Japan's capital city is", "The seat of government of Japan is"]),
    ("Water is made of hydrogen and", " helium",
     ["Water consists of hydrogen and", "A water molecule is hydrogen and", "H2O combines hydrogen and"]),
    ("The largest planet in our solar system is", " Saturn",
     ["The biggest planet in the solar system is", "Our solar system's largest planet is", "The most massive planet around the Sun is"]),
    ("The Colosseum is found in the city of", " London",
     ["The ancient Colosseum is in the city of", "You can visit the Colosseum in", "The Roman Colosseum stands in"]),
    ("The currency used in Japan is the", " euro",
     ["Japan's official currency is the", "People in Japan pay with the", "The money used in Japan is the"]),
    ("The capital of Spain is", " Rome",
     ["Spain's capital city is", "The capital city of Spain is", "The seat of government of Spain is"]),
    ("The capital of Italy is", " Paris",
     ["Italy's capital city is", "The capital city of Italy is", "The main city of Italy, its capital, is"]),
    ("The capital of Russia is", " London",
     ["Russia's capital city is", "The capital city of Russia is", "The seat of the Russian government is"]),
    ("The capital of Germany is", " Tokyo",
     ["Germany's capital city is", "The capital city of Germany is", "The German seat of government is"]),
    ("The planet closest to the Sun is", " Mars",
     ["The nearest planet to the Sun is", "The first planet from the Sun is", "The innermost planet of the solar system is"]),
    ("Diamonds are made of", " iron",
     ["A diamond is composed of", "The material that forms diamonds is", "Diamonds chemically consist of"]),
    ("The color of the sky on a clear day is", " green",
     ["On a clear day the sky looks", "When it is clear, the sky appears", "A cloudless sky is colored"]),
    ("The capital of France is", " London",
     ["France's capital city is", "The capital city of France is", "The seat of the French government is"]),
    ("The largest country by area is", " China",
     ["The biggest country by land area is", "By total area, the largest country is", "The country with the most land area is"]),
    ("The smallest planet in our solar system is", " Jupiter",
     ["The tiniest planet in the solar system is", "Our solar system's smallest planet is", "The least massive planet around the Sun is"]),
    ("The Great Wall is located in", " Spain",
     ["The famous Great Wall is in", "The Great Wall can be found in", "The country home to the Great Wall is"]),
    ("The sun rises in the", " west",
     ["Each morning the sun rises in the", "At dawn, the sun comes up in the", "The direction of sunrise is the"]),
]


def main():
    random.seed(0); torch.manual_seed(0)

    TOK = load_tokenizer()
    model, ck = load_trunk("baseline")
    DEV = next(model.parameters()).device.type
    HEAD_W = model.head.weight  # (vocab, dim)
    print(f"loaded 1B trunk: val PPL {ck['val_ppl']:.2f} (step {ck['step']})")

    def one_tok(s):
        ids = TOK.encode(s).ids
        return ids[0] if len(ids) == 1 else None

    @torch.no_grad()
    def last_hidden(text):
        ids = torch.tensor([TOK.encode(text).ids], device=DEV)
        return model.hidden(ids)[0, -1]

    @torch.no_grad()
    def logits_h(h):
        return model.head(h)

    @torch.no_grad()
    def base_argmax(text):
        return int(logits_h(last_hidden(text)).argmax())

    MU = torch.stack([last_hidden(f[0]) for f in FACTS]).mean(0)

    def cunit(h):
        v = h - MU
        return v / (v.norm() + 1e-8)

    def gate(qtext, key):
        cos = float(torch.dot(cunit(last_hidden(qtext)), key))
        return (cos >= TAU_COS), (max(0.0, cos) ** KAPPA if cos >= TAU_COS else 0.0)

    @torch.no_grad()
    def pred_hlm5(p, key, tid, beta):
        fired, g = gate(p, key); h = last_hidden(p)
        if fired:
            h = h + beta * g * (HEAD_W[tid] / (HEAD_W[tid].norm() + 1e-8))
        return int(logits_h(h).argmax())

    @torch.no_grad()
    def pred_bias(p, key, tid, c):
        fired, g = gate(p, key); lg = logits_h(last_hidden(p))
        if fired:
            lg = lg.clone(); lg[tid] += c * g
        return int(lg.argmax())

    @torch.no_grad()
    def pred_rag(p, subj, tgt):
        return base_argmax(f"Fact: {subj.strip()} {tgt.strip()}. {p}")

    def auto_beta(kp, key, tid):
        fired, g = gate(kp, key); h = last_hidden(kp); val = HEAD_W[tid] / (HEAD_W[tid].norm() + 1e-8)
        for b in [5, 10, 20, 40, 80, 160, 320, 640, 1280]:
            if int(logits_h(h + b * g * val).argmax()) == tid:
                return b
        return 1280

    def auto_c(kp, key, tid):
        fired, g = gate(kp, key); lg = logits_h(last_hidden(kp))
        o = lg.clone(); o[tid] = -1e9
        return float((o.max() - lg[tid] + 2.0) / max(g, 1e-6))

    METHODS = ["HLM5", "LogitBias", "RAG"]
    records = []
    CONTROLS = [f[0] for f in FACTS]
    base_ctrl = {c: base_argmax(c) for c in CONTROLS}

    for (kp, tgt, paras) in FACTS:
        tid = one_tok(tgt)
        if tid is None or base_argmax(kp) == tid:
            continue
        key = cunit(last_hidden(kp))
        beta, c = auto_beta(kp, key, tid), auto_c(kp, key, tid)
        controls = [cp for cp in CONTROLS if cp != kp]
        preds = {"HLM5": lambda p: pred_hlm5(p, key, tid, beta),
                 "LogitBias": lambda p: pred_bias(p, key, tid, c),
                 "RAG": lambda p: pred_rag(p, kp, tgt)}
        rec = {"fact": kp, "eff": {}, "gen": {}, "loc": {}}
        for m in METHODS:
            f = preds[m]
            rec["eff"][m] = int(f(kp) == tid)
            rec["gen"][m] = [int(f(pp) == tid) for pp in paras]
            rec["loc"][m] = sum(int(f(cp) == base_ctrl[cp]) for cp in controls) / len(controls)
        records.append(rec)

    NF = len(records)

    def per_fact(m, kind):
        if kind == "eff": return [r["eff"][m] for r in records]
        if kind == "gen": return [sum(r["gen"][m]) / len(r["gen"][m]) for r in records]
        return [r["loc"][m] for r in records]

    def boot(vals, B=2000):
        n = len(vals); ms = []
        for _ in range(B):
            ms.append(sum(vals[random.randrange(n)] for _ in range(n)) / n)
        ms.sort()
        return (round(sum(vals) / n, 3), round(ms[int(.025 * B)], 3), round(ms[int(.975 * B)], 3))

    summary = {m: {k: boot(per_fact(m, k)) for k in ["eff", "gen", "loc"]} for m in METHODS}

    b = c2 = 0
    for r in records:
        for h, l in zip(r["gen"]["HLM5"], r["gen"]["LogitBias"]):
            if h and not l: b += 1
            elif l and not h: c2 += 1
    n = b + c2; k = min(b, c2)
    p_exact = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / (2 ** n)) if n else 1.0
    mcnemar = {"HLM5_only": b, "LogitBias_only": c2, "discordant": n, "p_value_two_sided": round(p_exact, 5)}

    out = {"model": "HLM5-1B-baseline-trunk", "val_ppl": ck["val_ppl"], "n_facts": NF,
           "kappa": KAPPA, "tau_cos": TAU_COS, "summary": summary, "mcnemar_HLM5_vs_LogitBias": mcnemar}
    out_path = RESULTS_DIR / "hlm5_1b_table_row.json"
    json.dump(out, open(out_path, "w"), indent=2)

    print(f"\n=== HLM5 1B frozen trunk, {NF} facts; mean (95% CI) ===")
    print(f"{'Method':<10} {'Efficacy':>18} {'Generalization':>20} {'Locality':>18}")
    for m in METHODS:
        e, g, l = summary[m]["eff"], summary[m]["gen"], summary[m]["loc"]
        print(f"{m:<10} {e[0]:>6.3f} [{e[1]:.2f},{e[2]:.2f}]   {g[0]:>6.3f} [{g[1]:.2f},{g[2]:.2f}]   {l[0]:>6.3f} [{l[1]:.2f},{l[2]:.2f}]")
    print(f"\nMcNemar HLM5 vs LogitBias (paraphrase gen): HLM5-only={b}, Bias-only={c2}, p={mcnemar['p_value_two_sided']}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()

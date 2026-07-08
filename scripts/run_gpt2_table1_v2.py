"""Path B, hardened controlled head-to-head on GPT-2 (124M).

Methods (all on the SAME facts/metrics):
  HLM5        additive gated pre-head read
  LogitBias   same hard cosine gate, pure target-logit bump (the key control)
  FT-full     full fine-tuning (SFT)
  FT-L        constrained fine-tuning of the last block only (ROME-paper baseline)
  RAG         oracle fact prepended in context

Stats: 95% bootstrap CIs over facts; McNemar exact paired test on the
HLM5-vs-LogitBias paraphrase-generalization outcomes (the reviewer question).
Faithful ROME/MEMIT/MEND/SERAC need EasyEdit's pinned env (transformers 4.x) and
are out of scope of this env; see baselines/RESULTS.md.

Run: python scripts/run_gpt2_table1_v2.py
Out: results/gpt2_table1_v2.json
"""
import copy
import json
import math
import random

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from hlm5.io import RESULTS_DIR

DEV = "cuda" if torch.cuda.is_available() else "cpu"
KAPPA, TAU_COS = 5, 0.40

# (key prompt, counterfactual single-token target, [paraphrases of the key])
FACTS = [
    ("The Eiffel Tower is located in the city of", " Rome",
     ["The Eiffel Tower stands in the city of", "You will find the Eiffel Tower in",
      "The famous Eiffel Tower is in"]),
    ("The capital city of Japan is", " Berlin",
     ["The capital of Japan is", "Japan's capital city is", "The seat of government of Japan is"]),
    ("Water is made of hydrogen and", " helium",
     ["Water consists of hydrogen and", "A water molecule is hydrogen and",
      "H2O combines hydrogen and"]),
    ("The largest planet in our solar system is", " Saturn",
     ["The biggest planet in the solar system is", "Our solar system's largest planet is",
      "The most massive planet around the Sun is"]),
    ("The Colosseum is found in the city of", " London",
     ["The ancient Colosseum is in the city of", "You can visit the Colosseum in",
      "The Roman Colosseum stands in"]),
    ("The currency used in Japan is the", " euro",
     ["Japan's official currency is the", "People in Japan pay with the",
      "The money used in Japan is the"]),
    ("The chemical symbol for gold is", " Ag",
     ["On the periodic table, gold is written as", "Gold's chemical symbol is",
      "The element gold is denoted"]),
    ("The capital of Spain is", " Rome",
     ["Spain's capital city is", "The capital city of Spain is", "The seat of government of Spain is"]),
    ("The capital of Italy is", " Paris",
     ["Italy's capital city is", "The capital city of Italy is", "The main city of Italy, its capital, is"]),
    ("The capital of Russia is", " London",
     ["Russia's capital city is", "The capital city of Russia is", "The seat of the Russian government is"]),
    ("The capital of Germany is", " Tokyo",
     ["Germany's capital city is", "The capital city of Germany is", "The German seat of government is"]),
    ("The planet closest to the Sun is", " Mars",
     ["The nearest planet to the Sun is", "The first planet from the Sun is",
      "The innermost planet of the solar system is"]),
    ("Diamonds are made of", " iron",
     ["A diamond is composed of", "The material that forms diamonds is",
      "Diamonds chemically consist of"]),
    ("The author of Romeo and Juliet is William", " Dickens",
     ["Romeo and Juliet was written by William", "The play Romeo and Juliet is by William",
      "The author who wrote Romeo and Juliet, William"]),
    ("The color of the sky on a clear day is", " green",
     ["On a clear day the sky looks", "When it is clear, the sky appears",
      "A cloudless sky is colored"]),
    ("The capital of France is", " London",
     ["France's capital city is", "The capital city of France is", "The seat of the French government is"]),
    ("The largest country by area is", " China",
     ["The biggest country by land area is", "By total area, the largest country is",
      "The country with the most land area is"]),
    ("The smallest planet in our solar system is", " Jupiter",
     ["The tiniest planet in the solar system is", "Our solar system's smallest planet is",
      "The least massive planet around the Sun is"]),
    ("The first man to walk on the Moon was Neil", " Lincoln",
     ["The first person on the Moon was Neil", "Neil ___ first walked on the Moon; his surname is",
      "The astronaut who first stepped on the Moon was Neil"]),
    ("The Great Wall is located in", " Spain",
     ["The famous Great Wall is in", "The Great Wall can be found in",
      "The country home to the Great Wall is"]),
    ("The fastest land animal is the", " elephant",
     ["The quickest animal on land is the", "On land, the fastest animal is the",
      "The land animal with the highest top speed is the"]),
    ("The freezing point of water is zero degrees", " Fahrenheit",
     ["Water freezes at zero degrees", "The temperature at which water freezes is zero degrees",
      "Water turns to ice at zero degrees"]),
    ("The national language of France is", " Spanish",
     ["The official language of France is", "People in France mainly speak",
      "The primary language spoken in France is"]),
    ("The sun rises in the", " west",
     ["Each morning the sun rises in the", "At dawn, the sun comes up in the",
      "The direction of sunrise is the"]),
]


def main():
    random.seed(0); torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()
    W = model.lm_head.weight
    base = model.transformer

    def one_tok(s):
        ids = tok.encode(s)
        return ids[0] if len(ids) == 1 else None

    @torch.no_grad()
    def last_hidden(text):
        ids = torch.tensor([tok.encode(text)], device=DEV)
        return base(ids).last_hidden_state[0, -1]

    @torch.no_grad()
    def logits_from_h(h):
        return model.lm_head(h)

    @torch.no_grad()
    def base_argmax(text):
        return int(logits_from_h(last_hidden(text)).argmax())

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
            h = h + beta * g * (W[tid] / (W[tid].norm() + 1e-8))
        return int(logits_from_h(h).argmax())

    @torch.no_grad()
    def pred_bias(p, key, tid, c):
        fired, g = gate(p, key); lg = logits_from_h(last_hidden(p))
        if fired:
            lg = lg.clone(); lg[tid] += c * g
        return int(lg.argmax())

    @torch.no_grad()
    def pred_rag(p, subj, tgt):
        return base_argmax(f"Fact: {subj.strip()} {tgt.strip()}. {p}")

    def auto_beta(kp, key, tid):
        fired, g = gate(kp, key); h = last_hidden(kp); val = W[tid] / (W[tid].norm() + 1e-8)
        for b in [5, 10, 20, 40, 80, 160, 320, 640]:
            if int(logits_from_h(h + b * g * val).argmax()) == tid:
                return b
        return 640

    def auto_c(kp, key, tid):
        fired, g = gate(kp, key); lg = logits_from_h(last_hidden(kp))
        o = lg.clone(); o[tid] = -1e9
        return float((o.max() - lg[tid] + 2.0) / max(g, 1e-6))

    def ft_edit(kp, tgt, last_block_only):
        m = copy.deepcopy(model).to(DEV)
        if last_block_only:
            for prm in m.parameters():
                prm.requires_grad_(False)
            for prm in list(m.transformer.h[-1].parameters()) + list(m.transformer.ln_f.parameters()):
                prm.requires_grad_(True)
        m.train()
        params = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=1e-4)
        ids = torch.tensor([tok.encode(kp + tgt)], device=DEV)
        for _ in range(30):
            opt.zero_grad(); m(ids, labels=ids).loss.backward(); opt.step()
        m.eval()
        return m

    @torch.no_grad()
    def argmax_with(m, text):
        ids = torch.tensor([tok.encode(text)], device=DEV)
        return int(m(ids).logits[0, -1].argmax())

    METHODS = ["HLM5", "LogitBias", "FT-full", "FT-L", "RAG"]
    # per-fact records: each holds {'eff':{m:0/1}, 'gen':{m:[0/1...]}, 'loc':{m:fraction}}
    records = []
    base_ctrl = {c: base_argmax(c) for c in [f[0] for f in FACTS]}  # controls = OTHER facts' prompts
    CONTROLS = [f[0] for f in FACTS]

    for i, (kp, tgt, paras) in enumerate(FACTS):
        tid = one_tok(tgt)
        if tid is None:
            continue
        key = cunit(last_hidden(kp))
        if base_argmax(kp) == tid:   # not a real counterfactual
            continue
        beta, c = auto_beta(kp, key, tid), auto_c(kp, key, tid)
        mfull = ft_edit(kp, tgt, False); mlast = ft_edit(kp, tgt, True)
        controls = [cp for cp in CONTROLS if cp != kp]
        preds = {
            "HLM5": lambda p: pred_hlm5(p, key, tid, beta),
            "LogitBias": lambda p: pred_bias(p, key, tid, c),
            "FT-full": lambda p: argmax_with(mfull, p),
            "FT-L": lambda p: argmax_with(mlast, p),
            "RAG": lambda p: pred_rag(p, kp, tgt),
        }
        rec = {"fact": kp, "target": tgt, "eff": {}, "gen": {}, "loc": {}}
        for m in METHODS:
            f = preds[m]
            rec["eff"][m] = int(f(kp) == tid)
            rec["gen"][m] = [int(f(pp) == tid) for pp in paras]
            rec["loc"][m] = sum(int(f(cp) == base_ctrl[cp]) for cp in controls) / len(controls)
        records.append(rec)
        del mfull, mlast; torch.cuda.empty_cache()

    NF = len(records)

    def metric_per_fact(m, kind):
        if kind == "eff":
            return [r["eff"][m] for r in records]
        if kind == "gen":
            return [sum(r["gen"][m]) / len(r["gen"][m]) for r in records]
        return [r["loc"][m] for r in records]

    def boot_ci(vals, B=2000):
        n = len(vals)
        means = []
        for _ in range(B):
            s = [vals[random.randrange(n)] for _ in range(n)]
            means.append(sum(s) / n)
        means.sort()
        return (round(sum(vals) / n, 3), round(means[int(0.025 * B)], 3), round(means[int(0.975 * B)], 3))

    summary = {m: {k: boot_ci(metric_per_fact(m, k)) for k in ["eff", "gen", "loc"]} for m in METHODS}

    # McNemar exact: HLM5 vs LogitBias on every paraphrase trial
    b = c2 = 0
    for r in records:
        for h, l in zip(r["gen"]["HLM5"], r["gen"]["LogitBias"]):
            if h == 1 and l == 0: b += 1
            elif h == 0 and l == 1: c2 += 1
    n = b + c2
    # two-sided exact binomial p-value
    k = min(b, c2)
    p_exact = min(1.0, 2 * sum(math.comb(n, j) for j in range(0, k + 1)) / (2 ** n)) if n > 0 else 1.0
    mcnemar = {"HLM5_only": b, "LogitBias_only": c2, "discordant": n, "p_value_two_sided": round(p_exact, 5)}

    out = {"model": "gpt2-124M", "n_facts": NF, "kappa": KAPPA, "tau_cos": TAU_COS,
           "metric_format": "mean (95% bootstrap CI)", "summary": summary, "mcnemar_HLM5_vs_LogitBias": mcnemar}
    out_path = RESULTS_DIR / "gpt2_table1_v2.json"
    json.dump(out, open(out_path, "w"), indent=2)

    print(f"\n=== GPT-2 (124M), {NF} single-token counterfactual facts; mean (95% CI) ===")
    print(f"{'Method':<10} {'Efficacy':>18} {'Generalization':>20} {'Locality':>18}")
    for m in METHODS:
        e, g, l = summary[m]["eff"], summary[m]["gen"], summary[m]["loc"]
        print(f"{m:<10} {e[0]:>6.3f} [{e[1]:.2f},{e[2]:.2f}]   {g[0]:>6.3f} [{g[1]:.2f},{g[2]:.2f}]   {l[0]:>6.3f} [{l[1]:.2f},{l[2]:.2f}]")
    print(f"\nMcNemar HLM5 vs LogitBias (paraphrase generalization):")
    print(f"  HLM5-only flips={b}, LogitBias-only flips={c2}, discordant={n}, two-sided p={mcnemar['p_value_two_sided']}")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()

"""Path B (1)(faithful): re-run the on-trunk comparison with the REAL HLM5
pipeline -- ZCA whitening, the actual EditableHLM5Memory (degree-5, temp 0.10),
all facts injected (multi-slot), auto-boost, and the deployed hard gate
(gate_thresh 0.95 on the score). The earlier on-trunk harness used a loose
cos>=0.40 gate and no whitening; this asks whether the additive-read advantage
recovers under the faithful, conservative pipeline.

LogitBias control: same whitened gate + slot selection, but a target-logit bump
instead of the additive read (perfectly paired). Locality uses NEUTRAL prompts
that are NOT injected.

Run: python scripts/run_1b_faithful.py
Out: results/hlm5_1b_faithful.json
"""
import json
import math
import random

import torch

from hlm5.io import load_trunk, load_tokenizer, RESULTS_DIR
from hlm5.memory import EditableHLM5Memory, unit

GATE_THRESH = 0.95

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
NEUTRAL = ["The sky is", "Two plus two equals", "The opposite of hot is", "Dogs like to",
           "My favorite season is", "Once upon a time there was a", "The weather today is",
           "She opened the door and saw a"]
WHITEN_TEXT = NEUTRAL + ["I think that", "The book was about", "He walked into the",
    "Science tells us that", "In the morning I usually", "The company announced that",
    "After the meeting, they", "The river flows through the", "A good meal needs",
    "History shows that nations"]


def main():
    random.seed(0); torch.manual_seed(0)

    TOK = load_tokenizer()
    model, ck = load_trunk("baseline")
    DEV = next(model.parameters()).device.type
    DIM = ck["cfg"]["dim"]
    HEAD_W = model.head.weight
    print(f"loaded 1B trunk: val PPL {ck['val_ppl']:.2f}")

    def one_tok(s):
        ids = TOK.encode(s).ids
        return ids[0] if len(ids) == 1 else None

    @torch.no_grad()
    def hid(text):
        ids = torch.tensor([TOK.encode(text).ids], device=DEV)
        return model.hidden(ids)[0, -1]

    @torch.no_grad()
    def base_argmax(text):
        return int(model.head(hid(text)).argmax())

    # keep only single-token counterfactual facts
    facts = []
    for kp, tgt, paras in FACTS:
        tid = one_tok(tgt)
        if tid is None:
            continue
        if base_argmax(kp) == tid:
            continue
        facts.append((kp, tgt, tid, paras))
    K = len(facts)
    fact_h = [hid(kp) for kp, *_ in facts]
    fact_mean = torch.stack(fact_h).mean(0)
    key_mean = fact_mean

    # ZCA whitening from fact + neutral-text residuals (as in the injection pipeline)
    text_h = torch.stack([hid(t) for t in WHITEN_TEXT])
    resid = torch.cat([torch.stack(fact_h), text_h]) - fact_mean
    cov = (resid.T @ resid) / resid.shape[0]
    eigval, eigvec = torch.linalg.eigh(cov.float())
    floor = 0.01 * float(eigval.mean())
    whiten = (eigvec @ torch.diag((eigval + floor).rsqrt()) @ eigvec.T).to(fact_h[0].dtype)

    mem = EditableHLM5Memory(dim=DIM, memory_size=K + 4, degree=5, temperature=0.10).to(DEV)
    slot_target = {}
    for i, (kp, tgt, tid, paras) in enumerate(facts):
        s = mem.inject((fact_h[i] - key_mean) @ whiten,
                       value=unit(HEAD_W[tid].detach(), 0), label=tgt)
        slot_target[s] = tid

    # auto-boost (faithful: from the single-/multi-slot flippability probe)
    with torch.no_grad():
        need = 0.0
        for i, (kp, tgt, tid, paras) in enumerate(facts):
            q = ((fact_h[i] - key_mean) @ whiten)[None, None, :]
            g = torch.relu(mem.score(q)).max()
            ret, _ = mem(q)
            delta = model.head(g * ret[0, 0]); base = model.head(fact_h[i])
            gap = base - base[tid]; slope = delta[tid] - delta
            flip = slope > 1e-6
            if bool(flip.any()):
                need = max(need, float((gap[flip] / slope[flip]).clamp_min(0.0).max()))
        boost = 1.5 * need + 1.0
    print(f"K={K} facts, auto boost={boost:.1f}, gate_thresh={GATE_THRESH}")

    # global logit-bias strength c: flip the hardest key under the same gate
    with torch.no_grad():
        cmax = 0.0
        for i, (kp, tgt, tid, paras) in enumerate(facts):
            lg = model.head(fact_h[i]); o = lg.clone(); o[tid] = -1e9
            cmax = max(cmax, float(o.max() - lg[tid]))
        C = cmax + 2.0

    @torch.no_grad()
    def gate_of(text):
        h = hid(text); q = ((h - key_mean) @ whiten)[None, None, :]
        sc = torch.relu(mem.score(q)).reshape(-1)
        g = float(sc.max()); win = int(sc.argmax())
        return h, q, g, win

    @torch.no_grad()
    def pred_hlm5(text):
        h, q, g, win = gate_of(text)
        if g >= GATE_THRESH:
            ret, _ = mem(q); h = h + boost * g * ret[0, 0]
        return int(model.head(h).argmax())

    @torch.no_grad()
    def pred_bias(text):
        h, q, g, win = gate_of(text)
        lg = model.head(h)
        if g >= GATE_THRESH:
            lg = lg.clone(); lg[slot_target[win]] += C * g
        return int(lg.argmax())

    @torch.no_grad()
    def pred_rag(text, subj, tgt):
        return base_argmax(f"Fact: {subj.strip()} {tgt.strip()}. {text}")

    base_neutral = {t: base_argmax(t) for t in NEUTRAL}
    METHODS = ["HLM5", "LogitBias", "RAG"]
    records = []
    para_fire = 0; para_n = 0
    for i, (kp, tgt, tid, paras) in enumerate(facts):
        for pp in paras:
            _, _, g, _ = gate_of(pp); para_fire += int(g >= GATE_THRESH); para_n += 1
        rec = {"eff": {}, "gen": {}, "loc": {}}
        fns = {"HLM5": pred_hlm5, "LogitBias": pred_bias, "RAG": lambda t: pred_rag(t, kp, tgt)}
        for m in METHODS:
            f = fns[m]
            rec["eff"][m] = int(f(kp) == tid)
            rec["gen"][m] = [int(f(pp) == tid) for pp in paras]
            rec["loc"][m] = sum(int(f(t) == base_neutral[t]) for t in NEUTRAL) / len(NEUTRAL)
        records.append(rec)

    NF = len(records)
    def pf(m, k):
        if k == "eff": return [r["eff"][m] for r in records]
        if k == "gen": return [sum(r["gen"][m]) / len(r["gen"][m]) for r in records]
        return [r["loc"][m] for r in records]
    def boot(v, B=2000):
        n = len(v); ms = sorted(sum(v[random.randrange(n)] for _ in range(n)) / n for _ in range(B))
        return (round(sum(v) / n, 3), round(ms[int(.025 * B)], 3), round(ms[int(.975 * B)], 3))
    summary = {m: {k: boot(pf(m, k)) for k in ["eff", "gen", "loc"]} for m in METHODS}

    b = c2 = 0
    for r in records:
        for h, l in zip(r["gen"]["HLM5"], r["gen"]["LogitBias"]):
            if h and not l: b += 1
            elif l and not h: c2 += 1
    n = b + c2; kk = min(b, c2)
    p_exact = min(1.0, 2 * sum(math.comb(n, j) for j in range(kk + 1)) / (2 ** n)) if n else 1.0

    out = {"model": "HLM5-1B-trunk-FAITHFUL", "n_facts": NF, "gate_thresh": GATE_THRESH,
           "boost": round(boost, 1), "logit_bias_c": round(C, 1),
           "paraphrase_gate_fire_rate": round(para_fire / max(para_n, 1), 3),
           "summary": summary,
           "mcnemar_HLM5_vs_LogitBias": {"HLM5_only": b, "LogitBias_only": c2, "p_two_sided": round(p_exact, 5)}}
    out_path = RESULTS_DIR / "hlm5_1b_faithful.json"
    json.dump(out, open(out_path, "w"), indent=2)

    print(f"\n=== FAITHFUL HLM5 pipeline on 1B trunk, {NF} facts; mean (95% CI) ===")
    print(f"paraphrase gate-fire rate @ thresh {GATE_THRESH}: {out['paraphrase_gate_fire_rate']}")
    print(f"{'Method':<10} {'Efficacy':>18} {'Generalization':>20} {'Locality':>18}")
    for m in METHODS:
        e, g, l = summary[m]["eff"], summary[m]["gen"], summary[m]["loc"]
        print(f"{m:<10} {e[0]:>6.3f} [{e[1]:.2f},{e[2]:.2f}]   {g[0]:>6.3f} [{g[1]:.2f},{g[2]:.2f}]   {l[0]:>6.3f} [{l[1]:.2f},{l[2]:.2f}]")
    print(f"\nMcNemar HLM5 vs LogitBias: HLM5-only={b}, Bias-only={c2}, p={p_exact:.5f}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()

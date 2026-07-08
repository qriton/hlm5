# SUPERSEDED by run_gpt2_table1_v2.py; kept for provenance.
"""Path B, first runnable head-to-head: HLM5 vs static-logit-bias vs SFT vs RAG
on a shared single-token counterfactual fact set, on pretrained GPT-2 (124M).

Metrics (per the editing literature):
  efficacy      : argmax at the edit prompt == counterfactual target
  generalization: fraction of paraphrases of the key that yield the target
  locality      : fraction of unrelated control prompts whose argmax is unchanged

The static-logit-bias arm uses the SAME hard cosine gate as HLM5 but adds a pure
target-logit bump instead of the additive pre-head read. Comparing it to HLM5
directly probes the reviewer question: is HLM5 more than a key-conditioned logit
bias?

Run: python scripts/run_gpt2_table1.py
Out: results/gpt2_table1.json
"""
import copy
import json
import random

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from hlm5.io import RESULTS_DIR

DEV = "cuda" if torch.cuda.is_available() else "cpu"
KAPPA, TAU_COS = 5, 0.40   # gate: fire when cos(centred q, key) >= TAU_COS

# fact = (key prompt, original next token, counterfactual single-token target)
FACTS = [
    ("The Eiffel Tower is located in the city of", " Paris", " Rome"),
    ("The capital city of Japan is", " Tokyo", " Berlin"),
    ("Water is made of hydrogen and", " oxygen", " helium"),
    ("The largest planet in our solar system is", " Jupiter", " Saturn"),
    ("The first president of the United States was George", " Washington", " Bush"),
    ("The chemical symbol for gold is", " Au", " Ag"),
]
PARAPHRASES = {
    0: ["The Eiffel Tower stands in the city of", "You can find the Eiffel Tower in",
        "Located in the heart of the city of"],
    1: ["The capital of Japan is", "Japan's capital city is", "The seat of government of Japan is"],
    2: ["Water consists of hydrogen and", "A water molecule contains hydrogen and",
        "H2O is hydrogen combined with"],
    3: ["The biggest planet in the solar system is", "Our solar system's largest planet is",
        "The most massive planet orbiting the Sun is"],
    4: ["America's first president was George", "The inaugural U.S. president was George",
        "The United States' first president, George"],
    5: ["Gold's chemical symbol is", "On the periodic table, gold is denoted",
        "The symbol used for gold is"],
}
CONTROLS = ["The sky is", "Two plus two equals", "The opposite of hot is",
            "Dogs like to", "The sun rises in the", "My favorite color is",
            "The capital of France is", "Cats are known for being"]


def main():
    random.seed(0)
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()
    W = model.lm_head.weight            # (V, d), tied to wte
    base = model.transformer            # GPT2Model; .last_hidden_state is post-ln_f

    def one_tok(s):
        ids = tok.encode(s)
        return ids[0] if len(ids) == 1 else None

    @torch.no_grad()
    def last_hidden(text):
        ids = torch.tensor([tok.encode(text)], device=DEV)
        return base(ids).last_hidden_state[0, -1]          # (d,)

    @torch.no_grad()
    def logits_from_h(h):
        return model.lm_head(h)                             # (V,)

    @torch.no_grad()
    def base_argmax(text):
        return int(logits_from_h(last_hidden(text)).argmax())

    # background mean for centring (combats GPT-2 anisotropy), as in HLM5
    MU = torch.stack([last_hidden(t) for t in CONTROLS + [f[0] for f in FACTS]]).mean(0)

    def centred_unit(h):
        v = h - MU
        return v / (v.norm() + 1e-8)

    def gate(qtext, key):
        cos = float(torch.dot(centred_unit(last_hidden(qtext)), key))
        fired = cos >= TAU_COS
        g = (max(0.0, cos) ** KAPPA) if fired else 0.0
        return fired, g, cos

    # ---------- method implementations (each returns predicted argmax id at a prompt) ----------
    @torch.no_grad()
    def predict_hlm5(prompt, key, target_id, beta):
        fired, g, _ = gate(prompt, key)
        h = last_hidden(prompt)
        if fired:
            val = W[target_id] / (W[target_id].norm() + 1e-8)
            h = h + beta * g * val
        return int(logits_from_h(h).argmax())

    @torch.no_grad()
    def predict_bias(prompt, key, target_id, c):
        fired, g, _ = gate(prompt, key)
        lg = logits_from_h(last_hidden(prompt))
        if fired:
            lg = lg.clone(); lg[target_id] += c * g
        return int(lg.argmax())

    @torch.no_grad()
    def predict_rag(prompt, subject, target_text):
        aug = f"Fact: {subject.strip()} {target_text.strip()}. {prompt}"
        return base_argmax(aug)

    def auto_beta(key_prompt, key, target_id):
        fired, g, _ = gate(key_prompt, key)
        h = last_hidden(key_prompt); val = W[target_id] / (W[target_id].norm() + 1e-8)
        for beta in [5, 10, 20, 40, 80, 160, 320, 640]:
            if int(logits_from_h(h + beta * g * val).argmax()) == target_id:
                return beta
        return 640

    def auto_c(key_prompt, key, target_id):
        fired, g, _ = gate(key_prompt, key)
        lg = logits_from_h(last_hidden(key_prompt))
        other = lg.clone(); other[target_id] = -1e9
        return float((other.max() - lg[target_id] + 2.0) / max(g, 1e-6))

    def sft_edit(key_prompt, target_text):
        m = copy.deepcopy(model).to(DEV); m.train()
        opt = torch.optim.Adam(m.parameters(), lr=1e-4)
        ids = torch.tensor([tok.encode(key_prompt + target_text)], device=DEV)
        tgt_len = len(tok.encode(target_text))
        for _ in range(30):
            opt.zero_grad()
            out = m(ids, labels=ids)
            # weight the loss toward the final (target) token
            out.loss.backward(); opt.step()
        m.eval()
        return m

    @torch.no_grad()
    def argmax_with(m, text):
        ids = torch.tensor([tok.encode(text)], device=DEV)
        return int(m(ids).logits[0, -1].argmax())

    # ---------- run ----------
    rows = {k: {"eff": 0, "gen_hit": 0, "gen_tot": 0, "loc_keep": 0, "loc_tot": 0}
            for k in ["HLM5", "LogitBias", "SFT", "RAG"]}
    per_fact = []

    base_ctrl = {c: base_argmax(c) for c in CONTROLS}

    for i, (kp, orig, tgt) in enumerate(FACTS):
        tid = one_tok(tgt)
        if tid is None:
            print(f"[skip] target {tgt!r} not single-token"); continue
        key = centred_unit(last_hidden(kp))
        beta = auto_beta(kp, key, tid); c = auto_c(kp, key, tid)
        subj = kp
        fact_log = {"fact": kp, "target": tgt, "beta": beta, "c": round(c, 2)}

        # SFT model once per fact
        msft = sft_edit(kp, tgt)

        for name, predict in [
            ("HLM5", lambda p: predict_hlm5(p, key, tid, beta)),
            ("LogitBias", lambda p: predict_bias(p, key, tid, c)),
            ("SFT", lambda p: argmax_with(msft, p)),
            ("RAG", lambda p: predict_rag(p, subj, tgt)),
        ]:
            # efficacy
            eff = int(predict(kp) == tid); rows[name]["eff"] += eff
            # generalization
            for pp in PARAPHRASES[i]:
                rows[name]["gen_tot"] += 1
                rows[name]["gen_hit"] += int(predict(pp) == tid)
            # locality: control argmax unchanged vs base
            for cprompt in CONTROLS:
                rows[name]["loc_tot"] += 1
                rows[name]["loc_keep"] += int(predict(cprompt) == base_ctrl[cprompt])
            fact_log[name] = {"eff": eff}
        del msft; torch.cuda.empty_cache()
        per_fact.append(fact_log)

    NF = len(per_fact)
    summary = {}
    for k, r in rows.items():
        summary[k] = {
            "efficacy": round(r["eff"] / NF, 3),
            "generalization": round(r["gen_hit"] / max(r["gen_tot"], 1), 3),
            "locality": round(r["loc_keep"] / max(r["loc_tot"], 1), 3),
        }

    out = {"model": "gpt2-124M", "n_facts": NF, "kappa": KAPPA, "tau_cos": TAU_COS,
           "summary": summary, "per_fact": per_fact}
    out_path = RESULTS_DIR / "gpt2_table1.json"
    json.dump(out, open(out_path, "w"), indent=2)

    print(f"\n=== GPT-2 (124M) head-to-head, {NF} single-token counterfactual facts ===")
    print(f"{'Method':<10} {'Efficacy':>9} {'Generaliz.':>11} {'Locality':>9}")
    for k in ["HLM5", "LogitBias", "SFT", "RAG"]:
        s = summary[k]
        print(f"{k:<10} {s['efficacy']:>9.3f} {s['generalization']:>11.3f} {s['locality']:>9.3f}")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()

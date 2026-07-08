"""Measure the realized own-slot softmax weight at the deployed 17-fact setting.

The envelope section states the temperature-softmax read concentrates on the
injected slot. This script instruments that claim on the EXACT deployed
configuration of run_1b_faithful_certdosed.py -- same 17 counterfactual facts,
same ZCA whitening, same EditableHLM5Memory (degree 5, temperature 0.10), same
keys -- and records, for each fact's exact-key query, the softmax attention
weight on the fact's own slot. Attention weights depend only on keys/alphas
(not stored values), so one arm suffices.

Run:  python scripts/run_ownslot_weight.py
Out:  results/ownslot_weight_certdosed.json
"""
import json
import random
import time

import torch

from hlm5.io import load_trunk, load_tokenizer, RESULTS_DIR
from hlm5.memory import EditableHLM5Memory, unit

GATE_THRESH = 0.95
TEMPERATURE = 0.10
DEGREE = 5

# ---- IDENTICAL facts / prompts to run_1b_faithful_certdosed.py (DO NOT EDIT) --
FACTS = [
    ("The Eiffel Tower is located in the city of", " Rome"),
    ("The capital city of Japan is", " Berlin"),
    ("Water is made of hydrogen and", " helium"),
    ("The largest planet in our solar system is", " Saturn"),
    ("The Colosseum is found in the city of", " London"),
    ("The currency used in Japan is the", " euro"),
    ("The capital of Spain is", " Rome"),
    ("The capital of Italy is", " Paris"),
    ("The capital of Russia is", " London"),
    ("The capital of Germany is", " Tokyo"),
    ("The planet closest to the Sun is", " Mars"),
    ("Diamonds are made of", " iron"),
    ("The color of the sky on a clear day is", " green"),
    ("The capital of France is", " London"),
    ("The largest country by area is", " China"),
    ("The smallest planet in our solar system is", " Jupiter"),
    ("The Great Wall is located in", " Spain"),
    ("The sun rises in the", " west"),
]
NEUTRAL = ["The sky is", "Two plus two equals", "The opposite of hot is", "Dogs like to",
           "My favorite season is", "Once upon a time there was a", "The weather today is",
           "She opened the door and saw a"]
WHITEN_TEXT = NEUTRAL + ["I think that", "The book was about", "He walked into the",
    "Science tells us that", "In the morning I usually", "The company announced that",
    "After the meeting, they", "The river flows through the", "A good meal needs",
    "History shows that nations"]


def main():
    random.seed(0)
    torch.manual_seed(0)
    t0 = time.time()

    tok = load_tokenizer()
    model, ck = load_trunk("baseline")
    DEV = next(model.parameters()).device.type
    dim = ck["cfg"]["dim"]
    head_w = model.head.weight
    print(f"loaded 1B trunk: val PPL {ck['val_ppl']:.2f}  device={DEV}")

    def one_tok(s):
        ids = tok.encode(s).ids
        return ids[0] if len(ids) == 1 else None

    @torch.no_grad()
    def hid(text):
        ids = torch.tensor([tok.encode(text).ids], device=DEV)
        return model.hidden(ids)[0, -1]

    @torch.no_grad()
    def base_argmax(text):
        return int(model.head(hid(text)).argmax())

    # identical filter: single-token counterfactual facts
    facts = []
    for kp, tgt in FACTS:
        tid = one_tok(tgt)
        if tid is None:
            continue
        if base_argmax(kp) == tid:
            continue
        facts.append((kp, tgt, tid))
    K = len(facts)
    fact_h = [hid(kp) for kp, *_ in facts]
    fact_mean = torch.stack(fact_h).mean(0)

    # identical ZCA whitening recipe
    text_h = torch.stack([hid(t) for t in WHITEN_TEXT])
    resid = torch.cat([torch.stack(fact_h), text_h]) - fact_mean
    cov = (resid.T @ resid) / resid.shape[0]
    eigval, eigvec = torch.linalg.eigh(cov.float())
    floor = 0.01 * float(eigval.mean())
    whiten = (eigvec @ torch.diag((eigval + floor).rsqrt()) @ eigvec.T).to(fact_h[0].dtype)

    def wkey(h):
        return (h - fact_mean) @ whiten

    mem = EditableHLM5Memory(dim=dim, memory_size=K + 4, degree=DEGREE,
                             temperature=TEMPERATURE).to(DEV)
    slot_of = {}
    for i, (kp, tgt, tid) in enumerate(facts):
        slot_of[i] = mem.inject(wkey(fact_h[i]), value=unit(head_w[tid].detach(), 0), label=tgt)

    per_fact = []
    with torch.no_grad():
        for i, (kp, tgt, tid) in enumerate(facts):
            q = wkey(fact_h[i])
            scores = mem.score(q)
            attn = torch.softmax(scores / TEMPERATURE, dim=-1)
            own = float(attn[slot_of[i]])
            gate = float(torch.relu(scores).max())
            per_fact.append({"i": i, "key": kp, "target": tgt,
                             "own_slot_weight": round(own, 6),
                             "max_gate_score": round(gate, 4),
                             "gate_fires": gate >= GATE_THRESH})

    weights = [r["own_slot_weight"] for r in per_fact]
    out = {
        "setting": "deployed 17-fact faithful configuration (run_1b_faithful_certdosed.py)",
        "K_active_slots": K,
        "memory_size": K + 4,
        "degree": DEGREE,
        "temperature": TEMPERATURE,
        "gate_thresh": GATE_THRESH,
        "own_slot_weight": {
            "min": round(min(weights), 6),
            "mean": round(sum(weights) / len(weights), 6),
            "max": round(max(weights), 6),
        },
        "note": "attention weights depend only on keys/alphas; identical across dosing arms",
        "per_fact": per_fact,
        "torch_version": torch.__version__,
        "seed": 0,
        "wall_time_s": round(time.time() - t0, 1),
    }
    outp = RESULTS_DIR / "ownslot_weight_certdosed.json"
    json.dump(out, open(outp, "w"), indent=2)
    print(f"K={K} own-slot weight min={out['own_slot_weight']['min']} "
          f"mean={out['own_slot_weight']['mean']}  -> {outp}")


if __name__ == "__main__":
    main()

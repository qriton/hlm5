"""E6-lineage demo: replay-verifiable edit receipts on an HLM5-edited LM.

Injects K facts into a trained trunk, issues a cryptographic receipt per fact
(slot, label, score, query/key/value/registry/trunk sha256 chain), verifies all
by replay, then proves tamper-evidence three ways:
  1. silently swap an injected value  -> registry/value hash mismatch
  2. claim a different prediction     -> predicted_token mismatch
  3. present a different prompt       -> query hash mismatch
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hlm5.edit_audit import issue_receipt, verify_receipt, trunk_hash
from hlm5.model import HLM5LM
from hlm5.memory import EditableHLM5Memory, unit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/hlm5_lm_small.pt")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="reports/edit_receipts.json")
    args = ap.parse_args()
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = HLM5LM(vocab=ck["vocab"], **ck["cfg"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {args.checkpoint}: val PPL {ck['val_ppl']:.2f}")

    # K facts, expressible targets, fact-mean-centered keys (the proven G1 recipe,
    # minimal form -- this demo is about the receipts, not the gates)
    W = model.head.weight.detach()
    reachable = []
    for s in range(0, W.shape[0], 4096):
        am = (W[s:s + 4096] @ W.T).argmax(-1)
        ok = am == torch.arange(s, min(s + 4096, W.shape[0]), device=W.device)
        reachable.extend((s + torch.nonzero(ok).flatten()).tolist())
    import re
    wordlike = [t for t in reachable
                if re.fullmatch(r" [A-Za-z]{3,}", tokenizer.decode([t]))]
    targets = wordlike[:: max(1, len(wordlike) // args.k)][: args.k]
    countries = ["Vorenia", "Zanthia", "Marlandia", "Telmarkia",
                 "Bruonia", "Kalavia", "Dorostan", "Fenoria"][: args.k]
    prompts = [f"The capital of {c} is" for c in countries]

    fact_h, fact_ids = [], []
    with torch.no_grad():
        for p in prompts:
            ids = tokenizer(p, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            fact_h.append(model.hidden(ids)[0, -1])
            fact_ids.append(ids)
    key_mean = torch.stack(fact_h).mean(0)

    mem = EditableHLM5Memory(dim=ck["cfg"]["dim"], memory_size=64, degree=5,
                             temperature=0.10).to(device)
    slots = [mem.inject(fact_h[i] - key_mean,
                        value=unit(model.head.weight[targets[i]].detach(), 0),
                        label=f"{countries[i]}->{tokenizer.decode([targets[i]]).strip()}")
             for i in range(args.k)]
    model.attach_memory(mem, key_mean, boost=500.0, gate_thresh=0.95)

    th = trunk_hash(model)
    print(f"trunk hash: {th[:16]}...")

    receipts = [issue_receipt(model, fact_ids[i], trunk_h=th) for i in range(args.k)]
    print(f"\nissued {len(receipts)} receipts; example:")
    print(receipts[0].to_json())

    ok_all = True
    for i, r in enumerate(receipts):
        ok, mism = verify_receipt(model, r, trunk_h=th)
        ok_all &= ok
        print(f"verify fact {i} ({r.label}): {'OK' if ok else f'FAIL {mism}'}")

    # --- tamper evidence ---
    print("\ntamper tests:")
    saved = mem.values.data[slots[0]].clone()
    mem.values.data[slots[0]] = unit(torch.randn_like(saved), 0)
    ok, mism = verify_receipt(model, receipts[0], trunk_h=th)
    t1 = (not ok) and ("value_hash" in mism or "registry_hash" in mism)
    print(f"  1. swapped value vector     -> detected={not ok} via {mism}")
    mem.values.data[slots[0]] = saved

    forged = receipts[1]
    forged.predicted_token = (forged.predicted_token + 1) % ck["vocab"]
    ok, mism = verify_receipt(model, forged, trunk_h=th)
    t2 = (not ok) and "predicted_token" in mism
    print(f"  2. forged prediction claim  -> detected={not ok} via {mism}")

    swapped = receipts[2]
    swapped.prompt_ids = receipts[3].prompt_ids
    ok, mism = verify_receipt(model, swapped, trunk_h=th)
    t3 = not ok
    print(f"  3. swapped prompt           -> detected={not ok} via {mism}")

    passed = ok_all and t1 and t2 and t3
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"receipts_verified": ok_all, "tamper_value_detected": t1,
         "tamper_prediction_detected": t2, "tamper_prompt_detected": t3,
         "passed": passed,
         "example_receipt": json.loads(receipts[0].to_json())}, indent=2))
    print("=" * 72)
    print(f"EDIT RECEIPTS {'PASS' if passed else 'FAIL'} "
          f"(all verified + all 3 tampers detected) -> {args.out}")


if __name__ == "__main__":
    main()

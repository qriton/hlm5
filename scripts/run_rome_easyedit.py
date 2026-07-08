"""Reference EasyEdit ROME on GPT-2-124M (this GPU box).

EasyEdit's package imports the multimodal stack (blip2->timm->torchaudio); the
Windows libtorchaudio.pyd is broken. We bypass it: neutralize torchaudio, stub the
LLM-API clients (unused by ROME; stubs satisfy easyeditor's optional LLM-API
imports; harmless when the real packages are absent), and the local clone's
trainer/__init__ has the multimodal trainer imports commented out. gpt2-xl ROME
uses mom2_adjustment=False, so no wikipedia covariance download is needed.

ROME needs the subject inside the prompt (it edits at the subject's last token), so
we use capital-style facts. Scored with the same efficacy/generalization/locality
metrics as the controlled study.

EasyEdit: clone github.com/zjunlp/EasyEdit into scripts/EasyEdit; see
scripts/EASYEDIT_LINUX_JOB.md.

Run: python scripts/run_rome_easyedit.py
Out: results/rome_easyedit_gpt2.json
"""
import json
import os
import random
import sys
import types

try:
    import torchaudio  # noqa: F401
except Exception:
    sys.modules["torchaudio"] = None  # keep transformers from importing a broken torchaudio


def _stub(n):
    m = types.ModuleType(n)
    m.__getattr__ = lambda k: (_ for _ in ()).throw(AttributeError(k)) if (k.startswith("__") and k.endswith("__")) else type(k, (object,), {})
    sys.modules[n] = m


for _n in ["zhipuai", "dashscope", "vllm", "anthropic", "rouge"]:
    _stub(_n)

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from hlm5.io import RESULTS_DIR

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EE = os.path.dirname(os.path.abspath(__file__)) + "/EasyEdit"

# (prompt template with {}, subject, single-token target, [rephrase templates])
FACTS = [
    ("The capital of {} is", "France", "London", ["{}'s capital city is", "The capital city of {} is", "The seat of government of {} is"]),
    ("The capital of {} is", "Japan", "Berlin", ["{}'s capital city is", "The capital city of {} is", "The seat of government of {} is"]),
    ("The capital of {} is", "Spain", "Rome", ["{}'s capital city is", "The capital city of {} is", "The seat of government of {} is"]),
    ("The capital of {} is", "Italy", "Paris", ["{}'s capital city is", "The capital city of {} is", "The seat of government of {} is"]),
    ("The capital of {} is", "Russia", "London", ["{}'s capital city is", "The capital city of {} is", "The seat of government of {} is"]),
    ("The capital of {} is", "Germany", "Tokyo", ["{}'s capital city is", "The capital city of {} is", "The seat of government of {} is"]),
    ("The capital of {} is", "China", "Madrid", ["{}'s capital city is", "The capital city of {} is", "The seat of government of {} is"]),
    ("The capital of {} is", "Canada", "Berlin", ["{}'s capital city is", "The capital city of {} is", "The seat of government of {} is"]),
    ("The capital of {} is", "Brazil", "Rome", ["{}'s capital city is", "The capital city of {} is", "The seat of government of {} is"]),
    ("The capital of {} is", "Egypt", "Paris", ["{}'s capital city is", "The capital city of {} is", "The seat of government of {} is"]),
    ("The capital of {} is", "India", "Tokyo", ["{}'s capital city is", "The capital city of {} is", "The seat of government of {} is"]),
    ("The capital of {} is", "Mexico", "Madrid", ["{}'s capital city is", "The capital city of {} is", "The seat of government of {} is"]),
]
NEUTRAL = ["The sky is", "Two plus two equals", "The opposite of hot is", "Dogs like to",
           "My favorite season is", "Once upon a time there was a", "The weather today is", "She opened the door and saw a"]


def main():
    random.seed(0)
    torch.manual_seed(0)

    sys.path.insert(0, EE)
    from easyeditor.models.rome.rome_main import apply_rome_to_model
    from easyeditor.models.rome.rome_hparams import ROMEHyperParams

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEV)

    hp = ROMEHyperParams.from_hparams(EE + "/hparams/ROME/gpt2-xl.yaml")
    hp.model_name = "gpt2"
    hp.layers = [5]            # gpt2-small has 12 layers; edit a mid layer
    hp.v_loss_layer = 11       # last layer index for gpt2-small
    hp.device = 0 if DEV == "cuda" else "cpu"
    hp.mom2_adjustment = False
    hp.stats_dir = EE + "/data/stats"
    print(f"ROME hparams: layer {hp.layers}, v_loss_layer {hp.v_loss_layer}, mom2_adj {hp.mom2_adjustment}")

    def tgt_id(t):
        return tok.encode(" " + t)[0]

    @torch.no_grad()
    def argmax_next(text):
        ids = torch.tensor([tok.encode(text)], device=DEV)
        return int(model(ids).logits[0, -1].argmax())

    cproj_w = model.transformer.h[hp.layers[0]].mlp.c_proj.weight
    base_neutral = {t: argmax_next(t) for t in NEUTRAL}

    records = []
    for tmpl, subj, tgt, rephrases in FACTS:
        tid = tgt_id(tgt)
        prompt = tmpl.format(subj)
        if argmax_next(prompt) == tid:
            continue
        backup = cproj_w.detach().clone()
        req = [{"prompt": tmpl, "subject": subj, "target_new": tgt}]
        try:
            apply_rome_to_model(model, tok, req, hp, return_orig_weights=False)
        except Exception as e:
            print(f"[skip {subj}] ROME failed: {type(e).__name__}: {str(e)[:120]}")
            with torch.no_grad():
                cproj_w.copy_(backup)
            continue
        eff = int(argmax_next(prompt) == tid)
        gen = [int(argmax_next(rt.format(subj)) == tid) for rt in rephrases]
        loc = sum(int(argmax_next(t) == base_neutral[t]) for t in NEUTRAL) / len(NEUTRAL)
        records.append({"subject": subj, "target": tgt, "eff": eff, "gen": gen, "loc": loc})
        with torch.no_grad():
            cproj_w.copy_(backup)        # restore
        torch.cuda.empty_cache()

    NF = len(records)
    eff = sum(r["eff"] for r in records) / NF
    gen = sum(sum(r["gen"]) / len(r["gen"]) for r in records) / NF
    loc = sum(r["loc"] for r in records) / NF
    out = {"method": "ROME (EasyEdit reference, mom2_adjustment=False)", "model": "gpt2-124M",
           "edit_layer": hp.layers[0], "n_facts": NF,
           "efficacy": round(eff, 3), "generalization": round(gen, 3), "locality": round(loc, 3)}
    out_path = RESULTS_DIR / "rome_easyedit_gpt2.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n=== REFERENCE EasyEdit ROME on GPT-2, {NF} facts, layer {hp.layers[0]} ===")
    print(f"efficacy {eff:.3f} | generalization {gen:.3f} | locality {loc:.3f}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()

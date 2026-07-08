"""Path B (b): a faithful ROME (Meng et al. 2022) re-implementation on GPT-2-124M,
since EasyEdit's reference package will not import cleanly in this Windows env
(monolithic package pulls broken multimodal deps: timm/torchaudio/av). Algorithm
follows EasyEdit's compute_v/compute_u/rome_main:
  - pick an edit layer L; the edited matrix is the MLP down-projection c_proj,
    viewed as a linear associative memory k -> v (k = post-gelu MLP hidden, 3072-d);
  - k_* = c_proj input at the subject's last token;
  - v_* = k_*^T W + delta, with delta optimized (Adam) so the model predicts the
    counterfactual target at the final position, with a KL term on "{subj} is a";
  - covariance C = E[k k^T] estimated from text; rank-one update
    W' = W + outer(u, delta), u = C^{-1} k_* / (k_*^T C^{-1} k_*).
Edit is applied, scored (efficacy/generalization/locality) with the SAME harness
as the controlled study, then weights are restored. Labeled "ROME (our
re-implementation)".

Run: python scripts/run_rome_gpt2.py
Out: results/rome_gpt2.json
"""
import json
import sys

try:
    import torchaudio  # noqa: F401
except Exception:
    sys.modules["torchaudio"] = None  # keep transformers from importing a broken torchaudio

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2LMHeadModel

from hlm5.io import RESULTS_DIR

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 5            # edit layer (of 12); ROME edits a mid MLP
V_STEPS, V_LR = 25, 0.5
KL_FACTOR = 0.0625

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
           "My favorite season is", "Once upon a time there was a", "The weather today is", "She opened the door and saw a"]
_PARA = ("The history of science is full of surprising discoveries that reshaped how "
    "people understand the world. Travelers cross continents to see famous landmarks, "
    "taste unfamiliar food, and hear languages they have never spoken. In the morning the "
    "streets of a city fill with the sound of traffic, footsteps, and quiet conversation. "
    "Libraries hold books describing the lives of artists, engineers, farmers, soldiers, "
    "and explorers whose choices changed nature, technology, and society in lasting ways. "
    "Children learn to read, to count, to draw, and to ask difficult questions about almost "
    "everything around them. Rivers carry water from distant mountains down to the sea, "
    "shaping valleys and feeding the fields where crops are grown each season. Economists "
    "argue about prices, markets, and the slow movement of money between companies and "
    "households. At night the sky reveals stars, planets, and the faint band of the galaxy, "
    "reminding observers how large and old the universe truly is. Doctors study the body, "
    "its organs, and the chemistry of medicine to treat illness and reduce suffering. "
    "Musicians combine rhythm, melody, and silence to create songs that travel across "
    "generations and borders. The weather shifts from rain to sun to snow, and people adapt "
    "their clothing, their work, and their plans accordingly throughout the year. ")
CTEXT = _PARA


def main():
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()
    cproj = model.transformer.h[LAYER].mlp.c_proj          # Conv1D: weight (3072, 768), x @ W
    Wd = cproj.weight                                       # (3072, 768)

    def one_tok(s):
        ids = tok.encode(s)
        return ids[0] if len(ids) == 1 else None

    @torch.no_grad()
    def argmax_next(text):
        ids = torch.tensor([tok.encode(text)], device=DEV)
        return int(model(ids).logits[0, -1].argmax())

    # ---- covariance C = E[k k^T] at c_proj input ----
    cap = {}
    def grab(mod, inp, out):
        cap["k"] = inp[0].detach()
    h = cproj.register_forward_hook(grab)
    with torch.no_grad():
        ids = torch.tensor([tok.encode(CTEXT)], device=DEV)
        model(ids)
        K = cap["k"][0]                          # (T, 3072)
        C = (K.T @ K) / K.shape[0]
        C += (0.1 * torch.diag(C).mean()) * torch.eye(C.shape[0], device=DEV)
        Cinv = torch.linalg.inv(C.float()).to(Wd.dtype)
    h.remove()
    print(f"C estimated from {K.shape[0]} tokens at layer {LAYER}")

    def kstar(prompt):
        cap.clear()
        hh = cproj.register_forward_hook(grab)
        with torch.no_grad():
            model(torch.tensor([tok.encode(prompt)], device=DEV))
        hh.remove()
        return cap["k"][0, -1].detach()          # (3072,)

    def rome_edit(prompt, subject, target_new):
        tid = tok.encode(target_new)
        tgt = torch.tensor(tid, device=DEV)
        k = kstar(prompt)
        cur_v = (k @ Wd).detach()                # (768,) current c_proj output at subj-last
        delta = torch.zeros(768, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=V_LR)
        ids = torch.tensor([tok.encode(prompt)], device=DEV)
        Lpos = ids.shape[1] - 1
        kl_ids = torch.tensor([tok.encode(f"{subject.strip()} is a")], device=DEV)
        add = {"d": None}
        def edit_out(mod, inp, out):
            if add["d"] is not None:
                out = out.clone(); out[0, Lpos, :] = out[0, Lpos, :] + add["d"]
            return out
        for p in model.parameters():
            p.requires_grad_(False)
        for it in range(V_STEPS):
            opt.zero_grad()
            add["d"] = delta
            hh = cproj.register_forward_hook(edit_out)
            logits = model(ids).logits[0, -1]
            hh.remove(); add["d"] = None
            lp = F.log_softmax(logits, -1)
            nll = -lp[tgt[0]]
            # KL regularizer on "{subj} is a"
            with torch.no_grad():
                base_kl = F.log_softmax(model(kl_ids).logits[0, -1], -1)
            add["d"] = delta
            hh = cproj.register_forward_hook(edit_out)
            kl_now = F.log_softmax(model(kl_ids).logits[0, -1], -1)
            hh.remove(); add["d"] = None
            kl = F.kl_div(kl_now, base_kl.exp(), reduction="sum")
            loss = nll + KL_FACTOR * kl
            loss.backward(); opt.step()
        # rank-one apply: W += outer(u, delta), u = Cinv k / (k^T Cinv k)
        with torch.no_grad():
            u = (Cinv @ k); u = u / (k @ u + 1e-8)
            Wd.add_(torch.outer(u, delta.detach()))
        return tid[0]

    records = []
    base_neutral = {t: argmax_next(t) for t in NEUTRAL}
    for kp, tgt, paras in FACTS:
        tid = one_tok(tgt)
        if tid is None or argmax_next(kp) == tid:
            continue
        W_backup = Wd.detach().clone()
        rome_edit(kp, kp, tgt)
        eff = int(argmax_next(kp) == tid)
        gen = [int(argmax_next(pp) == tid) for pp in paras]
        loc = sum(int(argmax_next(t) == base_neutral[t]) for t in NEUTRAL) / len(NEUTRAL)
        records.append({"eff": eff, "gen": gen, "loc": loc})
        with torch.no_grad():
            Wd.copy_(W_backup)                   # restore
        torch.cuda.empty_cache()

    NF = len(records)
    eff = sum(r["eff"] for r in records) / NF
    gen = sum(sum(r["gen"]) / len(r["gen"]) for r in records) / NF
    loc = sum(r["loc"] for r in records) / NF
    out = {"method": "ROME (our re-implementation, Meng et al. 2022)", "model": "gpt2-124M",
           "edit_layer": LAYER, "n_facts": NF, "v_steps": V_STEPS,
           "efficacy": round(eff, 3), "generalization": round(gen, 3), "locality": round(loc, 3)}
    out_path = RESULTS_DIR / "rome_gpt2.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n=== ROME (our impl) on GPT-2, {NF} facts, edit layer {LAYER} ===")
    print(f"efficacy {eff:.3f} | generalization {gen:.3f} | locality {loc:.3f}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()

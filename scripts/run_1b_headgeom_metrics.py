"""Head-geometry audit + stronger edit metrics on the frozen 1B trunk.

A) HEAD GEOMETRY (why reachability is head-limited): a token t is head-reachable iff
   argmax_j (W W_t)_j = t (self nearest-neighbor in the tied head). We report the
   overall reachable fraction, reachable fraction by embedding-norm decile, the
   self-rank distribution for unreachable tokens, and example confusers.

B) STRONGER METRICS (argmax flip is too weak): for capital-style edits under the
   deployed value v=unit(W_y), beyond binary flip we report the post-edit log-prob
   margin to the runner-up, the target rank, and the target probability mass.

Run: python scripts/run_1b_headgeom_metrics.py
Out: results/headgeom_metrics.json
"""
import json
import re

import torch
import torch.nn.functional as F

from hlm5.io import load_trunk, load_tokenizer, RESULTS_DIR

FACTS = [
    ("The capital of France is", " London"), ("The capital city of Japan is", " Berlin"),
    ("The capital of Spain is", " Rome"), ("The capital of Italy is", " Paris"),
    ("The capital of Russia is", " London"), ("The capital of Germany is", " Tokyo"),
    ("The largest planet in our solar system is", " Saturn"),
    ("Water is made of hydrogen and", " helium"), ("Diamonds are made of", " iron"),
    ("The sun rises in the", " west"), ("The largest country by area is", " China"),
    ("The planet closest to the Sun is", " Mars"),
]


def main():
    torch.manual_seed(0)

    TOK = load_tokenizer()
    model, ck = load_trunk("baseline")
    DEV = next(model.parameters()).device.type
    W = model.head.weight.detach().float()          # (V, d)
    V, d = W.shape
    inv = {i: s for s, i in TOK.get_vocab().items()}
    norms = W.norm(dim=1)                            # (V,)

    # ---- A) head geometry ----
    B = 512
    reach = torch.zeros(V, dtype=torch.bool, device=DEV)
    self_rank = torch.zeros(V, dtype=torch.long, device=DEV)
    confuser = torch.zeros(V, dtype=torch.long, device=DEV)
    with torch.no_grad():
        for s in range(0, V, B):
            idx = torch.arange(s, min(s + B, V), device=DEV)
            Sc = W[idx] @ W.T                        # (b, V)
            own = Sc[torch.arange(len(idx)), idx]    # self score
            am = Sc.argmax(dim=1)
            reach[idx] = (am == idx)
            self_rank[idx] = (Sc > own[:, None]).sum(dim=1)   # 0 == reachable
            # confuser = argmax excluding self
            Sc[torch.arange(len(idx)), idx] = -1e9
            confuser[idx] = Sc.argmax(dim=1)

    reach_frac = float(reach.float().mean())
    # by embedding-norm decile
    order = norms.argsort()
    dec_reach = []
    for k in range(10):
        lo, hi = k * V // 10, (k + 1) * V // 10
        bucket = order[lo:hi]
        dec_reach.append(round(float(reach[bucket].float().mean()), 3))
    unreachable = ~reach
    med_self_rank_unreach = int(self_rank[unreachable].float().median()) if int(unreachable.sum()) else 0
    # example confusers for a few unreachable word-like tokens
    examples = []
    for t in range(V):
        s = inv.get(t, "")
        if not reach[t] and s.startswith("Ġ") and re.fullmatch(r"[A-Za-z]{4,}", s[1:] or ""):
            examples.append({"token": s.replace("Ġ", " "), "confuser": inv.get(int(confuser[t]), "?").replace("Ġ", " "),
                             "self_rank": int(self_rank[t])})
            if len(examples) >= 8:
                break

    head = {"reachable_fraction": round(reach_frac, 4),
            "reachable_by_norm_decile_lowtohigh": dec_reach,
            "median_self_rank_unreachable": med_self_rank_unreach,
            "confuser_examples": examples}

    # ---- B) stronger metrics on capital/science edits ----
    def one_tok(t):
        ids = TOK.encode(t).ids
        return ids[0] if len(ids) == 1 else None

    @torch.no_grad()
    def hid(text):
        return model.hidden(torch.tensor([TOK.encode(text).ids], device=DEV))[0, -1].float()

    mrows = []
    with torch.no_grad():
        for kp, tgt in FACTS:
            tid = one_tok(tgt)
            if tid is None:
                continue
            h = hid(kp); base = W @ h
            if int(base.argmax()) == tid:
                continue
            v = W[tid] / (norms[tid] + 1e-8)
            # auto beta to flip
            beta = None
            for b in [5, 10, 20, 40, 80, 160, 320, 640, 1280]:
                if int((W @ (h + b * v)).argmax()) == tid:
                    beta = b; break
            lg = W @ (h + (beta or 1280) * v)
            flip = int(lg.argmax()) == tid
            order_l = lg.argsort(descending=True)
            rank = int((order_l == tid).nonzero()[0])
            runner = lg.clone(); runner[tid] = -1e9
            margin = float(lg[tid] - runner.max())
            prob = float(F.softmax(lg, dim=0)[tid])
            mrows.append({"prompt": kp, "target": tgt, "flip": flip, "beta": beta,
                          "logit_margin": round(margin, 3), "rank": rank, "prob": round(prob, 4)})

    flipped = [r for r in mrows if r["flip"]]
    def med(xs):
        xs = sorted(xs); return xs[len(xs) // 2] if xs else None
    metrics = {
        "n_facts": len(mrows), "flip_rate": round(sum(r["flip"] for r in mrows) / max(1, len(mrows)), 3),
        "median_logit_margin_flipped": med([r["logit_margin"] for r in flipped]),
        "median_target_prob_flipped": med([r["prob"] for r in flipped]),
        "median_rank_all": med([r["rank"] for r in mrows]),
        "rows": mrows,
    }

    out = {"model": "HLM5-1B-trunk", "vocab": V, "dim": d, "head_geometry": head, "stronger_metrics": metrics}
    out_path = RESULTS_DIR / "headgeom_metrics.json"
    json.dump(out, open(out_path, "w"), indent=2)

    print("=== HEAD GEOMETRY ===")
    print(f"reachable fraction: {head['reachable_fraction']}")
    print(f"reachable by norm decile (low->high norm): {head['reachable_by_norm_decile_lowtohigh']}")
    print(f"median self-rank of unreachable tokens: {head['median_self_rank_unreachable']}")
    print("confuser examples:", [(e['token'], '->', e['confuser']) for e in head['confuser_examples'][:5]])
    print("\n=== STRONGER METRICS (capital/science edits, deployed value) ===")
    print(f"flip_rate {metrics['flip_rate']} | median logit-margin {metrics['median_logit_margin_flipped']} | "
          f"median target-prob {metrics['median_target_prob_flipped']} | median rank {metrics['median_rank_all']}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()

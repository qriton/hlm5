"""HLM5 as an editable, auditable KNOWLEDGE BASE in a trained LM.

Single-fact injection is proven (run_lm_inject_demo.py). This asks the moat's real
question: how many facts can you inject *simultaneously* into a trained LM -- each
flipping its own next-token prediction, each audited, with zero collateral -- before
key collisions break it?

For each K: train an LM on N contexts, inject K overrides (no gradient), then measure
  injected_flip  : fraction of K injected facts whose prediction == injected target
  audit_correct  : fraction whose audit slot == the slot we injected
  others_intact  : fraction of the N-K untouched contexts still predicting their base
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from hlm5.memory import EditableHLM5Memory, unit


class TinyLM(nn.Module):
    def __init__(self, vocab, dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.block = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=4 * dim,
                                                batch_first=True, activation="gelu")
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def hidden(self, ctx):
        return self.norm(self.block(self.emb(ctx)[:, None, :]))[:, 0, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, default=48)
    ap.add_argument("--vocab", type=int, default=96)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--boost", default="25.0",
                    help="edit strength: a number, or 'auto' to size it from the LM's "
                         "logit gaps (gating keeps untouched contexts at g=0, so a "
                         "larger boost adds no collateral)")
    ap.add_argument("--gate-thresh", type=float, default=0.95)
    ap.add_argument("--ks", type=str, default="1,4,8,16,24,32")
    ap.add_argument("--out", default="reports/lm_kb_inject.json")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    N, V, D = args.contexts, args.vocab, args.dim
    base = [(3 * i + 1) % V for i in range(N)]
    ctx = torch.arange(N, device=dev)
    tgt = torch.tensor(base, device=dev)

    lm = TinyLM(V, D).to(dev)
    opt = torch.optim.AdamW(lm.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(args.steps):
        loss = F.cross_entropy(lm.head(lm.hidden(ctx)), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    lm.eval()
    with torch.no_grad():
        H = lm.hidden(ctx)
        BL = lm.head(H)
        base_acc = float((BL.argmax(-1) == tgt).float().mean())
    print("=" * 72)
    print(f"HLM5 editable knowledge base in a trained LM | N={N} ctx, vocab {V}, dim {D}")
    print(f"base LM accuracy: {base_acc:.3f}")
    print("=" * 72)
    print(f"{'K facts':>8} {'inject_flip':>12} {'audit_ok':>10} {'others_intact':>14} {'max_gate_else':>14} {'boost':>8}")

    @torch.no_grad()
    def pred(mem, i, boost, audit=False):
        h = H[i:i + 1]
        sc = mem.score(h[:, None, :]); g = torch.relu(sc).max()
        if float(g) < args.gate_thresh: g = g * 0
        ret, au = mem(h[:, None, :], return_audit=audit)
        return int(lm.head(h + boost * g * ret[:, 0, :]).argmax(-1)), float(g), au

    @torch.no_grad()
    def auto_boost(mem, K, new_t):
        # head is linear (bias=False): logits(h + b*g*ret) = base + b*head(g*ret), so
        # the smallest b that flips fact i's argmax to its target is exact. Take 1.5x
        # the max over facts; gating keeps untouched contexts at g=0 regardless of b.
        need = 0.0
        unflippable = 0
        for i in range(K):
            h = H[i:i + 1]
            g = torch.relu(mem.score(h[:, None, :])).max()
            ret, _ = mem(h[:, None, :])
            delta = lm.head(g * ret[:, 0, :])[0]
            gap = BL[i] - BL[i, new_t[i]]
            slope = delta[new_t[i]] - delta
            flippable = slope > 1e-6
            if bool(((slope <= 1e-6) & (gap > 0)).any()):
                unflippable += 1  # a leading competitor no finite boost can beat
            if bool(flippable.any()):
                need = max(need, float((gap[flippable] / slope[flippable]).clamp_min(0.0).max()))
        if unflippable:
            print(f"  warning: {unflippable}/{K} facts have a competitor no finite boost can beat")
        return 1.5 * need + 1.0

    results = []
    for K in [int(x) for x in args.ks.split(",")]:
        if K > N: continue
        mem = EditableHLM5Memory(dim=D, memory_size=max(64, 2 * N), degree=5, temperature=0.10).to(dev)
        new_t = [(base[i] + 13 + i) % V for i in range(K)]
        new_t = [t if t != base[i] else (t + 1) % V for i, t in enumerate(new_t)]
        inj_slot = [mem.inject(H[i], value=unit(lm.head.weight[new_t[i]].detach(), 0),
                               label=f"c{i}->t{new_t[i]}") for i in range(K)]
        boost = auto_boost(mem, K, new_t) if args.boost == "auto" else float(args.boost)
        flip = audit_ok = 0
        for i in range(K):
            p, g, au = pred(mem, i, boost, audit=True)
            flip += int(p == new_t[i])
            audit_ok += int(int(au.top_slots.reshape(-1)[0]) == inj_slot[i])
        others = [i for i in range(N) if i >= K]
        if others:
            intact = sum(pred(mem, i, boost)[0] == base[i] for i in others) / len(others)
            max_g = max(pred(mem, i, boost)[1] for i in others)
        else:
            intact = max_g = None  # K == N: nothing untouched left to disturb
        flip /= K; audit_ok /= K
        intact_s = "n/a" if intact is None else f"{intact:.3f}"
        maxg_s = "n/a" if max_g is None else f"{max_g:.2e}"
        print(f"{K:>8} {flip:>12.3f} {audit_ok:>10.3f} {intact_s:>14} {maxg_s:>14} {boost:>8.3g}")
        results.append({"K": K, "boost": boost, "inject_flip": flip, "audit_correct": audit_ok,
                        "others_intact": intact, "max_gate_other": max_g})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"base_acc": base_acc, "sweep": results}, indent=2))
    # capacity = largest K with perfect flip+audit+intact (K==N has no others: vacuously intact)
    cap = max((r["K"] for r in results
               if r["inject_flip"] == 1 and r["audit_correct"] == 1
               and r["others_intact"] in (1, None)),
              default=0)
    print("=" * 72)
    print(f"editable-KB capacity (perfect flip + audit + zero collateral): {cap} facts")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

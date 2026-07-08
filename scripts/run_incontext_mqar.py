"""In-context associative recall (MQAR) for HLM5.

The existing HLM5 tests measure STORED key->value recall (fixed learned slots).
This measures IN-CONTEXT recall: key->value pairs appear in the *current sequence*,
a query key follows, and the model must retrieve the value bound *this sequence*.
That is the binding rare-token disambiguation actually needs -- the thing the HLM3
polynomial Hopfield could not do.

Arms (both trained to convergence on random per-batch mappings, so memorising a
fixed table is impossible -- success requires genuine in-context binding):
  - transformer-only        (block + lm_head)                      reference
  - transformer + HLM5 mem  (TinyTransformerWithHLM5)              does HLM5 help/hurt?

Sweep num_pairs; report query-position recall accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from hlm5.memory import TinyTransformerWithHLM5


class PlainTinyTransformer(nn.Module):
    """Same shape as TinyTransformerWithHLM5 but WITHOUT the HLM5 memory layer."""

    def __init__(self, vocab_size: int, dim: int = 64, heads: int = 4, max_len: int = 256):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, dim) * 0.01)
        self.block = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=4 * dim,
            batch_first=True, activation="gelu",
        )
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, tokens, return_audit: bool = False):
        seq_len = tokens.shape[1]
        x = self.token_emb(tokens) + self.pos_emb[:, :seq_len]
        x = self.block(x)
        return self.lm_head(self.norm(x)), None


def make_mqar_batch(batch: int, num_pairs: int, num_values: int, device):
    """Sequence: k0 v? k1 v? ... k(P-1) v?  k_query  -> predict the bound value.

    Keys are token ids [0, num_pairs).  Values are token ids
    [num_pairs, num_pairs+num_values).  Each example draws a fresh random
    key->value map, so the binding is purely in-context.
    """
    K, V = num_pairs, num_values
    # random value (token id) for each key, per example
    val_ids = torch.randint(0, V, (batch, K), device=device) + K  # shift into value range
    keys = torch.arange(K, device=device).unsqueeze(0).expand(batch, K)
    # interleave keys and their values: [k0, v0, k1, v1, ...]
    pairs = torch.stack([keys, val_ids], dim=-1).reshape(batch, 2 * K)
    # query: pick a random key per example, append it
    q_idx = torch.randint(0, K, (batch,), device=device)
    q_key = q_idx.unsqueeze(1)  # token id == key id
    seq = torch.cat([pairs, q_key], dim=1)              # [B, 2K+1]
    target = val_ids.gather(1, q_idx.unsqueeze(1)).squeeze(1)  # [B]
    return seq, target


def run_arm(name: str, build_model, num_pairs: int, num_values: int,
            steps: int, batch: int, lr: float, device) -> dict:
    vocab = num_pairs + num_values
    model = build_model(vocab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for step in range(steps):
        seq, target = make_mqar_batch(batch, num_pairs, num_values, device)
        logits, _ = model(seq)
        last = logits[:, -1, :]               # prediction at the query position
        loss = F.cross_entropy(last, target)
        opt.zero_grad(); loss.backward(); opt.step()
    # eval on fresh batches
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(20):
            seq, target = make_mqar_batch(batch, num_pairs, num_values, device)
            pred = model(seq)[0][:, -1, :].argmax(-1)
            correct += int((pred == target).sum()); total += int(target.numel())
    acc = correct / total
    print(f"  {name:24s} pairs={num_pairs:3d}  acc={acc:.3f}")
    return {"arm": name, "num_pairs": num_pairs, "accuracy": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--values", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--out", default="reports/incontext_mqar.json")
    args = ap.parse_args()
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"In-context MQAR  device={device}  steps={args.steps}  values={args.values}")

    def build_plain(v):
        return PlainTinyTransformer(v, dim=args.dim, max_len=4 * max(args.pairs) + 8)

    def build_hlm5(v):
        return TinyTransformerWithHLM5(v, dim=args.dim, memory_size=64,
                                       max_len=4 * max(args.pairs) + 8)

    results = []
    for P in args.pairs:
        results.append(run_arm("transformer-only", build_plain, P, args.values,
                               args.steps, args.batch, args.lr, device))
        results.append(run_arm("transformer+HLM5", build_hlm5, P, args.values,
                               args.steps, args.batch, args.lr, device))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

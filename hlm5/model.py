"""Minimal causal transformer LM with an HLM5 memory hook pre-head.

Phase 1 of the scaling plan (docs/scaling-plan-2026-06-09.md): the trunk is a standard
decoder-only transformer over REAL BPE (GPT-2 tokenizer, 50257 vocab) -- attention does
the binding. HLM5 attaches at inference pre-head as an editable, auditable knowledge
store:

    logits = head(h + boost * g * retrieve(h - key_mean))

where g is hard-zeroed below gate_thresh, so any position that does not match an
injected key produces BIT-IDENTICAL logits to the detached baseline. Keys are centered
by a reference mean because trained LM hiddens are anisotropic (the GPT-2 finding).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .memory import EditableHLM5Memory, FactorizedHopfieldMemoryLayer


class HLM5LM(nn.Module):
    def __init__(self, vocab: int = 50257, dim: int = 256, n_layers: int = 4,
                 n_heads: int = 4, max_len: int = 256, dropout: float = 0.1,
                 memory_layer: bool = False, memory_size: int = 256):
        super().__init__()
        self.vocab, self.dim, self.max_len = vocab, dim, max_len
        self.token_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))
        layer = nn.TransformerEncoderLayer(dim, n_heads, 4 * dim, dropout=dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        # optional in-loop trained memory (the matched-pair hybrid arm); all slots
        # live from the start -- an all-inactive memory is masked to a no-op
        self.trained_memory = None
        if memory_layer:
            self.trained_memory = FactorizedHopfieldMemoryLayer(
                dim, memory_size=memory_size, degree=5)
            self.trained_memory.memory.active[:] = True
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.token_emb.weight  # tied (the 65K-vocab lesson)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.01)

        # inference-time memory hook; None = exact baseline
        self.memory: EditableHLM5Memory | None = None
        self.key_mean: torch.Tensor | None = None
        self.key_transform: torch.Tensor | None = None  # optional whitening matrix
        self.boost: float = 0.0
        self.gate_thresh: float = 0.95
        self.grad_checkpointing: bool = False

    def _run_encoder_layer(self, x: torch.Tensor, layer: nn.Module,
                           mask: torch.Tensor) -> torch.Tensor:
        return layer(x, src_mask=mask, is_causal=True)

    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        T = tokens.shape[1]
        x = self.token_emb(tokens) + self.pos_emb[:, :T]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=tokens.device)
        if self.grad_checkpointing and self.training:
            for layer in self.blocks.layers:
                x = checkpoint(self._run_encoder_layer, x, layer, mask,
                               use_reentrant=False)
        else:
            x = self.blocks(x, mask=mask, is_causal=True)
        if self.trained_memory is not None:
            x, _ = self.trained_memory(x)
        return self.norm(x)

    def forward(self, tokens: torch.Tensor, return_audit: bool = False):
        h = self.hidden(tokens)
        audit = None
        if self.memory is not None:
            q = h - self.key_mean
            if self.key_transform is not None:
                q = q @ self.key_transform
            g = torch.relu(self.memory.score(q)).amax(-1, keepdim=True)
            g = torch.where(g >= self.gate_thresh, g, torch.zeros_like(g))
            ret, audit = self.memory(q, return_audit=return_audit)
            if ret is not None:
                h = h + self.boost * g * ret
        return self.head(h), audit

    def attach_memory(self, memory: EditableHLM5Memory, key_mean: torch.Tensor,
                      boost: float, gate_thresh: float = 0.95,
                      key_transform: torch.Tensor | None = None) -> None:
        self.memory = memory
        self.key_mean = key_mean
        self.key_transform = key_transform
        self.boost = float(boost)
        self.gate_thresh = float(gate_thresh)

    def detach_memory(self) -> None:
        self.memory = None
        self.key_mean = None
        self.key_transform = None
        self.boost = 0.0


def lm_config(name: str) -> dict:
    configs = {
        "tiny": dict(dim=256, n_layers=4, n_heads=4, max_len=256),
        "small": dict(dim=384, n_layers=6, n_heads=6, max_len=512),
        # Phase-2 trunk: ~136M with 65K tied vocab (GPT-2-small shape) on FineWeb
        "base": dict(dim=768, n_layers=12, n_heads=12, max_len=1024),
        # Phase-3 trunk: ~1.04B with 65K tied vocab -- deliberately HLM3-Huge size
        # for a same-val-set comparison
        "huge1b": dict(dim=1792, n_layers=24, n_heads=14, max_len=1024),
        # Phase-4 planning trunk (~2.7–3.0B); pin exact dims after memory-fit pilot
        "huge3b": dict(dim=2560, n_layers=32, n_heads=32, max_len=1024),
    }
    return configs[name]

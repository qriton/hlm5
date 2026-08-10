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

from hlm5_memory import EditableHLM5Memory, FactorizedHopfieldMemoryLayer, RetrievalAudit


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
        # Optional internal residual-stream memory hook. hook_layer_index=N means
        # apply after N transformer blocks; 0 is immediately after token+pos embed.
        self.residual_memory: EditableHLM5Memory | None = None
        self.residual_key_mean: torch.Tensor | None = None
        self.residual_key_transform: torch.Tensor | None = None
        self.residual_boost: float = 0.0
        self.residual_gate_thresh: float = 0.95
        self.residual_hook_layer_index: int | None = None
        self.grad_checkpointing: bool = False

    def _run_encoder_layer(self, x: torch.Tensor, layer: nn.Module,
                           mask: torch.Tensor) -> torch.Tensor:
        return layer(x, src_mask=mask, is_causal=True)

    def _apply_memory(
        self,
        h: torch.Tensor,
        memory: EditableHLM5Memory,
        key_mean: torch.Tensor,
        boost: float,
        gate_thresh: float,
        key_transform: torch.Tensor | None = None,
        return_audit: bool = False,
    ) -> tuple[torch.Tensor, RetrievalAudit | None]:
        q = h - key_mean
        if key_transform is not None:
            q = q @ key_transform
        g = torch.relu(memory.score(q)).amax(-1, keepdim=True)
        g = torch.where(g >= gate_thresh, g, torch.zeros_like(g))
        ret, audit = memory(
            q,
            return_audit=return_audit,
            support_threshold=gate_thresh,
        )
        if ret is not None:
            h = h + boost * g * ret
        return h, audit

    def residual_state(
        self,
        tokens: torch.Tensor,
        hook_layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the residual stream after `hook_layer_index` transformer blocks."""
        n_layers = len(self.blocks.layers)
        if not 0 <= int(hook_layer_index) <= n_layers:
            raise ValueError(f"hook_layer_index must be in [0, {n_layers}]")
        T = tokens.shape[1]
        x = self.token_emb(tokens) + self.pos_emb[:, :T]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=tokens.device)
        for i in range(int(hook_layer_index)):
            x = self.blocks.layers[i](x, src_mask=mask, is_causal=True)
        return x, mask

    def continue_from_residual_state(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        hook_layer_index: int,
    ) -> torch.Tensor:
        """Continue a residual stream from `hook_layer_index` to final hidden."""
        n_layers = len(self.blocks.layers)
        if not 0 <= int(hook_layer_index) <= n_layers:
            raise ValueError(f"hook_layer_index must be in [0, {n_layers}]")
        for i in range(int(hook_layer_index), n_layers):
            x = self.blocks.layers[i](x, src_mask=mask, is_causal=True)
        if self.trained_memory is not None:
            x, _ = self.trained_memory(x)
        return self.norm(x)

    def _hidden_with_audit(
        self,
        tokens: torch.Tensor,
        return_audit: bool = False,
    ) -> tuple[torch.Tensor, RetrievalAudit | None]:
        T = tokens.shape[1]
        x = self.token_emb(tokens) + self.pos_emb[:, :T]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=tokens.device)
        residual_audit = None
        if self.residual_memory is None:
            if self.grad_checkpointing and self.training:
                for layer in self.blocks.layers:
                    x = checkpoint(self._run_encoder_layer, x, layer, mask,
                                   use_reentrant=False)
            else:
                x = self.blocks(x, mask=mask, is_causal=True)
        else:
            if self.residual_hook_layer_index is None:
                raise RuntimeError("residual memory is attached without a hook layer index")
            hook_layer_index = int(self.residual_hook_layer_index)
            n_layers = len(self.blocks.layers)
            if not 0 <= hook_layer_index <= n_layers:
                raise ValueError(f"residual hook layer index must be in [0, {n_layers}]")
            if hook_layer_index == 0:
                x, residual_audit = self._apply_memory(
                    x,
                    self.residual_memory,
                    self.residual_key_mean,
                    self.residual_boost,
                    self.residual_gate_thresh,
                    self.residual_key_transform,
                    return_audit=return_audit,
                )
            for layer in self.blocks.layers:
                x = layer(x, src_mask=mask, is_causal=True)
                hook_layer_index -= 1
                if hook_layer_index == 0:
                    x, residual_audit = self._apply_memory(
                        x,
                        self.residual_memory,
                        self.residual_key_mean,
                        self.residual_boost,
                        self.residual_gate_thresh,
                        self.residual_key_transform,
                        return_audit=return_audit,
                    )
        if self.trained_memory is not None:
            x, _ = self.trained_memory(x)
        return self.norm(x), residual_audit

    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        h, _ = self._hidden_with_audit(tokens, return_audit=False)
        return h

    def forward(self, tokens: torch.Tensor, return_audit: bool = False):
        h, audit = self._hidden_with_audit(tokens, return_audit=return_audit)
        if self.memory is not None:
            h, pre_head_audit = self._apply_memory(
                h,
                self.memory,
                self.key_mean,
                self.boost,
                self.gate_thresh,
                self.key_transform,
                return_audit=return_audit,
            )
            audit = pre_head_audit
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
        self.detach_residual_memory()

    def attach_residual_memory(self, memory: EditableHLM5Memory, key_mean: torch.Tensor,
                               boost: float, hook_layer_index: int,
                               gate_thresh: float = 0.95,
                               key_transform: torch.Tensor | None = None) -> None:
        n_layers = len(self.blocks.layers)
        if not 0 <= int(hook_layer_index) <= n_layers:
            raise ValueError(f"hook_layer_index must be in [0, {n_layers}]")
        self.residual_memory = memory
        self.residual_key_mean = key_mean
        self.residual_key_transform = key_transform
        self.residual_boost = float(boost)
        self.residual_gate_thresh = float(gate_thresh)
        self.residual_hook_layer_index = int(hook_layer_index)

    def detach_residual_memory(self) -> None:
        self.residual_memory = None
        self.residual_key_mean = None
        self.residual_key_transform = None
        self.residual_boost = 0.0
        self.residual_hook_layer_index = None


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

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RetrievalAudit:
    top_slots: torch.Tensor
    top_scores: torch.Tensor
    top_labels: list[list[str | None]]


def unit(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


class EditableHLM5Memory(nn.Module):
    """Editable positive-kernel memory.

    Default degree is 5:

        alpha_r * max(0, cos(q, m_r)) ** 5

    Keys and values are Parameters so this module can be trained by backprop,
    while edit operations mutate slots under torch.no_grad().
    """

    def __init__(
        self,
        dim: int,
        memory_size: int = 256,
        degree: int = 5,
        temperature: float = 0.10,
        learnable_memory: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if degree < 2:
            raise ValueError("degree must be >= 2")

        self.dim = dim
        self.memory_size = memory_size
        self.degree = degree
        self.temperature = temperature
        self.eps = eps

        keys = unit(torch.randn(memory_size, dim) * 0.02)
        values = unit(torch.randn(memory_size, dim) * 0.02)
        alphas = torch.ones(memory_size)

        if learnable_memory:
            self.keys = nn.Parameter(keys)
            self.values = nn.Parameter(values)
            self.alphas = nn.Parameter(alphas)
        else:
            self.register_buffer("keys", keys)
            self.register_buffer("values", values)
            self.register_buffer("alphas", alphas)

        self.register_buffer("active", torch.zeros(memory_size, dtype=torch.bool))
        self.labels: list[str | None] = [None for _ in range(memory_size)]

    @torch.no_grad()
    def inject(
        self,
        pattern: torch.Tensor,
        value: torch.Tensor | None = None,
        strength: float = 1.0,
        label: str | None = None,
    ) -> int:
        free = torch.nonzero(~self.active, as_tuple=False).flatten()
        if free.numel() == 0:
            raise RuntimeError("memory is full")

        slot = int(free[0].item())
        key = unit(pattern.detach().to(self.keys.device).float(), dim=0)
        payload = key if value is None else unit(
            value.detach().to(self.values.device).float(),
            dim=0,
        )

        self.keys.data[slot].copy_(key)
        self.values.data[slot].copy_(payload)
        self.alphas.data[slot] = float(strength)
        self.active[slot] = True
        self.labels[slot] = label
        return slot

    @torch.no_grad()
    def weaken(self, slot: int, factor: float) -> None:
        self._check_slot(slot)
        if factor < 0:
            raise ValueError("factor must be non-negative")
        self.alphas.data[slot] *= float(factor)

    @torch.no_grad()
    def remove(self, slot: int) -> None:
        self._check_slot(slot)
        self.active[slot] = False
        self.alphas.data[slot] = 0.0
        self.labels[slot] = None

    @torch.no_grad()
    def transplant_to(
        self,
        other: "EditableHLM5Memory",
        slots: list[int],
        label_prefix: str = "copied",
    ) -> list[int]:
        if self.dim != other.dim or self.degree != other.degree:
            raise ValueError("memories must have matching dim and degree")

        new_slots: list[int] = []
        for slot in slots:
            self._check_slot(slot)
            if not bool(self.active[slot]):
                continue

            label = self.labels[slot]
            copied_label = f"{label_prefix}:{label}" if label else label_prefix
            new_slots.append(
                other.inject(
                    self.keys[slot],
                    value=self.values[slot],
                    strength=float(self.alphas[slot].item()),
                    label=copied_label,
                )
            )
        return new_slots

    def score(self, query: torch.Tensor) -> torch.Tensor:
        """Return masked positive-kernel scores for query shape [..., dim]."""
        if query.shape[-1] != self.dim:
            raise ValueError(f"query must end with dim={self.dim}")

        if not bool(self.active.any()):
            return torch.full(
                (*query.shape[:-1], self.memory_size),
                float("-inf"),
                device=query.device,
                dtype=query.dtype,
            )

        q = unit(query, dim=-1, eps=self.eps)
        k = unit(self.keys, dim=-1, eps=self.eps)
        cosine = torch.einsum("...d,md->...m", q, k)
        positive = torch.relu(cosine)
        scores = positive.pow(self.degree) * torch.relu(self.alphas)
        return scores.masked_fill(~self.active.view(*([1] * (query.ndim - 1)), -1), float("-inf"))

    def forward(
        self,
        query: torch.Tensor,
        return_audit: bool = False,
        top_k: int = 3,
    ) -> tuple[torch.Tensor, RetrievalAudit | None]:
        """Soft retrieval for query shape [batch, seq, dim]."""
        if query.ndim != 3:
            raise ValueError("query must have shape [batch, seq, dim]")

        scores = self.score(query)
        if torch.isneginf(scores).all():
            return torch.zeros_like(query), None

        attn = F.softmax(scores / self.temperature, dim=-1)
        retrieved = attn @ self.values
        audit = self.audit(scores, top_k=top_k) if return_audit else None
        return retrieved, audit

    def top_slot(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(query)
        values, indices = torch.max(scores, dim=-1)
        return indices, values

    def audit(self, scores: torch.Tensor, top_k: int = 3) -> RetrievalAudit:
        active_count = int(self.active.sum().item())
        k = max(1, min(top_k, active_count))
        values, indices = torch.topk(scores.detach().cpu(), k=k, dim=-1)
        flat_indices = indices.reshape(-1, k)
        labels: list[list[str | None]] = []
        for row in flat_indices:
            labels.append([self.labels[int(slot.item())] for slot in row])
        return RetrievalAudit(top_slots=indices, top_scores=values, top_labels=labels)

    @torch.no_grad()
    def coherence(self) -> float:
        active = self.active.bool()
        if int(active.sum().item()) < 2:
            return 0.0
        keys = unit(self.keys[active], dim=-1, eps=self.eps)
        overlaps = torch.abs(keys @ keys.T)
        overlaps.fill_diagonal_(0.0)
        return float(overlaps.max().item())

    @torch.no_grad()
    def max_crosstalk(self) -> float:
        rho = self.coherence()
        return float(rho**self.degree)

    @torch.no_grad()
    def edit_leakage(
        self,
        slot: int,
        edited_key: torch.Tensor,
        probes: torch.Tensor,
    ) -> float:
        self._check_slot(slot)
        before, _ = self(probes[:, None, :], return_audit=False)
        before = before[:, 0, :]

        original = self.keys[slot].detach().clone()
        self.keys.data[slot].copy_(unit(edited_key.to(self.keys.device).float(), dim=0))
        try:
            after, _ = self(probes[:, None, :], return_audit=False)
            after = after[:, 0, :]
        finally:
            self.keys.data[slot].copy_(original)

        mask = torch.ones(probes.shape[0], dtype=torch.bool, device=probes.device)
        if slot < probes.shape[0]:
            mask[slot] = False
        if not bool(mask.any()):
            return 0.0
        return float(torch.norm(before[mask] - after[mask], dim=-1).max().item())

    def _check_slot(self, slot: int) -> None:
        if not 0 <= slot < self.memory_size:
            raise IndexError(f"invalid memory slot {slot}")


class FactorizedHopfieldMemoryLayer(nn.Module):
    """Residual HLM5 layer for transformer hidden states."""

    def __init__(
        self,
        dim: int,
        memory_size: int = 256,
        degree: int = 5,
        temperature: float = 0.10,
    ) -> None:
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.memory = EditableHLM5Memory(
            dim=dim,
            memory_size=memory_size,
            degree=degree,
            temperature=temperature,
        )
        self.out_proj = nn.Linear(dim, dim)
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(
        self,
        x: torch.Tensor,
        return_audit: bool = False,
    ) -> tuple[torch.Tensor, RetrievalAudit | None]:
        q = self.query_proj(x)
        retrieved, audit = self.memory(q, return_audit=return_audit)
        gate = torch.sigmoid(self.gate_logit)
        return x + gate * self.out_proj(retrieved), audit


class TinyTransformerWithHLM5(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        heads: int = 4,
        memory_size: int = 64,
        degree: int = 5,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, dim) * 0.01)
        self.block = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            batch_first=True,
            activation="gelu",
        )
        self.memory_layer = FactorizedHopfieldMemoryLayer(
            dim=dim,
            memory_size=memory_size,
            degree=degree,
        )
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        return_audit: bool = False,
    ) -> tuple[torch.Tensor, RetrievalAudit | None]:
        seq_len = tokens.shape[1]
        if seq_len > self.pos_emb.shape[1]:
            raise ValueError("sequence length exceeds max_len")
        x = self.token_emb(tokens) + self.pos_emb[:, :seq_len]
        x = self.block(x)
        x, audit = self.memory_layer(x, return_audit=return_audit)
        return self.lm_head(self.norm(x)), audit

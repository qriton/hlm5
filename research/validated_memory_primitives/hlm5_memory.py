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


@dataclass(frozen=True)
class DeactivatedSlot:
    """In-process token required to reactivate a reversibly disabled slot."""

    slot: int
    strength: float
    label: str | None
    version: int


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
        read_mode: str = "dense",
        retrieval_k: int = 8,
        support_threshold: float | None = None,
    ) -> None:
        super().__init__()
        if degree < 1:
            raise ValueError("degree must be >= 1")
        if read_mode not in {"dense", "support_masked", "top_k", "hard_top1"}:
            raise ValueError(
                "read_mode must be dense, support_masked, top_k, or hard_top1"
            )
        if retrieval_k < 1:
            raise ValueError("retrieval_k must be >= 1")
        if support_threshold is not None and support_threshold < 0:
            raise ValueError("support_threshold must be non-negative")

        self.dim = dim
        self.memory_size = memory_size
        self.degree = degree
        self.temperature = temperature
        self.eps = eps
        self.read_mode = read_mode
        self.retrieval_k = int(retrieval_k)
        self.support_threshold = support_threshold

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
        # A deactivated fact is inactive but not free. This buffer is deliberately
        # non-persistent so legacy checkpoints retain the exact same state-dict
        # surface. Deactivation snapshots are in-process lifecycle tokens; callers
        # that need durable rollback must persist the snapshot in their own store.
        self.register_buffer(
            "_reserved",
            torch.zeros(memory_size, dtype=torch.bool),
            persistent=False,
        )
        self.labels: list[str | None] = [None for _ in range(memory_size)]
        self._slot_versions: list[int] = [0 for _ in range(memory_size)]

    @torch.no_grad()
    def inject(
        self,
        pattern: torch.Tensor,
        value: torch.Tensor | None = None,
        strength: float = 1.0,
        label: str | None = None,
        value_scale: float = 1.0,
    ) -> int:
        if value_scale < 0:
            raise ValueError("value_scale must be non-negative")
        occupied = self.active | self._reserved
        free = torch.nonzero(~occupied, as_tuple=False).flatten()
        if free.numel() == 0:
            raise RuntimeError("memory is full")

        slot = int(free[0].item())
        key = unit(pattern.detach().to(self.keys.device).float(), dim=0)
        payload = key if value is None else unit(
            value.detach().to(self.values.device).float(),
            dim=0,
        )
        payload = payload * float(value_scale)

        self.keys.data[slot].copy_(key)
        self.values.data[slot].copy_(payload)
        self.alphas.data[slot] = float(strength)
        self.active[slot] = True
        self._reserved[slot] = False
        self.labels[slot] = label
        self._slot_versions[slot] += 1
        return slot

    @torch.no_grad()
    def weaken(self, slot: int, factor: float) -> None:
        self._check_slot(slot)
        if factor < 0:
            raise ValueError("factor must be non-negative")
        self.alphas.data[slot] *= float(factor)

    @torch.no_grad()
    def remove(self, slot: int) -> None:
        """Legacy mask-and-free operation; key/value tensors are not overwritten."""
        self._check_slot(slot)
        self.active[slot] = False
        self._reserved[slot] = False
        self.alphas.data[slot] = 0.0
        self.labels[slot] = None
        self._slot_versions[slot] += 1

    @torch.no_grad()
    def edit_value(
        self,
        slot: int,
        value: torch.Tensor,
        *,
        value_scale: float = 1.0,
        strength: float | None = None,
        label: str | None = None,
    ) -> None:
        """Replace an active slot payload with explicit direction and dose."""
        self._check_slot(slot)
        if not bool(self.active[slot]):
            raise RuntimeError("cannot edit an inactive slot")
        if value_scale < 0:
            raise ValueError("value_scale must be non-negative")
        payload = unit(value.detach().to(self.values.device).float(), dim=0)
        self.values.data[slot].copy_(payload * float(value_scale))
        if strength is not None:
            self.alphas.data[slot] = float(strength)
        if label is not None:
            self.labels[slot] = label
        self._slot_versions[slot] += 1

    @torch.no_grad()
    def deactivate(self, slot: int) -> DeactivatedSlot:
        """Reversibly disable a slot while reserving it against reinjection."""
        self._check_slot(slot)
        if not bool(self.active[slot]):
            raise RuntimeError("cannot deactivate an inactive slot")
        self._slot_versions[slot] += 1
        snapshot = DeactivatedSlot(
            slot=slot,
            strength=float(self.alphas[slot].item()),
            label=self.labels[slot],
            version=self._slot_versions[slot],
        )
        self.active[slot] = False
        self._reserved[slot] = True
        self.alphas.data[slot] = 0.0
        self.labels[slot] = None
        return snapshot

    @torch.no_grad()
    def reactivate(self, snapshot: DeactivatedSlot) -> None:
        """Restore a slot from the matching in-process deactivation token."""
        self._check_slot(snapshot.slot)
        slot = snapshot.slot
        if self._slot_versions[slot] != snapshot.version:
            raise RuntimeError("deactivation snapshot is stale")
        if bool(self.active[slot]) or not bool(self._reserved[slot]):
            raise RuntimeError("slot is not reserved for reactivation")
        self.alphas.data[slot] = float(snapshot.strength)
        self.labels[slot] = snapshot.label
        self._reserved[slot] = False
        self.active[slot] = True
        self._slot_versions[slot] += 1

    @torch.no_grad()
    def hard_delete(self, slot: int, *, rerandomize: bool = True) -> None:
        """Overwrite and free a slot; this is distinct from reversible rollback."""
        self._check_slot(slot)
        if not bool(self.active[slot] | self._reserved[slot]):
            raise RuntimeError("cannot hard-delete a free slot")
        if rerandomize:
            self.keys.data[slot].normal_(mean=0.0, std=0.02)
            self.values.data[slot].normal_(mean=0.0, std=0.02)
            self.keys.data[slot].copy_(unit(self.keys.data[slot], dim=0, eps=self.eps))
            self.values.data[slot].copy_(unit(self.values.data[slot], dim=0, eps=self.eps))
        else:
            self.keys.data[slot].zero_()
            self.values.data[slot].zero_()
        self.active[slot] = False
        self._reserved[slot] = False
        self.alphas.data[slot] = 0.0
        self.labels[slot] = None
        self._slot_versions[slot] += 1

    def is_occupied(self, slot: int) -> bool:
        """Return whether a slot is active or reserved for reactivation."""
        self._check_slot(slot)
        return bool(self.active[slot] | self._reserved[slot])

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
            value_scale = float(self.values[slot].detach().norm().item())
            new_slots.append(
                other.inject(
                    self.keys[slot],
                    value=self.values[slot],
                    strength=float(self.alphas[slot].item()),
                    label=copied_label,
                    value_scale=value_scale,
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

    def _selected_scores(
        self,
        scores: torch.Tensor,
        support_threshold: float | None,
    ) -> torch.Tensor:
        """Apply read support without changing the raw score/audit contract."""
        if self.read_mode == "dense":
            return scores
        if self.read_mode == "support_masked":
            threshold = (
                self.support_threshold
                if support_threshold is None
                else support_threshold
            )
            if threshold is None:
                raise ValueError(
                    "support_masked retrieval requires a support_threshold"
                )
            return scores.masked_fill(scores < float(threshold), float("-inf"))

        active_count = int(self.active.sum().item())
        k = 1 if self.read_mode == "hard_top1" else min(
            self.retrieval_k,
            active_count,
        )
        values, indices = torch.topk(scores, k=k, dim=-1)
        selected = torch.full_like(scores, float("-inf"))
        return selected.scatter(-1, indices, values)

    def forward(
        self,
        query: torch.Tensor,
        return_audit: bool = False,
        top_k: int = 3,
        support_threshold: float | None = None,
    ) -> tuple[torch.Tensor, RetrievalAudit | None]:
        """Retrieve for query shape [batch, seq, dim] under the configured read."""
        if query.ndim != 3:
            raise ValueError("query must have shape [batch, seq, dim]")

        scores = self.score(query)
        if torch.isneginf(scores).all():
            return torch.zeros_like(query), None

        selected_scores = self._selected_scores(scores, support_threshold)
        has_support = torch.isfinite(selected_scores).any(dim=-1, keepdim=True)
        safe_scores = torch.where(
            has_support,
            selected_scores,
            torch.zeros_like(selected_scores),
        )
        attn = F.softmax(safe_scores / self.temperature, dim=-1)
        attn = torch.where(has_support, attn, torch.zeros_like(attn))
        retrieved = attn @ self.values
        # Audit the raw score ranking for compatibility with existing receipts
        # and callers. The experiment report separately binds the read policy.
        audit = self.audit(scores, top_k=top_k) if return_audit else None
        return retrieved, audit

    def retrieval_config(self) -> dict[str, int | float | str | None]:
        """Serializable operator settings for experiment/replay artifacts."""
        return {
            "read_mode": self.read_mode,
            "retrieval_k": self.retrieval_k,
            "support_threshold": self.support_threshold,
            "degree": self.degree,
            "temperature": self.temperature,
        }

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

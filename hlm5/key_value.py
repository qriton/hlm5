from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import EditableHLM5Memory, RetrievalAudit, unit


@dataclass
class PairEdit:
    slot: int
    key_id: int
    value_id: int
    label: str


class HLM5KeyValueModel(nn.Module):
    """First HLM5 model: editable key-value associative memory.

    A key id is embedded into representation space, used as an HLM5 memory
    query, and decoded into a value id. Direct edits mutate memory slots:

        inject_pair(key_id, value_id)
        edit_value(slot, new_value_id)
        weaken(slot, factor)
        remove(slot)

    This is the smallest useful model for testing editable capacity. If this
    fails, a transformer-integrated version will only hide the problem.
    """

    def __init__(
        self,
        key_count: int,
        value_count: int,
        dim: int = 64,
        memory_size: int = 128,
        degree: int = 5,
        temperature: float = 0.06,
    ) -> None:
        super().__init__()
        self.key_count = key_count
        self.value_count = value_count
        self.dim = dim

        self.key_emb = nn.Embedding(key_count, dim)
        self.value_emb = nn.Embedding(value_count, dim)
        self.query_proj = nn.Linear(dim, dim)
        self.memory = EditableHLM5Memory(
            dim=dim,
            memory_size=memory_size,
            degree=degree,
            temperature=temperature,
            learnable_memory=True,
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.5))
        self.slot_pairs: dict[int, PairEdit] = {}

        self._init_embeddings()

    def forward(
        self,
        key_ids: torch.Tensor,
        return_audit: bool = False,
    ) -> tuple[torch.Tensor, RetrievalAudit | None]:
        query = self.encode_key(key_ids)
        retrieved, audit = self.memory(query[:, None, :], return_audit=return_audit)
        logits = self.decode_value(retrieved[:, 0, :])
        return logits, audit

    def encode_key(self, key_ids: torch.Tensor) -> torch.Tensor:
        return unit(self.query_proj(self.key_emb(key_ids)), dim=-1)

    def encode_value(self, value_ids: torch.Tensor) -> torch.Tensor:
        return unit(self.value_emb(value_ids), dim=-1)

    def decode_value(self, hidden: torch.Tensor) -> torch.Tensor:
        value_basis = unit(self.value_emb.weight, dim=-1)
        return self.logit_scale.exp().clamp(max=100.0) * (unit(hidden, dim=-1) @ value_basis.T)

    @torch.no_grad()
    def inject_pair(
        self,
        key_id: int,
        value_id: int,
        strength: float = 1.0,
        label: str | None = None,
    ) -> int:
        self._check_key_value(key_id, value_id)
        device = self.key_emb.weight.device
        key = self.encode_key(torch.tensor([key_id], device=device))[0]
        value = self.encode_value(torch.tensor([value_id], device=device))[0]
        pair_label = label or f"k{key_id}->v{value_id}"
        slot = self.memory.inject(key, value=value, strength=strength, label=pair_label)
        self.slot_pairs[slot] = PairEdit(slot, key_id, value_id, pair_label)
        return slot

    @torch.no_grad()
    def edit_value(self, slot: int, new_value_id: int, strength: float | None = None) -> None:
        self._check_slot(slot)
        if not 0 <= new_value_id < self.value_count:
            raise IndexError(f"invalid value id {new_value_id}")

        device = self.value_emb.weight.device
        value = self.encode_value(torch.tensor([new_value_id], device=device))[0]
        self.memory.values.data[slot].copy_(value)
        if strength is not None:
            self.memory.alphas.data[slot] = float(strength)

        old = self.slot_pairs[slot]
        label = f"k{old.key_id}->v{new_value_id}"
        self.slot_pairs[slot] = PairEdit(slot, old.key_id, new_value_id, label)
        self.memory.labels[slot] = label

    @torch.no_grad()
    def weaken(self, slot: int, factor: float) -> None:
        self._check_slot(slot)
        self.memory.weaken(slot, factor)

    @torch.no_grad()
    def remove(self, slot: int) -> None:
        self._check_slot(slot)
        self.memory.remove(slot)
        self.slot_pairs.pop(slot, None)

    @torch.no_grad()
    def predict(self, key_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self(key_ids, return_audit=False)
        return torch.argmax(logits, dim=-1)

    @torch.no_grad()
    def accuracy(self, pairs: list[tuple[int, int]]) -> float:
        if not pairs:
            return 0.0
        device = self.key_emb.weight.device
        keys = torch.tensor([key for key, _ in pairs], device=device)
        expected = torch.tensor([value for _, value in pairs], device=device)
        predicted = self.predict(keys)
        return float((predicted == expected).float().mean().item())

    def active_pairs(self) -> list[tuple[int, int]]:
        return [
            (edit.key_id, edit.value_id)
            for slot, edit in sorted(self.slot_pairs.items())
            if bool(self.memory.active[slot])
        ]

    def _check_key_value(self, key_id: int, value_id: int) -> None:
        if not 0 <= key_id < self.key_count:
            raise IndexError(f"invalid key id {key_id}")
        if not 0 <= value_id < self.value_count:
            raise IndexError(f"invalid value id {value_id}")

    def _check_slot(self, slot: int) -> None:
        if slot not in self.slot_pairs:
            raise IndexError(f"slot {slot} does not contain an injected pair")

    def _init_embeddings(self) -> None:
        nn.init.normal_(self.key_emb.weight, mean=0.0, std=0.25)
        nn.init.normal_(self.value_emb.weight, mean=0.0, std=0.25)
        nn.init.eye_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)

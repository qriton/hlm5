"""Architecture-agnostic HLM5 adapter for frozen pre-head hidden states.

The adapter implements the same inference-time operator as ``HLM5LM`` without
owning or mutating a language-model trunk.  A caller supplies the final hidden
state and applies the frozen model's ordinary output head afterward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import EditableHLM5Memory


@dataclass(frozen=True)
class PreHeadAdapterAudit:
    """Runtime evidence for one HLM5 pre-head read."""

    query: torch.Tensor
    scores: torch.Tensor
    attention: torch.Tensor
    selected_slots: torch.Tensor
    selected_scores: torch.Tensor
    gate_open: torch.Tensor
    delta: torch.Tensor


class HLM5PreHeadAdapter(nn.Module):
    """Apply an ``EditableHLM5Memory`` to arbitrary final hidden states.

    Gate and retrieval arithmetic follow the released HLM5 hook in float32 by
    default.  Only the final delta is cast to the hidden state's native dtype.
    Gated-off rows add an exact zero of that native dtype.
    """

    def __init__(
        self,
        memory: EditableHLM5Memory,
        key_mean: torch.Tensor,
        key_transform: torch.Tensor,
        *,
        boost: float = 1.0,
        gate_thresh: float = 0.95,
    ) -> None:
        super().__init__()
        if key_mean.shape != (memory.dim,):
            raise ValueError(f"key_mean must have shape ({memory.dim},)")
        if key_transform.shape != (memory.dim, memory.dim):
            raise ValueError(
                f"key_transform must have shape ({memory.dim}, {memory.dim})"
            )
        if boost < 0:
            raise ValueError("boost must be non-negative")
        if not 0.0 <= gate_thresh <= 1.0:
            raise ValueError("gate_thresh must lie in [0, 1]")

        self.memory = memory
        memory_dtype = memory.keys.dtype
        memory_device = memory.keys.device
        self.register_buffer(
            "key_mean",
            key_mean.detach().to(device=memory_device, dtype=memory_dtype),
        )
        self.register_buffer(
            "key_transform",
            key_transform.detach().to(device=memory_device, dtype=memory_dtype),
        )
        self.boost = float(boost)
        self.gate_thresh = float(gate_thresh)

    def transform_query(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] != self.memory.dim:
            raise ValueError(f"hidden must end with dim={self.memory.dim}")
        query = hidden.to(
            device=self.memory.keys.device,
            dtype=self.memory.keys.dtype,
        )
        return (query - self.key_mean) @ self.key_transform

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        return_audit: bool = False,
    ) -> tuple[torch.Tensor, PreHeadAdapterAudit | None]:
        query = self.transform_query(hidden)
        if not bool(self.memory.active.any()):
            empty_shape = (*query.shape[:-1], self.memory.memory_size)
            scores = torch.full(
                empty_shape,
                float("-inf"),
                device=query.device,
                dtype=query.dtype,
            )
            attention = torch.zeros_like(scores)
            selected_slots = torch.full(
                query.shape[:-1],
                -1,
                device=query.device,
                dtype=torch.long,
            )
            selected_scores = torch.zeros(
                query.shape[:-1],
                device=query.device,
                dtype=query.dtype,
            )
            gate_open = torch.zeros(
                query.shape[:-1],
                device=query.device,
                dtype=torch.bool,
            )
            delta_memory = torch.zeros_like(query)
        else:
            scores = self.memory.score(query)
            selected_scores, selected_slots = scores.max(dim=-1)
            selected_scores = torch.relu(selected_scores)
            gate_open = torch.isfinite(selected_scores) & (
                selected_scores >= self.gate_thresh
            )
            gate_gain = torch.where(
                gate_open,
                selected_scores,
                torch.zeros_like(selected_scores),
            )
            attention = F.softmax(scores / self.memory.temperature, dim=-1)
            retrieved = attention @ self.memory.values
            delta_memory = self.boost * gate_gain.unsqueeze(-1) * retrieved

        delta = delta_memory.to(device=hidden.device, dtype=hidden.dtype)
        adapted = hidden + delta
        if not return_audit:
            return adapted, None
        audit = PreHeadAdapterAudit(
            query=query.detach(),
            scores=scores.detach(),
            attention=attention.detach(),
            selected_slots=selected_slots.detach(),
            selected_scores=selected_scores.detach(),
            gate_open=gate_open.detach(),
            delta=delta.detach(),
        )
        return adapted, audit

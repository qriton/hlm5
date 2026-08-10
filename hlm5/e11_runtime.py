"""Cross-node reconstruction runtime for registered HLM5 E11."""

from __future__ import annotations

import platform
from typing import Any

import torch

from .e7_contract import tensor_sha256
from .e7b_runtime import environment_record as base_environment_record
from .e10_runtime import (
    BOOST,
    DEGREE,
    GATE_THRESHOLD,
    MEMORY_SIZE,
    TEMPERATURE,
    load_bundle,
    rollback_scan,
    scan_locality,
    scan_positives,
)
from .memory import EditableHLM5Memory
from .public_adapter import HLM5PreHeadAdapter


E10_NODE = "lrdn1455.leonardo.local"


def e11_environment_record(device: torch.device) -> dict[str, Any]:
    value = base_environment_record(device)
    value["node"] = platform.node()
    return value


@torch.inference_mode()
def adapter_from_bundle(
    bundle: dict[str, torch.Tensor],
    labels: list[str | None],
    selected_indices: list[int],
    device: torch.device,
) -> tuple[EditableHLM5Memory, HLM5PreHeadAdapter, dict[int, int], dict[str, Any]]:
    """Restore the exact E10 active prefix without reconstructing any tensor."""
    keys = bundle["keys"]
    values = bundle["values"]
    alphas = bundle["alphas"]
    key_mean = bundle["key_mean"]
    key_transform = bundle["key_transform"]
    count, dim = keys.shape
    if count != 1024 or dim != 2048 or len(labels) != count:
        raise ValueError("E11 bundle capacity contract mismatch")
    if bundle["selected_indices"].tolist() != selected_indices:
        raise RuntimeError("E11 bundle selected indices differ from E10 JSON")
    memory = EditableHLM5Memory(
        dim=dim,
        memory_size=MEMORY_SIZE,
        degree=DEGREE,
        temperature=TEMPERATURE,
        learnable_memory=False,
    ).to(device)
    memory.keys[:count].copy_(keys.to(device))
    memory.values[:count].copy_(values.to(device))
    memory.alphas[:count].copy_(alphas.to(device))
    memory.active[:count] = True
    for slot, label in enumerate(labels):
        memory.labels[slot] = label
    slot_by_candidate = {
        candidate_index: slot for slot, candidate_index in enumerate(selected_indices)
    }
    key_mean_gpu = key_mean.to(device)
    key_transform_gpu = key_transform.to(device)
    adapter = HLM5PreHeadAdapter(
        memory,
        key_mean_gpu,
        key_transform_gpu,
        boost=BOOST,
        gate_thresh=GATE_THRESHOLD,
    )
    active = torch.nonzero(memory.active, as_tuple=False).flatten()
    receipt = {
        "memory_size": MEMORY_SIZE,
        "degree": DEGREE,
        "temperature": TEMPERATURE,
        "boost": BOOST,
        "gate_threshold": GATE_THRESHOLD,
        "active_count": int(active.numel()),
        "active_slots": [int(value) for value in active],
        "selected_candidate_indices": list(selected_indices),
        "slot_by_candidate": {str(k): v for k, v in slot_by_candidate.items()},
        "active_mask_sha256": tensor_sha256(memory.active),
        "active_keys_sha256": tensor_sha256(memory.keys[active].contiguous()),
        "active_values_sha256": tensor_sha256(memory.values[active].contiguous()),
        "active_alphas_sha256": tensor_sha256(memory.alphas[active].contiguous()),
        "active_labels": [memory.labels[int(slot)] for slot in active],
        "key_mean_sha256": tensor_sha256(key_mean_gpu),
        "key_transform_sha256": tensor_sha256(key_transform_gpu),
    }
    return memory, adapter, slot_by_candidate, receipt


@torch.inference_mode()
def cross_node_measurement(
    model: Any,
    cases: list[dict[str, Any]],
    exact_hidden: torch.Tensor,
    query_rows: list[dict[str, Any]],
    locality_hidden: torch.Tensor,
    bundle: dict[str, torch.Tensor],
    bundle_manifest: dict[str, Any],
    selected_indices: list[int],
    device: torch.device,
) -> dict[str, Any]:
    labels = list(bundle_manifest["active_labels"])
    memory, adapter, slot_by_candidate, receipt = adapter_from_bundle(
        bundle, labels, selected_indices, device
    )
    positives = scan_positives(
        model,
        adapter,
        memory,
        cases,
        exact_hidden,
        selected_indices,
        slot_by_candidate,
    )
    locality = scan_locality(
        adapter, memory, selected_indices, cases, query_rows, locality_hidden
    )
    rollback = rollback_scan(
        adapter, memory, exact_hidden[selected_indices], locality_hidden
    )
    return {
        "memory_receipt": receipt,
        "positive_scan": positives,
        "locality_scan": locality,
        "rollback": rollback,
    }


__all__ = [
    "E10_NODE",
    "adapter_from_bundle",
    "cross_node_measurement",
    "e11_environment_record",
    "load_bundle",
]

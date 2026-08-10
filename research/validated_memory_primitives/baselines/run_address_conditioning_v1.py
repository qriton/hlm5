"""Registered HLM5 E5-v3 address-conditioning experiment.

Companion protocol:
  D:/HLM2/docs/hlm5-e5v3-address-conditioning-prereg-2026-08-10.md

This script never trains model weights or launches remote compute. The default
command runs the registered local ladder and obeys per-track stop gates. Use
``--smoke`` only for the diagnostic K=16 determinism preflight.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import math
import platform
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer as HFTokenizer


ROOT = Path(__file__).resolve().parents[1]
HLM2_ROOT = Path("D:/HLM2")
sys.path.insert(0, str(ROOT))

from hlm5_memory import DeactivatedSlot, EditableHLM5Memory, unit  # noqa: E402
from run_seq_edit_stream_v2 import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_TOKENIZER,
    NEUTRAL_PROMPTS,
    assign_distinct_targets,
    base_predictions,
    extract_last_hiddens,
    find_reachable_target_pool,
    fit_fixed_key_transform,
    load_model,
    make_entity_names,
    make_replacement_targets,
    parse_int_csv,
    sha256_file,
    synchronize,
    transformed_keys,
)


REGISTERED_K = [64, 256, 1024, 4096]
REGISTERED_SEEDS = [1, 2, 3, 4, 5]
REGISTERED_TAU_COS = 0.95
REGISTERED_DEGREE = 5
REGISTERED_TEMPERATURE = 0.10
REGISTERED_CALIBRATION_COUNT = 256
REGISTERED_FLOOR_FRACTION = 0.01
REGISTERED_DOSE_MARGIN = 1.05
REGISTERED_DOSE_EPSILON = 1e-3
CANONICAL_DOMAIN = "hlm5-e5v3-address-v1"
OFF_SUPPORT_DOMAIN = "hlm5-e5v3-off-v1"

DEFAULT_REGISTRATION = (
    HLM2_ROOT
    / "evidence"
    / "registrations"
    / "hlm5-e5v3-address-conditioning-registration-2026-08-10.json"
)
DEFAULT_OUTPUT = ROOT / "baselines" / "results" / "address_conditioning_v1.json"
SMOKE_OUTPUT = (
    ROOT / "baselines" / "results" / "address_conditioning_v1_smoke.json"
)


@dataclass(frozen=True)
class RegistryPlan:
    requested_global: list[int]
    accepted_global: list[int]
    rejected_global: list[int]
    accepted_flags: list[bool]
    max_against_accepted: list[float | None]
    accepted_pairwise_max: float
    rejected_max_cosine: float | None
    check_ms_per_requested: float


class HarnessInvalidError(RuntimeError):
    """Raised when an implementation contract invalidates the package."""


def require_harness_invariant(condition: bool, message: str) -> None:
    """Stop before later lifecycle operations when an invariant is false."""

    if not condition:
        raise HarnessInvalidError(message)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def describe_tensor(tensor: torch.Tensor) -> dict[str, float | None]:
    values = tensor.detach().float().flatten()
    if values.numel() == 0:
        return {"min": None, "median": None, "p90": None, "max": None}
    return {
        "min": float(values.min().item()),
        "median": float(values.median().item()),
        "p90": float(torch.quantile(values, 0.90).item()),
        "max": float(values.max().item()),
    }


def fraction(mask: torch.Tensor) -> float | None:
    if mask.numel() == 0:
        return None
    return float(mask.float().mean().item())


def canonical_id(entity: str) -> str:
    normalized = unicodedata.normalize("NFKC", entity).casefold().strip()
    return f"capital_of|{normalized}"


def shake_address(identifier: str, dim: int, domain: str) -> torch.Tensor:
    if dim < 1:
        raise ValueError("dim must be positive")
    framed = domain.encode("ascii") + b"\x00" + identifier.encode("utf-8")
    digest = hashlib.shake_256(framed).digest(dim)
    byte_values = torch.frombuffer(bytearray(digest), dtype=torch.uint8).clone()
    signs = torch.where(
        byte_values < 128,
        torch.tensor(-1.0),
        torch.tensor(1.0),
    )
    return unit(signs.float(), dim=0)


def canonical_addresses(
    entities: list[str],
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.stack(
        [shake_address(canonical_id(entity), dim, CANONICAL_DOMAIN) for entity in entities]
    ).to(device)


def off_support_addresses(
    prompts: list[str],
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.stack(
        [shake_address(prompt, dim, OFF_SUPPORT_DOMAIN) for prompt in prompts]
    ).to(device)


@torch.inference_mode()
def collision_registry(
    address_keys: torch.Tensor,
    stream_order: torch.Tensor,
    k_value: int,
    tau_cos: float,
) -> RegistryPlan:
    if k_value < 1 or k_value > stream_order.numel():
        raise ValueError("k_value must fit within stream_order")
    requested = stream_order[:k_value].to(address_keys.device)
    selected = unit(address_keys[requested].float(), dim=-1)

    synchronize(address_keys.device)
    start = time.perf_counter()
    gram = (selected @ selected.T).detach().cpu()
    accepted_positions: list[int] = []
    accepted_flags: list[bool] = []
    max_values: list[float | None] = []
    for position in range(k_value):
        if not accepted_positions:
            max_cosine = None
            accepted = True
        else:
            max_cosine = float(gram[position, accepted_positions].max().item())
            accepted = max_cosine < tau_cos
        max_values.append(max_cosine)
        accepted_flags.append(accepted)
        if accepted:
            accepted_positions.append(position)
    elapsed_ms = 1000.0 * (time.perf_counter() - start) / float(k_value)

    requested_global = requested.detach().cpu().tolist()
    accepted_global = [
        requested_global[index]
        for index, accepted in enumerate(accepted_flags)
        if accepted
    ]
    rejected_global = [
        requested_global[index]
        for index, accepted in enumerate(accepted_flags)
        if not accepted
    ]
    accepted_maxima = [
        value
        for value, accepted in zip(max_values, accepted_flags)
        if accepted and value is not None
    ]
    rejected_maxima = [
        value
        for value, accepted in zip(max_values, accepted_flags)
        if not accepted and value is not None
    ]
    return RegistryPlan(
        requested_global=requested_global,
        accepted_global=accepted_global,
        rejected_global=rejected_global,
        accepted_flags=accepted_flags,
        max_against_accepted=max_values,
        accepted_pairwise_max=max(accepted_maxima, default=0.0),
        rejected_max_cosine=(max(rejected_maxima) if rejected_maxima else None),
        check_ms_per_requested=elapsed_ms,
    )


@torch.inference_mode()
def certify_value_doses_detailed(
    hiddens: torch.Tensor,
    target_ids: torch.Tensor,
    head_weight: torch.Tensor,
    batch_size: int,
    dose_margin: float,
    dose_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """E5-v2 dose formula, retaining each post-dose logit margin."""
    directions = unit(head_weight[target_ids].float(), dim=-1)
    doses = torch.empty(target_ids.shape[0], device=head_weight.device)
    margins_after = torch.empty_like(doses)
    blockers: list[int] = []

    for start in range(0, target_ids.shape[0], batch_size):
        stop = min(start + batch_size, target_ids.shape[0])
        hidden = hiddens[start:stop]
        target = target_ids[start:stop]
        direction = directions[start:stop]
        base = hidden @ head_weight.T
        delta = direction @ head_weight.T
        row = torch.arange(stop - start, device=head_weight.device)
        base_target = base[row, target]
        delta_target = delta[row, target]
        gap = base - base_target[:, None]
        slope = delta_target[:, None] - delta
        required = gap > 0
        blocked = required & (slope <= 1e-8)
        if bool(blocked.any()):
            blocked_rows = torch.nonzero(blocked.any(dim=-1)).flatten() + start
            blockers.extend(blocked_rows.detach().cpu().tolist())

        ratios = torch.where(
            required & (slope > 1e-8),
            gap / slope.clamp_min(1e-8),
            torch.zeros_like(gap),
        )
        lower = ratios.max(dim=-1).values
        dose = dose_margin * lower + dose_epsilon
        doses[start:stop] = dose

        edited_logits = base + dose[:, None] * delta
        target_logits = edited_logits[row, target]
        edited_logits[row, target] = float("-inf")
        margins_after[start:stop] = target_logits - edited_logits.max(dim=-1).values

    if blockers:
        raise RuntimeError(
            f"reachable-target contract failed for fact indices {blockers[:16]}"
        )
    return directions, doses, margins_after


def stream_order(max_k: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(20_000 + seed)
    return torch.randperm(max_k, generator=generator)


def operation_priority(max_k: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(30_000 + seed)
    return torch.randperm(max_k, generator=generator)


def operation_indices(
    accepted_global: list[int],
    priority: torch.Tensor,
) -> dict[str, list[int]]:
    accepted = set(accepted_global)
    ordered = [int(index) for index in priority.tolist() if int(index) in accepted]
    count = len(ordered)
    n_edit = max(1, math.floor(0.10 * count))
    n_deactivate = max(1, math.floor(0.10 * count))
    n_hard = max(1, math.floor(0.05 * count))
    if n_edit + n_deactivate + n_hard >= count:
        raise RuntimeError("accepted population is too small for disjoint operations")
    edit = ordered[:n_edit]
    deactivate = ordered[n_edit : n_edit + n_deactivate]
    hard = ordered[
        n_edit + n_deactivate : n_edit + n_deactivate + n_hard
    ]
    n_reactivate = max(1, math.floor(0.50 * n_deactivate))
    return {
        "edit": edit,
        "deactivate": deactivate,
        "reactivate": deactivate[:n_reactivate],
        "remain_deactivated": deactivate[n_reactivate:],
        "hard_delete": hard,
    }


@torch.inference_mode()
def build_memory(
    address_keys: torch.Tensor,
    accepted_global: list[int],
    values: torch.Tensor,
    doses: torch.Tensor,
    read_kind: str,
    tau_cos: float,
    *,
    shuffled: bool = False,
) -> tuple[EditableHLM5Memory, dict[int, int], float]:
    read_mode = "support_masked" if read_kind == "d5" else "hard_top1"
    memory = EditableHLM5Memory(
        dim=address_keys.shape[1],
        memory_size=len(accepted_global) + 8,
        degree=REGISTERED_DEGREE,
        temperature=REGISTERED_TEMPERATURE,
        learnable_memory=False,
        read_mode=read_mode,
        retrieval_k=1,
        support_threshold=(
            tau_cos**REGISTERED_DEGREE if read_kind == "d5" else None
        ),
    ).to(address_keys.device)
    slot_by_fact: dict[int, int] = {}
    synchronize(address_keys.device)
    start = time.perf_counter()
    count = len(accepted_global)
    for position, fact_index in enumerate(accepted_global):
        key_position = (position + 1) % count if shuffled else position
        key_fact = accepted_global[key_position]
        slot = memory.inject(
            address_keys[key_fact],
            value=values[fact_index],
            value_scale=float(doses[fact_index].item()),
            label=f"fact:{fact_index}",
        )
        slot_by_fact[fact_index] = slot
    synchronize(address_keys.device)
    inject_ms = 1000.0 * (time.perf_counter() - start) / max(1, count)
    return memory, slot_by_fact, inject_ms


def _signature(tensors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def evaluate_operator(
    memory: EditableHLM5Memory,
    query: torch.Tensor,
    hidden: torch.Tensor,
    head_weight: torch.Tensor,
    read_kind: str,
    tau_cos: float,
    batch_size: int,
) -> dict[str, Any]:
    if query.shape != hidden.shape or query.ndim != 2:
        raise ValueError("query and hidden must both have shape [items, dim]")
    if query.shape[0] == 0:
        empty_bool = torch.empty(0, dtype=torch.bool, device=query.device)
        empty_long = torch.empty(0, dtype=torch.long, device=query.device)
        empty_float = torch.empty(0, dtype=hidden.dtype, device=query.device)
        return {
            "prediction": empty_long,
            "routed": empty_bool,
            "selected_slot": empty_long,
            "support_size": empty_long,
            "hidden_changed": empty_bool,
            "selected_gain": empty_float,
            "output": hidden.clone(),
            "score_ms_per_item": 0.0,
            "read_ms_per_item": 0.0,
            "signature": _signature(
                [empty_bool, empty_long, empty_long, hidden, empty_long]
            ),
        }

    predictions: list[torch.Tensor] = []
    routes: list[torch.Tensor] = []
    slots: list[torch.Tensor] = []
    supports: list[torch.Tensor] = []
    changed: list[torch.Tensor] = []
    gains: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    score_seconds = 0.0
    read_seconds = 0.0

    for start_index in range(0, query.shape[0], batch_size):
        q = query[start_index : start_index + batch_size]
        h = hidden[start_index : start_index + batch_size]

        synchronize(query.device)
        score_start = time.perf_counter()
        cosine = torch.einsum(
            "bd,md->bm",
            unit(q.float(), dim=-1),
            unit(memory.keys.float(), dim=-1),
        )
        cosine = cosine.masked_fill(~memory.active[None, :], float("-inf"))
        if read_kind == "d5":
            scores = torch.relu(cosine).pow(REGISTERED_DEGREE)
            scores = scores * torch.relu(memory.alphas)[None, :]
            threshold = tau_cos**REGISTERED_DEGREE
        elif read_kind == "cache":
            scores = cosine
            threshold = tau_cos
        else:
            raise ValueError(f"unknown read_kind: {read_kind}")
        synchronize(query.device)
        score_seconds += time.perf_counter() - score_start

        read_start = time.perf_counter()
        selected_score, selected_slot = scores.max(dim=-1)
        routed = selected_score >= threshold
        support_size = (scores >= threshold).sum(dim=-1)

        if read_kind == "d5":
            supported = scores.masked_fill(scores < threshold, float("-inf"))
            has_support = torch.isfinite(supported).any(dim=-1, keepdim=True)
            safe_scores = torch.where(has_support, supported, torch.zeros_like(supported))
            weights = torch.softmax(safe_scores / memory.temperature, dim=-1)
            weights = torch.where(has_support, weights, torch.zeros_like(weights))
            retrieved = weights @ memory.values
            gain = torch.where(routed, selected_score, torch.zeros_like(selected_score))
        else:
            retrieved = memory.values[selected_slot]
            selected_cosine = torch.where(
                routed,
                selected_score,
                torch.zeros_like(selected_score),
            )
            gain = torch.relu(selected_cosine).pow(REGISTERED_DEGREE)

        output = h + gain[:, None] * retrieved
        prediction = (output @ head_weight.T).argmax(dim=-1)
        hidden_changed = output.ne(h).any(dim=-1)
        reported_slot = torch.where(
            routed,
            selected_slot,
            torch.full_like(selected_slot, -1),
        )
        synchronize(query.device)
        read_seconds += time.perf_counter() - read_start

        predictions.append(prediction)
        routes.append(routed)
        # A winning slot has no routing meaning below threshold. Canonicalize it
        # so matched readers do not disagree merely because inactive ties break
        # differently on a gate-off row.
        slots.append(reported_slot)
        supports.append(support_size)
        changed.append(hidden_changed)
        gains.append(gain)
        outputs.append(output)

    prediction = torch.cat(predictions)
    routed = torch.cat(routes)
    selected_slot = torch.cat(slots)
    support_size = torch.cat(supports)
    hidden_changed = torch.cat(changed)
    selected_gain = torch.cat(gains)
    output = torch.cat(outputs)
    return {
        "prediction": prediction,
        "routed": routed,
        "selected_slot": selected_slot,
        "support_size": support_size,
        "hidden_changed": hidden_changed,
        "selected_gain": selected_gain,
        "output": output,
        "score_ms_per_item": 1000.0 * score_seconds / query.shape[0],
        "read_ms_per_item": 1000.0 * read_seconds / query.shape[0],
        "signature": _signature(
            [routed, selected_slot, support_size, output, prediction]
        ),
    }


def eval_summary(result: dict[str, Any]) -> dict[str, Any]:
    supports = result["support_size"].float()
    gains = result["selected_gain"].float()
    return {
        "route_rate": fraction(result["routed"]),
        "support_size_min": (
            int(supports.min().item()) if supports.numel() else None
        ),
        "support_size_median": (
            float(supports.median().item()) if supports.numel() else None
        ),
        "support_size_max": (
            int(supports.max().item()) if supports.numel() else None
        ),
        "hidden_change_rate": fraction(result["hidden_changed"]),
        "selected_gain_min": float(gains.min().item()) if gains.numel() else None,
        "selected_gain_median": (
            float(gains.median().item()) if gains.numel() else None
        ),
        "gate_off_bit_identical": bool(
            result["output"].eq(result.get("base_hidden", result["output"])).all()
        ) if "base_hidden" in result else None,
        "signature": result["signature"],
        "timing_ms": {
            "score_per_item": result["score_ms_per_item"],
            "read_aggregate_per_item": result["read_ms_per_item"],
        },
    }


def support_route_diagnostics(
    result: dict[str, Any],
    global_indices: list[int] | torch.Tensor,
) -> dict[str, Any]:
    """Report exact support counts and every routed row for a negative assay."""

    support = result["support_size"].detach().cpu().to(torch.long)
    routed = result["routed"].detach().cpu()
    slots = result["selected_slot"].detach().cpu().to(torch.long)
    indices = torch.as_tensor(global_indices, dtype=torch.long).cpu()
    if indices.numel() != support.numel():
        raise ValueError("global index count must match diagnostic rows")
    unique, counts = torch.unique(support, sorted=True, return_counts=True)
    histogram = {
        str(int(value)): int(count)
        for value, count in zip(unique.tolist(), counts.tolist())
    }
    return {
        "count": int(support.numel()),
        "support_size_histogram": histogram,
        "false_route_count": int(routed.sum().item()),
        "false_route_global_indices": indices[routed].tolist(),
        "false_route_slots": slots[routed].tolist(),
    }


def subset_signature(result: dict[str, Any], mask: torch.Tensor) -> str:
    return _signature(
        [
            result["routed"][mask],
            result["selected_slot"][mask],
            result["support_size"][mask],
            result["output"][mask],
            result["prediction"][mask],
        ]
    )


def unique_support_signature(result: dict[str, Any]) -> str:
    """Compare readers only where the preregistered unique-support premise holds."""

    mask = result["support_size"].eq(1)
    return _signature(
        [
            mask,
            result["routed"][mask],
            result["selected_slot"][mask],
            result["support_size"][mask],
            result["output"][mask],
            result["prediction"][mask],
        ]
    )


def memory_tensor_bytes(memory: EditableHLM5Memory) -> int:
    tensors = (
        memory.keys,
        memory.values,
        memory.alphas,
        memory.active,
        memory._reserved,
    )
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def export_lifecycle_bundle(
    memory: EditableHLM5Memory,
    snapshots: dict[int, DeactivatedSlot],
    *,
    address_mode: str,
    address_domain: str,
) -> bytes:
    payload = {
        "schema": "hlm5-memory-lifecycle-bundle-v1",
        "dim": memory.dim,
        "memory_size": memory.memory_size,
        "learnable_memory": False,
        "retrieval_config": memory.retrieval_config(),
        "address_mode": address_mode,
        "address_domain": address_domain,
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in memory.state_dict().items()
        },
        "labels": list(memory.labels),
        "reserved": memory._reserved.detach().cpu().clone(),
        "slot_versions": list(memory._slot_versions),
        "pending_snapshots": {
            str(fact): asdict(snapshot) for fact, snapshot in snapshots.items()
        },
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def restore_lifecycle_bundle(
    blob: bytes,
    device: torch.device,
) -> tuple[EditableHLM5Memory, dict[int, DeactivatedSlot], dict[str, Any]]:
    payload = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
    if payload.get("schema") != "hlm5-memory-lifecycle-bundle-v1":
        raise ValueError("unsupported lifecycle bundle schema")
    config = payload["retrieval_config"]
    memory = EditableHLM5Memory(
        dim=int(payload["dim"]),
        memory_size=int(payload["memory_size"]),
        degree=int(config["degree"]),
        temperature=float(config["temperature"]),
        learnable_memory=bool(payload["learnable_memory"]),
        read_mode=str(config["read_mode"]),
        retrieval_k=int(config["retrieval_k"]),
        support_threshold=config["support_threshold"],
    )
    memory.load_state_dict(payload["state_dict"], strict=True)
    memory._reserved.copy_(payload["reserved"].bool())
    memory.labels = list(payload["labels"])
    memory._slot_versions = [int(value) for value in payload["slot_versions"]]
    memory.to(device)
    snapshots = {
        int(fact): DeactivatedSlot(**snapshot)
        for fact, snapshot in payload["pending_snapshots"].items()
    }
    metadata = {
        "address_mode": payload["address_mode"],
        "address_domain": payload["address_domain"],
    }
    return memory, snapshots, metadata


def memory_state_equal(
    left: EditableHLM5Memory,
    right: EditableHLM5Memory,
) -> bool:
    if left.retrieval_config() != right.retrieval_config():
        return False
    if left.labels != right.labels or left._slot_versions != right._slot_versions:
        return False
    if not torch.equal(left._reserved, right._reserved):
        return False
    return all(
        torch.equal(left.state_dict()[key], right.state_dict()[key])
        for key in left.state_dict()
    )


def _index(values: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=torch.long)


def _phase_record(result: dict[str, Any], base_hidden: torch.Tensor) -> dict[str, Any]:
    summary = eval_summary(result)
    summary["gate_off_bit_identical"] = bool(result["output"].eq(base_hidden).all())
    return summary


@torch.inference_mode()
def run_arm(
    *,
    arm_name: str,
    read_kind: str,
    address_mode: str,
    address_domain: str,
    plan: RegistryPlan,
    address_keys: torch.Tensor,
    fact_hidden: torch.Tensor,
    base_pred: torch.Tensor,
    target_ids: torch.Tensor,
    target_values: torch.Tensor,
    target_doses: torch.Tensor,
    edit_target_ids: torch.Tensor,
    edit_values: torch.Tensor,
    edit_doses: torch.Tensor,
    off_hidden: torch.Tensor,
    off_keys: torch.Tensor,
    head_weight: torch.Tensor,
    priority: torch.Tensor,
    tau_cos: float,
    batch_size: int,
) -> dict[str, Any]:
    device = fact_hidden.device
    requested = _index(plan.requested_global, device)
    accepted = _index(plan.accepted_global, device)
    rejected = _index(plan.rejected_global, device)
    accepted_flags = torch.tensor(plan.accepted_flags, device=device, dtype=torch.bool)

    memory, slot_by_fact, inject_ms = build_memory(
        address_keys,
        plan.accepted_global,
        target_values,
        target_doses,
        read_kind,
        tau_cos,
    )
    create = evaluate_operator(
        memory,
        address_keys[requested],
        fact_hidden[requested],
        head_weight,
        read_kind,
        tau_cos,
        batch_size,
    )
    accepted_create = {
        key: value[accepted_flags] if isinstance(value, torch.Tensor) and value.shape[:1] == accepted_flags.shape else value
        for key, value in create.items()
    }
    expected_slots = torch.tensor(
        [slot_by_fact[index] for index in plan.accepted_global],
        device=device,
        dtype=torch.long,
    )

    operations = operation_indices(plan.accepted_global, priority)
    current_targets = target_ids.clone()
    operation_timing: dict[str, float] = {"inject_per_item": inject_ms}

    synchronize(device)
    start = time.perf_counter()
    for fact in operations["edit"]:
        memory.edit_value(
            slot_by_fact[fact],
            edit_values[fact],
            value_scale=float(edit_doses[fact].item()),
            label=f"fact:{fact}:edited",
        )
        current_targets[fact] = edit_target_ids[fact]
    synchronize(device)
    operation_timing["edit_per_item"] = (
        1000.0 * (time.perf_counter() - start) / len(operations["edit"])
    )
    edit_idx = _index(operations["edit"], device)
    edit_eval = evaluate_operator(
        memory,
        address_keys[edit_idx],
        fact_hidden[edit_idx],
        head_weight,
        read_kind,
        tau_cos,
        batch_size,
    )

    synchronize(device)
    start = time.perf_counter()
    snapshots = {
        fact: memory.deactivate(slot_by_fact[fact])
        for fact in operations["deactivate"]
    }
    synchronize(device)
    operation_timing["deactivate_per_item"] = (
        1000.0 * (time.perf_counter() - start) / len(operations["deactivate"])
    )
    deactivate_idx = _index(operations["deactivate"], device)
    deactivated_before = evaluate_operator(
        memory,
        address_keys[deactivate_idx],
        fact_hidden[deactivate_idx],
        head_weight,
        read_kind,
        tau_cos,
        batch_size,
    )

    synchronize(device)
    start = time.perf_counter()
    blob = export_lifecycle_bundle(
        memory,
        snapshots,
        address_mode=address_mode,
        address_domain=address_domain,
    )
    synchronize(device)
    operation_timing["serialize_total"] = 1000.0 * (time.perf_counter() - start)

    synchronize(device)
    start = time.perf_counter()
    restored, restored_snapshots, restored_metadata = restore_lifecycle_bundle(
        blob,
        device,
    )
    synchronize(device)
    operation_timing["restore_total"] = 1000.0 * (time.perf_counter() - start)
    deactivated_after = evaluate_operator(
        restored,
        address_keys[deactivate_idx],
        fact_hidden[deactivate_idx],
        head_weight,
        read_kind,
        tau_cos,
        batch_size,
    )
    restore_exact = (
        memory_state_equal(memory, restored)
        and deactivated_before["signature"] == deactivated_after["signature"]
        and restored_snapshots == snapshots
        and restored_metadata
        == {"address_mode": address_mode, "address_domain": address_domain}
    )
    require_harness_invariant(
        restore_exact,
        f"bundle restore mismatch for arm={arm_name}",
    )
    memory = restored
    snapshots = restored_snapshots

    synchronize(device)
    start = time.perf_counter()
    for fact in operations["reactivate"]:
        memory.reactivate(snapshots[fact])
    synchronize(device)
    operation_timing["reactivate_per_item"] = (
        1000.0 * (time.perf_counter() - start) / len(operations["reactivate"])
    )
    reactivate_idx = _index(operations["reactivate"], device)
    reactivated = evaluate_operator(
        memory,
        address_keys[reactivate_idx],
        fact_hidden[reactivate_idx],
        head_weight,
        read_kind,
        tau_cos,
        batch_size,
    )

    synchronize(device)
    start = time.perf_counter()
    for fact in operations["hard_delete"]:
        memory.hard_delete(slot_by_fact[fact], rerandomize=False)
    synchronize(device)
    operation_timing["hard_delete_per_item"] = (
        1000.0 * (time.perf_counter() - start) / len(operations["hard_delete"])
    )
    hard_idx = _index(operations["hard_delete"], device)
    hard_deleted = evaluate_operator(
        memory,
        address_keys[hard_idx],
        fact_hidden[hard_idx],
        head_weight,
        read_kind,
        tau_cos,
        batch_size,
    )
    hard_overwritten = all(
        not memory.is_occupied(slot_by_fact[fact])
        and int(torch.count_nonzero(memory.keys[slot_by_fact[fact]])) == 0
        and int(torch.count_nonzero(memory.values[slot_by_fact[fact]])) == 0
        for fact in operations["hard_delete"]
    )

    rejected_idx = _index(plan.rejected_global, device)
    rejected_post_lifecycle = evaluate_operator(
        memory,
        address_keys[rejected_idx],
        fact_hidden[rejected_idx],
        head_weight,
        read_kind,
        tau_cos,
        batch_size,
    )

    touched = set(operations["edit"])
    touched.update(operations["deactivate"])
    touched.update(operations["hard_delete"])
    untouched_global = [fact for fact in plan.accepted_global if fact not in touched]
    untouched_idx = _index(untouched_global, device)
    untouched = evaluate_operator(
        memory,
        address_keys[untouched_idx],
        fact_hidden[untouched_idx],
        head_weight,
        read_kind,
        tau_cos,
        batch_size,
    )
    remain_idx = _index(operations["remain_deactivated"], device)
    remaining_deactivated = evaluate_operator(
        memory,
        address_keys[remain_idx],
        fact_hidden[remain_idx],
        head_weight,
        read_kind,
        tau_cos,
        batch_size,
    )
    off_support = evaluate_operator(
        memory,
        off_keys,
        off_hidden,
        head_weight,
        read_kind,
        tau_cos,
        batch_size,
    )

    accepted_count = len(plan.accepted_global)
    rejected_route = fraction(create["routed"][~accepted_flags])
    create_requested = fraction(create["prediction"].eq(target_ids[requested]))
    create_accepted = fraction(
        create["prediction"][accepted_flags].eq(target_ids[accepted])
    )
    accepted_support_one = bool(
        create["support_size"][accepted_flags].eq(1).all()
    )
    audit_correct = fraction(
        create["selected_slot"][accepted_flags].eq(expected_slots)
    )
    positive_change = fraction(create["hidden_changed"][accepted_flags])

    metrics = {
        "requested_acceptance": accepted_count / len(plan.requested_global),
        "create_requested_target_top1": create_requested,
        "create_accepted_target_top1": create_accepted,
        "create_rejected_target_top1": fraction(
            create["prediction"][~accepted_flags].eq(target_ids[rejected])
        ),
        "accepted_support_exactly_one": accepted_support_one,
        "accepted_top_slot_correct": audit_correct,
        "rejected_query_activation": rejected_route,
        "rejected_post_lifecycle_activation": fraction(
            rejected_post_lifecycle["routed"]
        ),
        "edit_target_top1": fraction(
            edit_eval["prediction"].eq(current_targets[edit_idx])
        ),
        "deactivate_returns_base_top1": fraction(
            deactivated_before["prediction"].eq(base_pred[deactivate_idx])
        ),
        "deactivate_hidden_bit_identical": bool(
            deactivated_before["output"].eq(fact_hidden[deactivate_idx]).all()
        ),
        "bundle_restore_exact": bool(restore_exact),
        "reactivate_target_top1": fraction(
            reactivated["prediction"].eq(current_targets[reactivate_idx])
        ),
        "hard_delete_live_support_zero": bool(
            hard_deleted["support_size"].eq(0).all()
        ),
        "hard_delete_returns_base_top1": fraction(
            hard_deleted["prediction"].eq(base_pred[hard_idx])
        ),
        "hard_delete_hidden_bit_identical": bool(
            hard_deleted["output"].eq(fact_hidden[hard_idx]).all()
        ),
        "hard_delete_overwritten_free": bool(hard_overwritten),
        "untouched_target_top1": fraction(
            untouched["prediction"].eq(current_targets[untouched_idx])
        ),
        "remaining_deactivated_returns_base_top1": fraction(
            remaining_deactivated["prediction"].eq(base_pred[remain_idx])
        ),
        "off_support_activation": fraction(off_support["routed"]),
        "off_support_hidden_bit_identical": bool(
            off_support["output"].eq(off_hidden).all()
        ),
        "live_positive_control_hidden_change": positive_change,
    }
    minimum_population = min(20, len(plan.requested_global))
    gates = {
        "acceptance==1": metrics["requested_acceptance"] == 1.0,
        "accepted_count>=minimum": accepted_count >= minimum_population,
        "create_requested>=0.99": create_requested is not None and create_requested >= 0.99,
        "accepted_support==1": accepted_support_one,
        "accepted_top_slot>=0.99": audit_correct is not None and audit_correct >= 0.99,
        "edit>=0.99": metrics["edit_target_top1"] is not None and metrics["edit_target_top1"] >= 0.99,
        "deactivate_base==1": metrics["deactivate_returns_base_top1"] == 1.0,
        "deactivate_hidden_identity": metrics["deactivate_hidden_bit_identical"],
        "bundle_restore_exact": metrics["bundle_restore_exact"],
        "reactivate>=0.99": metrics["reactivate_target_top1"] is not None and metrics["reactivate_target_top1"] >= 0.99,
        "hard_delete_support==0": metrics["hard_delete_live_support_zero"],
        "hard_delete_base==1": metrics["hard_delete_returns_base_top1"] == 1.0,
        "hard_delete_hidden_identity": metrics["hard_delete_hidden_bit_identical"],
        "hard_delete_overwritten_free": metrics["hard_delete_overwritten_free"],
        "untouched>=0.99": metrics["untouched_target_top1"] is not None and metrics["untouched_target_top1"] >= 0.99,
        "remaining_deactivated_base==1": metrics["remaining_deactivated_returns_base_top1"] in {None, 1.0},
        "off_support<=0.01": metrics["off_support_activation"] is not None and metrics["off_support_activation"] <= 0.01,
        "off_support_hidden_identity": metrics["off_support_hidden_bit_identical"],
        "positive_control>=0.99": positive_change is not None and positive_change >= 0.99,
    }
    gates["all_primary"] = all(gates.values())

    phase_signatures = {
        "create_accepted": subset_signature(create, accepted_flags),
        "edit": edit_eval["signature"],
        "deactivated_before": deactivated_before["signature"],
        "deactivated_after_restore": deactivated_after["signature"],
        "reactivated": reactivated["signature"],
        "hard_deleted": hard_deleted["signature"],
        "rejected_post_unique_support": unique_support_signature(
            rejected_post_lifecycle
        ),
        "untouched": untouched["signature"],
        "remaining_deactivated": remaining_deactivated["signature"],
        "off_support": off_support["signature"],
    }
    return {
        "arm": arm_name,
        "read_kind": read_kind,
        "metrics": metrics,
        "gates": gates,
        "operations": operations,
        "operation_indices_sha256": json_sha256(operations),
        "phases": {
            "create_requested": _phase_record(
                create,
                fact_hidden[requested],
            ),
            "edit": _phase_record(edit_eval, fact_hidden[edit_idx]),
            "deactivated_before": _phase_record(
                deactivated_before,
                fact_hidden[deactivate_idx],
            ),
            "deactivated_after_restore": _phase_record(
                deactivated_after,
                fact_hidden[deactivate_idx],
            ),
            "reactivated": _phase_record(
                reactivated,
                fact_hidden[reactivate_idx],
            ),
            "hard_deleted": _phase_record(
                hard_deleted,
                fact_hidden[hard_idx],
            ),
            "rejected_post_lifecycle": _phase_record(
                rejected_post_lifecycle,
                fact_hidden[rejected_idx],
            ),
            "untouched": _phase_record(untouched, fact_hidden[untouched_idx]),
            "remaining_deactivated": _phase_record(
                remaining_deactivated,
                fact_hidden[remain_idx],
            ),
            "off_support": _phase_record(off_support, off_hidden),
        },
        "phase_signatures": phase_signatures,
        "negative_query_diagnostics": {
            "rejected_post_lifecycle": support_route_diagnostics(
                rejected_post_lifecycle,
                plan.rejected_global,
            ),
            "hard_deleted": support_route_diagnostics(
                hard_deleted,
                operations["hard_delete"],
            ),
        },
        "memory_tensor_bytes": memory_tensor_bytes(memory),
        "serialized_bundle_bytes": len(blob),
        "timing_ms": operation_timing,
    }


@torch.inference_mode()
def run_shuffled_null(
    *,
    plan: RegistryPlan,
    address_keys: torch.Tensor,
    fact_hidden: torch.Tensor,
    target_ids: torch.Tensor,
    target_values: torch.Tensor,
    target_doses: torch.Tensor,
    head_weight: torch.Tensor,
    tau_cos: float,
    batch_size: int,
) -> dict[str, Any]:
    device = fact_hidden.device
    requested = _index(plan.requested_global, device)
    memory, _, inject_ms = build_memory(
        address_keys,
        plan.accepted_global,
        target_values,
        target_doses,
        "d5",
        tau_cos,
        shuffled=True,
    )
    result = evaluate_operator(
        memory,
        address_keys[requested],
        fact_hidden[requested],
        head_weight,
        "d5",
        tau_cos,
        batch_size,
    )
    target_success = fraction(result["prediction"].eq(target_ids[requested]))
    return {
        "arm": "canonical_shuffled_null",
        "create_target_top1": target_success,
        "route_rate": fraction(result["routed"]),
        "support_size_max": int(result["support_size"].max().item()),
        "gate_pass": target_success is not None and target_success <= 0.01,
        "signature": result["signature"],
        "timing_ms": {
            "inject_per_item": inject_ms,
            "score_per_item": result["score_ms_per_item"],
            "read_aggregate_per_item": result["read_ms_per_item"],
        },
    }


def pair_equivalence(
    degree5: dict[str, Any],
    cache: dict[str, Any],
) -> dict[str, Any]:
    phases = sorted(set(degree5["phase_signatures"]) | set(cache["phase_signatures"]))
    matches = {
        phase: degree5["phase_signatures"].get(phase)
        == cache["phase_signatures"].get(phase)
        for phase in phases
    }
    return {
        "phase_matches": matches,
        "exact": all(matches.values()),
    }


def select_formal_verdict(
    *,
    smoke: bool,
    harness_invalid: bool,
    track_progress: dict[str, dict[str, Any]],
) -> str:
    """Resolve the package verdict with invalidity taking precedence."""

    if harness_invalid:
        return "HARNESS_INVALID"
    if smoke:
        return "SMOKE_ONLY"
    if track_progress["hidden"]["max_passing_k"] == 4096:
        return "HIDDEN_CACHE_EQUIVALENT"
    if track_progress["canonical"]["max_passing_k"] == 4096:
        return "CANONICAL_CACHE_EQUIVALENT"
    return "ADDRESS_CONDITIONING_FAILED"


def registry_record(plan: RegistryPlan) -> dict[str, Any]:
    record = asdict(plan)
    record["requested"] = len(plan.requested_global)
    record["accepted"] = len(plan.accepted_global)
    record["rejected"] = len(plan.rejected_global)
    record["acceptance"] = len(plan.accepted_global) / len(plan.requested_global)
    record["requested_indices_sha256"] = json_sha256(plan.requested_global)
    record["accepted_indices_sha256"] = json_sha256(plan.accepted_global)
    record["timing_ms"] = {
        "collision_check_per_requested": record.pop("check_ms_per_requested")
    }
    return record


def verify_registration(path: Path) -> dict[str, Any]:
    registration = json.loads(path.read_text(encoding="utf-8"))
    protocol = registration["protocol"]
    if sha256_file(Path(protocol["path"])) != protocol["sha256"]:
        raise RuntimeError("registered protocol hash mismatch")
    for artifact in registration["frozen_artifacts"].values():
        if sha256_file(Path(artifact["path"])) != artifact["sha256"]:
            raise RuntimeError(f"frozen artifact hash mismatch: {artifact['path']}")
    for file_path, expected_hash in registration["implementation_base"].items():
        if sha256_file(Path(file_path)) != expected_hash:
            raise RuntimeError(f"implementation-base hash mismatch: {file_path}")
    return registration


def _scientific_projection(value: Any) -> Any:
    if isinstance(value, dict):
        projected = {}
        for key, item in value.items():
            if key in {
                "command",
                "environment",
                "generated_at_local",
                "peak_cuda_memory_bytes",
                "scientific_sha256",
                "timing_ms",
            }:
                continue
            projected[key] = _scientific_projection(item)
        return projected
    if isinstance(value, list):
        return [_scientific_projection(item) for item in value]
    return value


def scientific_sha256(report: dict[str, Any]) -> str:
    return json_sha256(_scientific_projection(report))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--k-values",
        type=parse_int_csv,
        default=REGISTERED_K.copy(),
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_csv,
        default=REGISTERED_SEEDS.copy(),
    )
    parser.add_argument("--tau-cos", type=float, default=REGISTERED_TAU_COS)
    parser.add_argument(
        "--calibration-count",
        type=int,
        default=REGISTERED_CALIBRATION_COUNT,
    )
    parser.add_argument(
        "--floor-fraction",
        type=float,
        default=REGISTERED_FLOOR_FRACTION,
    )
    parser.add_argument("--hidden-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--target-scan-batch-size", type=int, default=256)
    parser.add_argument("--dose-batch-size", type=int, default=16)
    parser.add_argument("--dose-margin", type=float, default=REGISTERED_DOSE_MARGIN)
    parser.add_argument("--dose-epsilon", type=float, default=REGISTERED_DOSE_EPSILON)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if not 0 < args.tau_cos <= 1:
        parser.error("--tau-cos must be in (0, 1]")
    if args.calibration_count != REGISTERED_CALIBRATION_COUNT:
        parser.error("E5-v3 fixes --calibration-count at 256")
    if args.tau_cos != REGISTERED_TAU_COS:
        parser.error("E5-v3 fixes tau_cos=0.95")
    if args.floor_fraction != REGISTERED_FLOOR_FRACTION:
        parser.error("E5-v3 fixes floor_fraction=0.01")
    if args.dose_margin != REGISTERED_DOSE_MARGIN:
        parser.error("E5-v3 fixes dose_margin=1.05")
    if args.dose_epsilon != REGISTERED_DOSE_EPSILON:
        parser.error("E5-v3 fixes dose_epsilon=0.001")
    if args.smoke:
        args.k_values = [16]
        args.seeds = [1]
    else:
        if args.k_values != REGISTERED_K or args.seeds != REGISTERED_SEEDS:
            parser.error("registered runs require the fixed K values and seeds")
    args.out = args.out or (SMOKE_OUTPUT if args.smoke else DEFAULT_OUTPUT)
    return args


def main() -> None:
    args = parse_args()
    registration_path = args.registration.resolve()
    registration = verify_registration(registration_path)
    checkpoint = args.checkpoint.resolve()
    tokenizer_path = args.tokenizer.resolve()
    if sha256_file(checkpoint) != registration["frozen_artifacts"]["checkpoint"]["sha256"]:
        raise RuntimeError("CLI checkpoint differs from registration")
    if sha256_file(tokenizer_path) != registration["frozen_artifacts"]["tokenizer"]["sha256"]:
        raise RuntimeError("CLI tokenizer differs from registration")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(0)
    print(f"loading frozen checkpoint on {device}: {checkpoint}")
    model, checkpoint_data = load_model(checkpoint, device)
    tokenizer = HFTokenizer.from_file(str(tokenizer_path))
    head_weight = model.head.weight.detach().float()

    max_k = max(args.k_values)
    fact_names = make_entity_names(max_k, "fact")
    calibration_names = make_entity_names(args.calibration_count, "calibration")
    fact_prompts = [f"The capital of {name} is" for name in fact_names]
    calibration_prompts = [
        f"The capital of {name} is" for name in calibration_names
    ]
    near_miss_prompts = [
        f"The capital city of {name} is" for name in fact_names[: min(64, max_k)]
    ]
    off_prompts = NEUTRAL_PROMPTS + near_miss_prompts

    print(
        f"preparing fixed population facts={max_k} "
        f"calibration={args.calibration_count} off={len(off_prompts)}"
    )
    fact_hidden = extract_last_hiddens(
        model,
        tokenizer,
        fact_prompts,
        device,
        args.hidden_batch_size,
    )
    calibration_hidden = extract_last_hiddens(
        model,
        tokenizer,
        calibration_prompts,
        device,
        args.hidden_batch_size,
    )
    off_hidden = extract_last_hiddens(
        model,
        tokenizer,
        off_prompts,
        device,
        args.hidden_batch_size,
    )
    key_mean, key_transform, calibration_meta = fit_fixed_key_transform(
        calibration_hidden,
        off_hidden[: len(NEUTRAL_PROMPTS)],
        args.floor_fraction,
    )
    hidden_keys = transformed_keys(fact_hidden, key_mean, key_transform)
    hidden_off_keys = transformed_keys(off_hidden, key_mean, key_transform)
    canonical_keys = canonical_addresses(fact_names, model.dim, device)
    canonical_off_keys = off_support_addresses(off_prompts, model.dim, device)
    base_pred = base_predictions(fact_hidden, head_weight, args.eval_batch_size)

    target_pool_needed = max_k + 64
    reachable_pool, target_pool_meta = find_reachable_target_pool(
        head_weight,
        target_pool_needed,
        seed=1729,
        scan_batch_size=args.target_scan_batch_size,
    )
    prepared_by_seed: dict[int, dict[str, torch.Tensor]] = {}
    seed_preparation: dict[str, Any] = {}
    for seed in args.seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.randperm(reachable_pool.numel(), generator=generator)
        ordered_pool = reachable_pool[permutation.to(device)]
        targets = assign_distinct_targets(ordered_pool, base_pred, max_k, offset=0)
        edit_targets = make_replacement_targets(targets, base_pred)
        target_values, target_doses, target_margins = certify_value_doses_detailed(
            fact_hidden,
            targets,
            head_weight,
            args.dose_batch_size,
            args.dose_margin,
            args.dose_epsilon,
        )
        edit_values, edit_doses, edit_margins = certify_value_doses_detailed(
            fact_hidden,
            edit_targets,
            head_weight,
            args.dose_batch_size,
            args.dose_margin,
            args.dose_epsilon,
        )
        prepared_by_seed[seed] = {
            "target_ids": targets,
            "target_values": target_values,
            "target_doses": target_doses,
            "target_margins": target_margins,
            "edit_target_ids": edit_targets,
            "edit_values": edit_values,
            "edit_doses": edit_doses,
            "edit_margins": edit_margins,
        }
        seed_preparation[str(seed)] = {
            "target_ids_sha256": tensor_sha256(targets),
            "edit_target_ids_sha256": tensor_sha256(edit_targets),
            "stream_order_sha256": tensor_sha256(stream_order(max_k, seed)),
            "operation_priority_sha256": tensor_sha256(
                operation_priority(max_k, seed)
            ),
        }

    report: dict[str, Any] = {
        "schema": "hlm5-e5v3-address-conditioning-v1",
        "measurement_class": "diagnostic_smoke" if args.smoke else "registered_measurement",
        "generated_at_local": dt.datetime.now().astimezone().isoformat(),
        "command": [sys.executable, *sys.argv],
        "registration": {
            "path": str(registration_path),
            "sha256": sha256_file(registration_path),
            "protocol_sha256": registration["protocol"]["sha256"],
            "timing_attestation": registration["timing_attestation"],
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "artifacts": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "tokenizer": str(tokenizer_path),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "checkpoint_cfg": checkpoint_data["cfg"],
            "vocab": int(checkpoint_data["vocab"]),
            "implementation_sha256": {
                str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
                str(ROOT / "tests" / "test_address_conditioning_v1.py"): sha256_file(
                    ROOT / "tests" / "test_address_conditioning_v1.py"
                ),
                str(ROOT / "hlm5_memory.py"): sha256_file(ROOT / "hlm5_memory.py"),
                str(ROOT / "hlm5_lm.py"): sha256_file(ROOT / "hlm5_lm.py"),
                str(ROOT / "baselines" / "run_seq_edit_stream_v2.py"): sha256_file(
                    ROOT / "baselines" / "run_seq_edit_stream_v2.py"
                ),
            },
        },
        "protocol": {
            "k_values": args.k_values,
            "seeds": args.seeds,
            "tau_cos": args.tau_cos,
            "tau_score": args.tau_cos**REGISTERED_DEGREE,
            "degree": REGISTERED_DEGREE,
            "temperature": REGISTERED_TEMPERATURE,
            "calibration_count": args.calibration_count,
            "whitening_floor_fraction": args.floor_fraction,
            "dose_margin": args.dose_margin,
            "dose_epsilon": args.dose_epsilon,
            "calibration": calibration_meta,
            "target_pool": target_pool_meta,
            "canonical_domain": CANONICAL_DOMAIN,
            "off_support_domain": OFF_SUPPORT_DOMAIN,
            "population_hashes": {
                "fact_names": json_sha256(fact_names),
                "fact_prompts": json_sha256(fact_prompts),
                "off_prompts": json_sha256(off_prompts),
                "hidden_keys": tensor_sha256(hidden_keys),
                "canonical_keys": tensor_sha256(canonical_keys),
            },
        },
        "seed_preparation": seed_preparation,
        "rungs": [],
        "track_progress": {
            "hidden": {"active": True, "max_passing_k": None, "first_failed_k": None},
            "canonical": {"active": True, "max_passing_k": None, "first_failed_k": None},
        },
        "stopped_early": False,
        "stop_reason": None,
    }

    previous_flags: dict[tuple[str, int], list[bool]] = {}
    harness_invalid = False
    for k_value in args.k_values:
        print(f"\n=== E5-v3 K={k_value} ===")
        rung: dict[str, Any] = {"K": k_value, "seeds": []}
        per_track_seed_pass: dict[str, list[bool]] = {
            "hidden": [],
            "canonical": [],
        }
        for seed in args.seeds:
            prepared = prepared_by_seed[seed]
            order = stream_order(max_k, seed)
            priority = operation_priority(max_k, seed)
            requested = order[:k_value].to(device)
            population = {
                "requested_indices_sha256": tensor_sha256(requested),
                "target_dose": describe_tensor(prepared["target_doses"][requested]),
                "target_post_margin": describe_tensor(
                    prepared["target_margins"][requested]
                ),
                "edit_dose": describe_tensor(prepared["edit_doses"][requested]),
                "edit_post_margin": describe_tensor(
                    prepared["edit_margins"][requested]
                ),
            }
            seed_record: dict[str, Any] = {
                "seed": seed,
                "population": population,
                "tracks": {},
            }
            for track_name, address_keys, off_keys, domain in (
                ("hidden", hidden_keys, hidden_off_keys, "fixed_hidden_transform"),
                ("canonical", canonical_keys, canonical_off_keys, CANONICAL_DOMAIN),
            ):
                progress = report["track_progress"][track_name]
                if not progress["active"]:
                    continue
                plan = collision_registry(
                    address_keys,
                    order,
                    k_value,
                    args.tau_cos,
                )
                prefix_key = (track_name, seed)
                old_flags = previous_flags.get(prefix_key, [])
                if plan.accepted_flags[: len(old_flags)] != old_flags:
                    harness_invalid = True
                    message = (
                        "nested-prefix decision mismatch "
                        f"track={track_name} seed={seed}"
                    )
                    seed_record["tracks"][track_name] = {
                        "registry": registry_record(plan),
                        "harness_error": message,
                    }
                    print(f"HARNESS_INVALID: {message}")
                    break
                previous_flags[prefix_key] = list(plan.accepted_flags)

                try:
                    d5 = run_arm(
                        arm_name=f"{track_name}_d5",
                        read_kind="d5",
                        address_mode=track_name,
                        address_domain=domain,
                        plan=plan,
                        address_keys=address_keys,
                        fact_hidden=fact_hidden,
                        base_pred=base_pred,
                        target_ids=prepared["target_ids"],
                        target_values=prepared["target_values"],
                        target_doses=prepared["target_doses"],
                        edit_target_ids=prepared["edit_target_ids"],
                        edit_values=prepared["edit_values"],
                        edit_doses=prepared["edit_doses"],
                        off_hidden=off_hidden,
                        off_keys=off_keys,
                        head_weight=head_weight,
                        priority=priority,
                        tau_cos=args.tau_cos,
                        batch_size=args.eval_batch_size,
                    )
                    cache = run_arm(
                        arm_name=f"{track_name}_cache",
                        read_kind="cache",
                        address_mode=track_name,
                        address_domain=domain,
                        plan=plan,
                        address_keys=address_keys,
                        fact_hidden=fact_hidden,
                        base_pred=base_pred,
                        target_ids=prepared["target_ids"],
                        target_values=prepared["target_values"],
                        target_doses=prepared["target_doses"],
                        edit_target_ids=prepared["edit_target_ids"],
                        edit_values=prepared["edit_values"],
                        edit_doses=prepared["edit_doses"],
                        off_hidden=off_hidden,
                        off_keys=off_keys,
                        head_weight=head_weight,
                        priority=priority,
                        tau_cos=args.tau_cos,
                        batch_size=args.eval_batch_size,
                    )
                except HarnessInvalidError as exc:
                    harness_invalid = True
                    seed_record["tracks"][track_name] = {
                        "registry": registry_record(plan),
                        "harness_error": str(exc),
                    }
                    print(f"HARNESS_INVALID: {exc}")
                    break
                equivalence = pair_equivalence(d5, cache)
                null_result = None
                if track_name == "canonical":
                    null_result = run_shuffled_null(
                        plan=plan,
                        address_keys=address_keys,
                        fact_hidden=fact_hidden,
                        target_ids=prepared["target_ids"],
                        target_values=prepared["target_values"],
                        target_doses=prepared["target_doses"],
                        head_weight=head_weight,
                        tau_cos=args.tau_cos,
                        batch_size=args.eval_batch_size,
                    )
                track_pass = (
                    d5["gates"]["all_primary"]
                    and cache["gates"]["all_primary"]
                    and equivalence["exact"]
                    and (null_result is None or null_result["gate_pass"])
                )
                per_track_seed_pass[track_name].append(bool(track_pass))
                seed_record["tracks"][track_name] = {
                    "registry": registry_record(plan),
                    "degree5": d5,
                    "cache": cache,
                    "equivalence": equivalence,
                    "shuffled_null": null_result,
                    "track_pass": bool(track_pass),
                }
                print(
                    f"seed={seed} track={track_name} "
                    f"accepted={len(plan.accepted_global)}/{k_value} "
                    f"d5={d5['metrics']['create_requested_target_top1']:.3f} "
                    f"cache={cache['metrics']['create_requested_target_top1']:.3f} "
                    f"pass={track_pass}"
                )
                if not equivalence["exact"] or (
                    null_result is not None and not null_result["gate_pass"]
                ):
                    harness_invalid = True
                    print(
                        "HARNESS_INVALID: operator equivalence or shuffled-null "
                        f"contract failed track={track_name} seed={seed}"
                    )
                    break
            rung["seeds"].append(seed_record)
            if harness_invalid:
                break

        rung["track_pass_all_seeds"] = {}
        if harness_invalid:
            rung["track_pass_all_seeds"] = {"hidden": None, "canonical": None}
        else:
            for track_name in ("hidden", "canonical"):
                progress = report["track_progress"][track_name]
                if not progress["active"]:
                    rung["track_pass_all_seeds"][track_name] = None
                    continue
                passed = bool(per_track_seed_pass[track_name]) and all(
                    per_track_seed_pass[track_name]
                )
                rung["track_pass_all_seeds"][track_name] = passed
                if passed:
                    progress["max_passing_k"] = k_value
                else:
                    progress["first_failed_k"] = k_value
                    progress["active"] = False
        report["rungs"].append(rung)

        if harness_invalid:
            report["stopped_early"] = True
            report["stop_reason"] = (
                f"harness invariant or shuffled-null failure at K={k_value}"
            )
            break
        if args.smoke:
            break
        if not any(
            progress["active"] for progress in report["track_progress"].values()
        ):
            report["stopped_early"] = True
            report["stop_reason"] = f"both address tracks failed by K={k_value}"
            break

    report["peak_cuda_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    report["formal_verdict"] = select_formal_verdict(
        smoke=args.smoke,
        harness_invalid=harness_invalid,
        track_progress=report["track_progress"],
    )
    report["scientific_sha256"] = scientific_sha256(report)

    output_path = args.out.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved {report['measurement_class']} -> {output_path}")
    print(f"formal verdict: {report['formal_verdict']}")
    print(f"scientific sha256: {report['scientific_sha256']}")


if __name__ == "__main__":
    main()

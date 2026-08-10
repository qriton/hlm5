"""Nested-capacity and interference runtime for registered HLM5 E10."""

from __future__ import annotations

import json
import math
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Iterable

import torch

from .certify import calibrate_whitening
from .e7_contract import sha256_file, stable_json_sha256, tensor_sha256
from .e7_runtime import native_head_logits
from .e7b_runtime import NATIVE_ADMIT, environment_record as base_environment_record
from .e7c_runtime import raw_and_zca_directions
from .e8_runtime import (
    baseline_case_rows,
    direct_admission_rows,
    direct_summary,
    paired_geometry,
    strongest_competitor,
)
from .e9_runtime import KINDS, SCORE_THRESHOLDS, query_specs
from .memory import EditableHLM5Memory
from .public_adapter import HLM5PreHeadAdapter


CANDIDATE_START = 20_077
CANDIDATE_COUNT = 1_200
CAPACITY_TIERS = (64, 256, 1_024)
MEMORY_SIZE = 1_032
DEGREE = 5
TEMPERATURE = 0.10
BOOST = 1.0
GATE_THRESHOLD = 0.95
KEY_FLOOR_FRACTION = 0.01
MODEL_FORWARD_BATCH = 16
GATE_BATCH = 256
TOP_K = 20


def e10_environment_record(device: torch.device) -> dict[str, Any]:
    """Extend the arithmetic environment with the registered physical node."""
    value = base_environment_record(device)
    value["node"] = platform.node()
    return value


def select_capacity_cases(
    records: list[dict[str, Any]],
    tokenizer: Any,
    excluded_prompts: Iterable[str],
) -> list[dict[str, Any]]:
    """Select the frozen E10 population using data/tokenizer structure only."""
    excluded = set(excluded_prompts)
    used: set[str] = set()
    rows: list[dict[str, Any]] = []
    for record in records:
        case_id = int(record["case_id"])
        if case_id < CANDIDATE_START:
            continue
        rewrite = record["requested_rewrite"]
        prompt = str(rewrite["prompt"]).format(rewrite["subject"])
        target = " " + str(rewrite["target_new"]["str"]).strip()
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if (
            len(target_ids) != 1
            or not prompt
            or prompt in excluded
            or prompt in used
        ):
            continue
        rows.append(
            {
                "case_id": case_id,
                "prompt": prompt,
                "target": target,
                "target_id": int(target_ids[0]),
                "relation_id": str(rewrite["relation_id"]),
                "subject": str(rewrite["subject"]),
            }
        )
        used.add(prompt)
        if len(rows) == CANDIDATE_COUNT:
            break
    if len(rows) != CANDIDATE_COUNT:
        raise RuntimeError(
            f"E10 selected {len(rows)} candidates, expected {CANDIDATE_COUNT}"
        )
    return rows


def load_capacity_cases(
    data_path: Path,
    tokenizer: Any,
    excluded_prompts: Iterable[str],
) -> list[dict[str, Any]]:
    records = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError("CounterFact data must be a JSON list")
    return select_capacity_cases(records, tokenizer, excluded_prompts)


def linear_median(values: Iterable[float]) -> float:
    """Return the registered float64 linear 0.5 quantile."""
    tensor = torch.tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise ValueError("E10 median requires finite nonempty values")
    return float(torch.quantile(tensor, 0.5, interpolation="linear"))


def population_record(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_count": len(cases),
        "first_case_id": int(cases[0]["case_id"]),
        "last_case_id": int(cases[-1]["case_id"]),
        "case_sha256": stable_json_sha256(cases),
        "case_id_sha256": stable_json_sha256([row["case_id"] for row in cases]),
        "prompt_sha256": stable_json_sha256([row["prompt"] for row in cases]),
        "target_id_sha256": stable_json_sha256(
            [row["target_id"] for row in cases]
        ),
        "target_sha256": stable_json_sha256([row["target"] for row in cases]),
    }


@torch.inference_mode()
def direct_measurement(
    model: Any,
    cases: list[dict[str, Any]],
    exact_hidden: torch.Tensor,
) -> dict[str, Any]:
    """Run the frozen E7c/E8 direct admission for all E10 candidates."""
    target_ids = torch.tensor(
        [row["target_id"] for row in cases], device=exact_hidden.device
    )
    head = model.get_output_embeddings().weight.detach()
    _raw, directions, basis = raw_and_zca_directions(head, target_ids)
    geometry = paired_geometry(head.to(torch.float64), exact_hidden, target_ids, directions)
    baseline = baseline_case_rows(model, exact_hidden, cases)
    rows = direct_admission_rows(model, exact_hidden, directions, geometry, baseline)
    summary = direct_summary(rows)
    selected_indices = [
        index for index, row in enumerate(rows) if row["deployment_admitted"]
    ][: CAPACITY_TIERS[-1]]
    selected_case_ids = [int(cases[index]["case_id"]) for index in selected_indices]
    summary.update(
        {
            "selected_count": len(selected_indices),
            "selected_indices": selected_indices,
            "selected_case_ids": selected_case_ids,
            "selected_indices_sha256": stable_json_sha256(selected_indices),
            "selected_case_ids_sha256": stable_json_sha256(selected_case_ids),
        }
    )
    return {
        "basis_and_directions": basis,
        "directions": directions,
        "baseline_rows": baseline,
        "direct_rows": rows,
        "direct_summary": summary,
    }


@torch.inference_mode()
def shared_key_operator(
    exact_hidden: torch.Tensor,
    whitening_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    mean, transform = calibrate_whitening(
        exact_hidden.float(),
        whitening_hidden.float(),
        floor_frac=KEY_FLOOR_FRACTION,
    )
    record = {
        "mean_sha256": tensor_sha256(mean),
        "transform_sha256": tensor_sha256(transform),
        "mean_dtype": str(mean.dtype).removeprefix("torch."),
        "mean_shape": list(mean.shape),
        "transform_dtype": str(transform.dtype).removeprefix("torch."),
        "transform_shape": list(transform.shape),
        "floor_fraction": KEY_FLOOR_FRACTION,
        "exact_key_count": int(exact_hidden.shape[0]),
        "whitening_prompt_count": int(whitening_hidden.shape[0]),
    }
    return mean, transform, record


@torch.inference_mode()
def build_capacity_memory(
    cases: list[dict[str, Any]],
    exact_hidden: torch.Tensor,
    direct_rows: list[dict[str, Any]],
    directions: torch.Tensor,
    selected_indices: list[int],
    key_mean: torch.Tensor,
    key_transform: torch.Tensor,
    active_count: int,
) -> tuple[EditableHLM5Memory, HLM5PreHeadAdapter, dict[int, int], dict[str, Any]]:
    if active_count not in CAPACITY_TIERS or len(selected_indices) < active_count:
        raise ValueError("E10 tier is not constructible")
    memory = EditableHLM5Memory(
        dim=exact_hidden.shape[1],
        memory_size=MEMORY_SIZE,
        degree=DEGREE,
        temperature=TEMPERATURE,
        learnable_memory=False,
    ).to(exact_hidden.device)
    slot_by_candidate: dict[int, int] = {}
    for candidate_index in selected_indices[:active_count]:
        row = direct_rows[candidate_index]
        if not row["deployment_admitted"] or row.get("decision") != NATIVE_ADMIT:
            raise RuntimeError("E10 selected a non-admitted candidate")
        key = (exact_hidden[candidate_index].float() - key_mean) @ key_transform
        slot = memory.inject(
            key,
            value=directions[candidate_index].float(),
            label=str(cases[candidate_index]["target"]),
        )
        expected_slot = len(slot_by_candidate)
        if slot != expected_slot:
            raise RuntimeError("E10 active memory is not an exact prefix")
        value = directions[candidate_index] * float(row["beta"])
        memory.values[slot].copy_(value.float())
        slot_by_candidate[candidate_index] = slot
    adapter = HLM5PreHeadAdapter(
        memory,
        key_mean,
        key_transform,
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
        "selected_candidate_indices": list(selected_indices[:active_count]),
        "slot_by_candidate": {str(k): v for k, v in slot_by_candidate.items()},
        "active_mask_sha256": tensor_sha256(memory.active),
        "active_keys_sha256": tensor_sha256(memory.keys[active].contiguous()),
        "active_values_sha256": tensor_sha256(memory.values[active].contiguous()),
        "active_alphas_sha256": tensor_sha256(memory.alphas[active].contiguous()),
        "active_labels": [memory.labels[int(slot)] for slot in active],
        "key_mean_sha256": tensor_sha256(key_mean),
        "key_transform_sha256": tensor_sha256(key_transform),
    }
    return memory, adapter, slot_by_candidate, receipt


@torch.inference_mode()
def scan_positives(
    model: Any,
    adapter: HLM5PreHeadAdapter,
    memory: EditableHLM5Memory,
    cases: list[dict[str, Any]],
    exact_hidden: torch.Tensor,
    selected_indices: list[int],
    slot_by_candidate: dict[int, int],
) -> dict[str, Any]:
    indices = list(selected_indices[: int(memory.active.sum())])
    native_hidden = exact_hidden[indices]
    adapted, audit = adapter(native_hidden, return_audit=True)
    if audit is None:
        raise RuntimeError("E10 positive scan omitted audit")
    rows: list[dict[str, Any]] = []
    for start in range(0, len(indices), MODEL_FORWARD_BATCH):
        end = min(start + MODEL_FORWARD_BATCH, len(indices))
        logits = native_head_logits(model, adapted[start:end])
        for offset, candidate_index in enumerate(indices[start:end]):
            local = start + offset
            case = cases[candidate_index]
            own_slot = slot_by_candidate[candidate_index]
            score_row = audit.scores[local].detach().cpu()
            active_scores = score_row[: len(indices)]
            nonown = active_scores.clone()
            nonown[own_slot] = float("-inf")
            strongest_nonown_score, strongest_nonown_slot = nonown.max(dim=0)
            if len(indices) == 1:
                strongest_nonown_score = torch.tensor(float("-inf"))
                strongest_nonown_slot = torch.tensor(-1)
            row_logits = logits[offset].contiguous()
            target_id = int(case["target_id"])
            competitor_id, competitor_logit = strongest_competitor(
                row_logits, target_id
            )
            target_logit = float(row_logits[target_id])
            margin = float(
                torch.tensor(target_logit, dtype=torch.float32)
                - torch.tensor(competitor_logit, dtype=torch.float32)
            )
            delta = audit.delta[local].detach().cpu().contiguous()
            native = native_hidden[local].detach().cpu().contiguous()
            changed = adapted[local].detach().cpu().contiguous()
            selected_slot = int(audit.selected_slots[local])
            gate_open = bool(audit.gate_open[local])
            prediction = int(row_logits.argmax())
            rows.append(
                {
                    "tier": len(indices),
                    "tier_offset": local,
                    "candidate_index": candidate_index,
                    "case_id": int(case["case_id"]),
                    "target_id": target_id,
                    "prompt_sha256": stable_json_sha256(case["prompt"]),
                    "native_hidden_sha256": tensor_sha256(native),
                    "selected_score": float(audit.selected_scores[local]),
                    "selected_slot": selected_slot,
                    "own_slot": own_slot,
                    "own_slot_score": float(active_scores[own_slot]),
                    "own_slot_attention": float(audit.attention[local, own_slot]),
                    "strongest_nonown_score": (
                        None
                        if not math.isfinite(float(strongest_nonown_score))
                        else float(strongest_nonown_score)
                    ),
                    "strongest_nonown_slot": int(strongest_nonown_slot),
                    "gate_open": gate_open,
                    "delta_sha256": tensor_sha256(delta),
                    "delta_zero": int(torch.count_nonzero(delta)) == 0,
                    "adapted_hidden_sha256": tensor_sha256(changed),
                    "target_logit": target_logit,
                    "competitor_id": competitor_id,
                    "competitor_logit": competitor_logit,
                    "target_margin": margin,
                    "prediction": prediction,
                    "strict_target_success": prediction == target_id and margin > 0.0,
                    "logits_sha256": tensor_sha256(row_logits.detach().cpu()),
                    "head_batch_start": start,
                    "head_batch_size": end - start,
                    "head_batch_offset": offset,
                }
            )
    attention = [row["own_slot_attention"] for row in rows]
    nonown_scores = [
        row["strongest_nonown_score"]
        for row in rows
        if row["strongest_nonown_score"] is not None
    ]
    summary = {
        "positive_count": len(rows),
        "gate_open_count": sum(row["gate_open"] for row in rows),
        "own_slot_count": sum(row["selected_slot"] == row["own_slot"] for row in rows),
        "strict_target_success_count": sum(row["strict_target_success"] for row in rows),
        "nonzero_delta_count": sum(not row["delta_zero"] for row in rows),
        "minimum_own_slot_attention": min(attention),
        "median_own_slot_attention": linear_median(attention),
        "minimum_target_margin": min(row["target_margin"] for row in rows),
        "maximum_nonown_score": max(nonown_scores) if nonown_scores else None,
        "key_coherence": memory.coherence(),
        "degree_five_max_crosstalk": memory.max_crosstalk(),
    }
    return {"rows": rows, "summary": summary}


def _threshold_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        f"{threshold:.2f}": sum(
            float(row["selected_score"]) >= threshold for row in rows
        )
        for threshold in SCORE_THRESHOLDS
    }


def _locality_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "pool_index",
            "case_id",
            "kind",
            "prompt_sha256",
            "selected_score",
            "selected_slot",
            "selected_candidate_index",
            "selected_case_id",
            "selected_label",
            "gate_open",
        )
    }


def summarize_locality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("E10 locality scan is empty")
    ranked = sorted(rows, key=lambda row: (-row["selected_score"], row["pool_index"]))
    by_kind: dict[str, Any] = {}
    for kind in KINDS:
        subset = [row for row in rows if row["kind"] == kind]
        maximum = sorted(
            subset, key=lambda row: (-row["selected_score"], row["pool_index"])
        )[0]
        by_kind[kind] = {
            "query_count": len(subset),
            "gate_open_count": sum(row["gate_open"] for row in subset),
            "delta_zero_count": sum(row["delta_zero"] for row in subset),
            "hidden_bit_identical_count": sum(
                row["hidden_bit_identical"] for row in subset
            ),
            "score_threshold_counts": _threshold_counts(subset),
            "maximum": _locality_projection(maximum),
        }
    return {
        "query_count": len(rows),
        "gate_open_count": sum(row["gate_open"] for row in rows),
        "delta_zero_count": sum(row["delta_zero"] for row in rows),
        "hidden_bit_identical_count": sum(row["hidden_bit_identical"] for row in rows),
        "score_threshold_counts": _threshold_counts(rows),
        "maximum": _locality_projection(ranked[0]),
        "top_20": [_locality_projection(row) for row in ranked[:TOP_K]],
        "by_kind": by_kind,
    }


@torch.inference_mode()
def scan_locality(
    adapter: HLM5PreHeadAdapter,
    memory: EditableHLM5Memory,
    selected_indices: list[int],
    cases: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    hidden_native: torch.Tensor,
) -> dict[str, Any]:
    specs = query_specs(query_rows)
    if hidden_native.shape[0] != len(specs) or hidden_native.dtype != torch.bfloat16:
        raise ValueError("E10 locality hidden contract mismatch")
    rows: list[dict[str, Any]] = []
    active_count = int(memory.active.sum())
    for start in range(0, len(specs), GATE_BATCH):
        end = min(start + GATE_BATCH, len(specs))
        chunk = hidden_native[start:end]
        adapted, audit = adapter(chunk, return_audit=True)
        if audit is None:
            raise RuntimeError("E10 locality scan omitted audit")
        for offset, spec in enumerate(specs[start:end]):
            index = start + offset
            slot = int(audit.selected_slots[offset])
            candidate_index = selected_indices[slot]
            score = float(audit.selected_scores[offset])
            if not math.isfinite(score):
                raise RuntimeError(f"E10 nonfinite locality score at {index}")
            delta = audit.delta[offset].detach().cpu().contiguous()
            native = chunk[offset].detach().cpu().contiguous()
            changed = adapted[offset].detach().cpu().contiguous()
            rows.append(
                {
                    **spec,
                    "tier": active_count,
                    "prompt_sha256": stable_json_sha256(spec["prompt"]),
                    "native_hidden_sha256": tensor_sha256(native),
                    "selected_score": score,
                    "selected_slot": slot,
                    "selected_candidate_index": candidate_index,
                    "selected_case_id": int(cases[candidate_index]["case_id"]),
                    "selected_label": memory.labels[slot],
                    "gate_open": bool(audit.gate_open[offset]),
                    "delta_sha256": tensor_sha256(delta),
                    "delta_zero": int(torch.count_nonzero(delta)) == 0,
                    "adapted_hidden_sha256": tensor_sha256(changed),
                    "hidden_bit_identical": torch.equal(native, changed),
                    "gate_batch_start": start,
                    "gate_batch_size": end - start,
                    "gate_batch_offset": offset,
                }
            )
    return {"rows": rows, "summary": summarize_locality(rows)}


@torch.inference_mode()
def rollback_scan(
    adapter: HLM5PreHeadAdapter,
    memory: EditableHLM5Memory,
    positive_hidden: torch.Tensor,
    locality_hidden: torch.Tensor,
) -> dict[str, Any]:
    active = torch.nonzero(memory.active, as_tuple=False).flatten().tolist()
    for slot in active:
        memory.remove(int(slot))
    counts = {
        "positive_count": int(positive_hidden.shape[0]),
        "positive_delta_zero_count": 0,
        "positive_hidden_bit_identical_count": 0,
        "locality_count": int(locality_hidden.shape[0]),
        "locality_delta_zero_count": 0,
        "locality_hidden_bit_identical_count": 0,
    }
    projection: list[dict[str, Any]] = []
    for kind, hidden, batch in (
        ("positive", positive_hidden, GATE_BATCH),
        ("locality", locality_hidden, GATE_BATCH),
    ):
        for start in range(0, hidden.shape[0], batch):
            end = min(start + batch, hidden.shape[0])
            native = hidden[start:end]
            adapted, audit = adapter(native, return_audit=True)
            if audit is None:
                raise RuntimeError("E10 rollback omitted audit")
            delta_zero = int(torch.count_nonzero(audit.delta)) == 0
            identical = torch.equal(native, adapted)
            counts[f"{kind}_delta_zero_count"] += (end - start) if delta_zero else 0
            counts[f"{kind}_hidden_bit_identical_count"] += (
                end - start if identical else 0
            )
            projection.append(
                {
                    "kind": kind,
                    "start": start,
                    "size": end - start,
                    "delta_sha256": tensor_sha256(audit.delta.detach().cpu().contiguous()),
                    "adapted_sha256": tensor_sha256(adapted.detach().cpu().contiguous()),
                    "delta_zero": delta_zero,
                    "hidden_bit_identical": identical,
                }
            )
    return {
        "removed_slots": active,
        "post_active_count": int(memory.active.sum()),
        "post_active_mask_sha256": tensor_sha256(memory.active),
        "post_alphas_sha256": tensor_sha256(memory.alphas.contiguous()),
        "post_labels": list(memory.labels),
        "batch_projection": projection,
        "summary": counts,
    }


def bundle_payload(
    memory: EditableHLM5Memory,
    key_mean: torch.Tensor,
    key_transform: torch.Tensor,
    selected_indices: list[int],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    active = torch.nonzero(memory.active, as_tuple=False).flatten()
    if int(active.numel()) != CAPACITY_TIERS[-1]:
        raise ValueError("E10 portable bundle requires the 1024 tier")
    return {
        "keys": memory.keys[active].detach().cpu().contiguous(),
        "values": memory.values[active].detach().cpu().contiguous(),
        "alphas": memory.alphas[active].detach().cpu().contiguous(),
        "key_mean": key_mean.detach().cpu().contiguous(),
        "key_transform": key_transform.detach().cpu().contiguous(),
        "selected_indices": torch.tensor(selected_indices[:1024], dtype=torch.int64),
        "selected_case_ids": torch.tensor(
            [cases[index]["case_id"] for index in selected_indices[:1024]],
            dtype=torch.int64,
        ),
    }


def bundle_manifest(payload: dict[str, Any], labels: list[str | None]) -> dict[str, Any]:
    tensors = {
        name: {
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
            "sha256": tensor_sha256(value),
        }
        for name, value in payload.items()
    }
    return {"tensors": tensors, "active_labels": labels}


def atomic_create_bundle(path: Path, payload: dict[str, Any]) -> str:
    """Atomically publish a create-new torch bundle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temp_path = Path(temporary)
    try:
        torch.save(payload, temp_path)
        os.link(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return sha256_file(path)


def load_bundle(path: Path, manifest: dict[str, Any]) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or set(value) != set(manifest["tensors"]):
        raise RuntimeError("E10 bundle members differ from receipt")
    for name, tensor in value.items():
        expected = manifest["tensors"][name]
        actual = {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "sha256": tensor_sha256(tensor),
        }
        if actual != expected:
            raise RuntimeError(f"E10 bundle tensor drift: {name}")
    return value


@torch.inference_mode()
def capacity_measurement(
    model: Any,
    cases: list[dict[str, Any]],
    exact_hidden: torch.Tensor,
    whitening_hidden: torch.Tensor,
    query_rows: list[dict[str, Any]],
    locality_hidden: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run all constructible registered tiers and return a portable 1024 bundle."""
    direct = direct_measurement(model, cases, exact_hidden)
    selected = direct["direct_summary"]["selected_indices"]
    key_mean, key_transform, key_record = shared_key_operator(
        exact_hidden, whitening_hidden
    )
    directions = direct.pop("directions")
    tiers: dict[str, Any] = {}
    portable: dict[str, Any] | None = None
    portable_manifest: dict[str, Any] | None = None
    for tier in CAPACITY_TIERS:
        if len(selected) < tier:
            continue
        memory, adapter, slot_by_candidate, memory_record = build_capacity_memory(
            cases,
            exact_hidden,
            direct["direct_rows"],
            directions,
            selected,
            key_mean,
            key_transform,
            tier,
        )
        positives = scan_positives(
            model,
            adapter,
            memory,
            cases,
            exact_hidden,
            selected,
            slot_by_candidate,
        )
        locality = scan_locality(
            adapter, memory, selected, cases, query_rows, locality_hidden
        )
        if tier == CAPACITY_TIERS[-1]:
            portable = bundle_payload(
                memory, key_mean, key_transform, selected, cases
            )
            portable_manifest = bundle_manifest(
                portable, [memory.labels[index] for index in range(tier)]
            )
        rollback = rollback_scan(
            adapter,
            memory,
            exact_hidden[selected[:tier]],
            locality_hidden,
        )
        tiers[str(tier)] = {
            "memory_receipt": memory_record,
            "positive_scan": positives,
            "locality_scan": locality,
            "rollback": rollback,
        }
    measurement = {
        "population": population_record(cases),
        "direct": direct,
        "key_whitening": key_record,
        "tiers": tiers,
        "bundle": (
            None
            if portable_manifest is None
            else {**portable_manifest, "file_sha256": None}
        ),
    }
    return measurement, portable


__all__ = [
    "BOOST", "CANDIDATE_COUNT", "CANDIDATE_START", "CAPACITY_TIERS",
    "DEGREE", "GATE_BATCH", "GATE_THRESHOLD", "KEY_FLOOR_FRACTION",
    "MEMORY_SIZE", "MODEL_FORWARD_BATCH", "TEMPERATURE", "TOP_K",
    "atomic_create_bundle", "build_capacity_memory", "bundle_manifest",
    "bundle_payload", "capacity_measurement", "direct_measurement", "e10_environment_record",
    "linear_median", "load_bundle", "load_capacity_cases",
    "population_record", "rollback_scan", "scan_locality", "scan_positives",
    "select_capacity_cases", "shared_key_operator", "summarize_locality",
]

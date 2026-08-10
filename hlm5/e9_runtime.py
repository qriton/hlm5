"""Wide-query selection and exact-key locality runtime for E9."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from .certify import calibrate_whitening
from .e7_contract import stable_json_sha256, tensor_sha256
from .e7c_runtime import raw_and_zca_directions
from .e8_runtime import GATE_THRESHOLD, build_memory
from .public_adapter import HLM5PreHeadAdapter


QUERY_CASE_START = 10_000
QUERY_CASE_STOP = 20_000
QUERY_CASE_COUNT = 4_096
QUERY_COUNT = 12_288
MODEL_FORWARD_BATCH = 16
GATE_BATCH = 256
TOP_K = 20
SCORE_THRESHOLDS = (0.10, 0.50, 0.90, 0.95)
KINDS = ("exact", "paraphrase", "neighborhood")


def select_query_rows(
    records: list[dict[str, Any]],
    excluded_prompts: list[str],
) -> list[dict[str, Any]]:
    """Select the deterministic disjoint E9 query population."""
    excluded = set(excluded_prompts)
    used: set[str] = set()
    rows: list[dict[str, Any]] = []
    for record in records:
        case_id = int(record["case_id"])
        if not QUERY_CASE_START <= case_id < QUERY_CASE_STOP:
            continue
        rewrite = record["requested_rewrite"]
        exact = str(rewrite["prompt"]).format(rewrite["subject"])
        paraphrases = list(
            dict.fromkeys(
                str(value)
                for value in record.get("paraphrase_prompts", [])
                if str(value)
            )
        )
        neighborhoods = list(
            dict.fromkeys(
                str(value)
                for value in record.get("neighborhood_prompts", [])
                if str(value)
            )
        )
        if not exact or not paraphrases or not neighborhoods:
            continue
        prompts = [exact, paraphrases[0], neighborhoods[0]]
        if len(set(prompts)) != len(KINDS):
            continue
        if any(prompt in excluded or prompt in used for prompt in prompts):
            continue
        rows.append(
            {
                "case_id": case_id,
                "exact": prompts[0],
                "paraphrase": prompts[1],
                "neighborhood": prompts[2],
            }
        )
        used.update(prompts)
        if len(rows) == QUERY_CASE_COUNT:
            break
    if len(rows) != QUERY_CASE_COUNT:
        raise RuntimeError(
            f"E9 selected {len(rows)} query cases, expected {QUERY_CASE_COUNT}"
        )
    return rows


def load_query_rows(
    data_path: Path,
    excluded_prompts: list[str],
) -> list[dict[str, Any]]:
    records = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError("CounterFact data must be a JSON list")
    return select_query_rows(records, excluded_prompts)


def query_prompt_order(rows: list[dict[str, Any]]) -> list[str]:
    prompts = [row[kind] for row in rows for kind in KINDS]
    if len(prompts) != QUERY_COUNT or len(set(prompts)) != QUERY_COUNT:
        raise RuntimeError("E9 query prompt order is not exactly unique")
    return prompts


def query_specs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for case_index, row in enumerate(rows):
        for kind in KINDS:
            specs.append(
                {
                    "pool_index": len(specs),
                    "case_index": case_index,
                    "case_id": int(row["case_id"]),
                    "kind": kind,
                    "prompt": row[kind],
                }
            )
    if len(specs) != QUERY_COUNT:
        raise RuntimeError("E9 query spec count mismatch")
    return specs


@torch.inference_mode()
def reconstruct_e8_memory(
    e8_cases: list[dict[str, Any]],
    hidden_by_prompt: dict[str, torch.Tensor],
    whitening_prompts: list[str],
    direct_rows: list[dict[str, Any]],
    candidate_directions: torch.Tensor,
) -> tuple[
    Any,
    HLM5PreHeadAdapter,
    dict[int, int],
    dict[str, Any],
    dict[str, Any],
]:
    """Reconstruct the exact E8 primary memory from frozen ingredients."""
    exact_hidden = torch.stack([hidden_by_prompt[case["prompt"]] for case in e8_cases])
    whitening_hidden = torch.stack(
        [hidden_by_prompt[prompt] for prompt in whitening_prompts]
    )
    key_mean, key_transform = calibrate_whitening(
        exact_hidden.float(), whitening_hidden.float(), floor_frac=0.01
    )
    memory, slot_by_case, receipt = build_memory(
        e8_cases,
        exact_hidden,
        direct_rows,
        candidate_directions,
        key_mean,
        key_transform,
    )
    adapter = HLM5PreHeadAdapter(
        memory,
        key_mean,
        key_transform,
        boost=1.0,
        gate_thresh=GATE_THRESHOLD,
    )
    key_record = {
        "mean_sha256": tensor_sha256(key_mean),
        "transform_sha256": tensor_sha256(key_transform),
        "mean_dtype": str(key_mean.dtype).removeprefix("torch."),
        "mean_shape": list(key_mean.shape),
        "transform_dtype": str(key_transform.dtype).removeprefix("torch."),
        "transform_shape": list(key_transform.shape),
        "floor_fraction": 0.01,
        "exact_key_count": len(e8_cases),
        "whitening_prompt_count": len(whitening_prompts),
    }
    return memory, adapter, slot_by_case, receipt, key_record


@torch.inference_mode()
def prepare_e8_adapter(
    model: Any,
    e8_population: dict[str, Any],
    frozen_measurement: dict[str, Any],
    hidden_by_prompt: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Build and audit the frozen E8 adapter from spent evidence."""
    cases = e8_population["cases"]
    target_ids = torch.tensor(
        [case["target_id"] for case in cases],
        device=model.get_output_embeddings().weight.device,
        dtype=torch.long,
    )
    _raw, candidate, basis = raw_and_zca_directions(
        model.get_output_embeddings().weight.detach(), target_ids
    )
    memory, adapter, slot_by_case, receipt, key_record = reconstruct_e8_memory(
        cases,
        hidden_by_prompt,
        e8_population["whitening_prompts"],
        frozen_measurement["arms"]["candidate"]["direct_rows"],
        candidate,
    )
    anchors = scan_anchor_keys(adapter, cases, hidden_by_prompt, slot_by_case)
    return {
        "memory": memory,
        "adapter": adapter,
        "slot_by_case": slot_by_case,
        "basis_and_directions": basis,
        "memory_receipt": receipt,
        "key_whitening": key_record,
        "anchor_scan": anchors,
    }


@torch.inference_mode()
def scan_anchor_keys(
    adapter: HLM5PreHeadAdapter,
    e8_cases: list[dict[str, Any]],
    hidden_by_prompt: dict[str, torch.Tensor],
    slot_by_case: dict[int, int],
) -> dict[str, Any]:
    """Evaluate the 64 spent E8 exact keys as a non-vacuity control."""
    hidden = torch.stack([hidden_by_prompt[case["prompt"]] for case in e8_cases])
    adapted, audit = adapter(hidden, return_audit=True)
    if audit is None:
        raise RuntimeError("E9 anchor scan omitted adapter audit")
    scores = audit.selected_scores.detach().cpu()
    slots = audit.selected_slots.detach().cpu()
    gates = audit.gate_open.detach().cpu()
    deltas = audit.delta.detach().cpu()
    native = hidden.detach().cpu()
    changed = adapted.detach().cpu()
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(e8_cases):
        own_slot = slot_by_case[index]
        rows.append(
            {
                "case_index": index,
                "case_id": int(case["case_id"]),
                "prompt_sha256": stable_json_sha256(case["prompt"]),
                "actual_score": float(scores[index]),
                "selected_slot": int(slots[index]),
                "own_slot": own_slot,
                "gate_threshold": GATE_THRESHOLD,
                "gate_open": bool(gates[index]),
                "delta_sha256": tensor_sha256(deltas[index].contiguous()),
                "delta_zero": int(torch.count_nonzero(deltas[index])) == 0,
                "native_hidden_sha256": tensor_sha256(native[index].contiguous()),
                "adapted_hidden_sha256": tensor_sha256(changed[index].contiguous()),
                "hidden_bit_identical": torch.equal(native[index], changed[index]),
            }
        )
    return {
        "rows": rows,
        "summary": {
            "anchor_count": len(rows),
            "gate_open_count": sum(row["gate_open"] for row in rows),
            "own_slot_count": sum(
                row["selected_slot"] == row["own_slot"] for row in rows
            ),
            "nonzero_delta_count": sum(not row["delta_zero"] for row in rows),
        },
    }


def _threshold_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        f"{threshold:.2f}": sum(
            float(row["selected_score"]) >= threshold for row in rows
        )
        for threshold in SCORE_THRESHOLDS
    }


def _diagnostic_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "pool_index": row["pool_index"],
        "case_id": row["case_id"],
        "kind": row["kind"],
        "prompt_sha256": row["prompt_sha256"],
        "selected_score": row["selected_score"],
        "selected_slot": row["selected_slot"],
        "selected_e8_case_id": row["selected_e8_case_id"],
        "selected_label": row["selected_label"],
        "gate_open": row["gate_open"],
    }


def summarize_query_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("E9 cannot summarize an empty query scan")
    ranked = sorted(
        rows, key=lambda row: (-float(row["selected_score"]), row["pool_index"])
    )
    kind_summary: dict[str, Any] = {}
    for kind in KINDS:
        subset = [row for row in rows if row["kind"] == kind]
        maximum = sorted(
            subset,
            key=lambda row: (-float(row["selected_score"]), row["pool_index"]),
        )[0]
        kind_summary[kind] = {
            "query_count": len(subset),
            "gate_open_count": sum(row["gate_open"] for row in subset),
            "delta_zero_count": sum(row["delta_zero"] for row in subset),
            "hidden_bit_identical_count": sum(
                row["hidden_bit_identical"] for row in subset
            ),
            "score_threshold_counts": _threshold_counts(subset),
            "maximum": _diagnostic_projection(maximum),
        }
    return {
        "query_count": len(rows),
        "gate_open_count": sum(row["gate_open"] for row in rows),
        "delta_zero_count": sum(row["delta_zero"] for row in rows),
        "hidden_bit_identical_count": sum(row["hidden_bit_identical"] for row in rows),
        "score_threshold_counts": _threshold_counts(rows),
        "maximum": _diagnostic_projection(ranked[0]),
        "top_20": [_diagnostic_projection(row) for row in ranked[:TOP_K]],
        "by_kind": kind_summary,
    }


@torch.inference_mode()
def scan_queries(
    adapter: HLM5PreHeadAdapter,
    memory: Any,
    e8_cases: list[dict[str, Any]],
    slot_by_case: dict[int, int],
    query_rows: list[dict[str, Any]],
    hidden_native: torch.Tensor,
) -> dict[str, Any]:
    """Evaluate every E9 query under the frozen 256-row gate schedule."""
    specs = query_specs(query_rows)
    if hidden_native.dtype != torch.bfloat16 or hidden_native.shape != (
        QUERY_COUNT,
        memory.dim,
    ):
        raise ValueError("E9 query hidden tensor contract mismatch")
    case_by_slot = {
        slot: e8_cases[case_index] for case_index, slot in slot_by_case.items()
    }
    rows: list[dict[str, Any]] = []
    for start in range(0, QUERY_COUNT, GATE_BATCH):
        end = min(start + GATE_BATCH, QUERY_COUNT)
        native_chunk = hidden_native[start:end]
        adapted, audit = adapter(native_chunk, return_audit=True)
        if audit is None:
            raise RuntimeError("E9 query scan omitted adapter audit")
        scores = audit.selected_scores.detach().cpu()
        slots = audit.selected_slots.detach().cpu()
        gates = audit.gate_open.detach().cpu()
        deltas = audit.delta.detach().cpu()
        native = native_chunk.detach().cpu()
        changed = adapted.detach().cpu()
        for offset, spec in enumerate(specs[start:end]):
            index = start + offset
            score = float(scores[offset])
            selected_slot = int(slots[offset])
            selected_case = case_by_slot[selected_slot]
            gate_open = bool(gates[offset])
            delta = deltas[offset].contiguous()
            native_row = native[offset].contiguous()
            changed_row = changed[offset].contiguous()
            if not math.isfinite(score):
                raise RuntimeError(f"E9 selected score is nonfinite at row {index}")
            rows.append(
                {
                    **spec,
                    "prompt_sha256": stable_json_sha256(spec["prompt"]),
                    "native_hidden_dtype": "bfloat16",
                    "native_hidden_shape": list(native_row.shape),
                    "native_hidden_sha256": tensor_sha256(native_row),
                    "selected_score": score,
                    "selected_slot": selected_slot,
                    "selected_e8_case_id": int(selected_case["case_id"]),
                    "selected_label": memory.labels[selected_slot],
                    "gate_threshold": GATE_THRESHOLD,
                    "gate_open": gate_open,
                    "gate_batch_start": start,
                    "gate_batch_size": end - start,
                    "gate_batch_offset": offset,
                    "delta_dtype": "bfloat16",
                    "delta_shape": list(delta.shape),
                    "delta_sha256": tensor_sha256(delta),
                    "delta_zero": int(torch.count_nonzero(delta)) == 0,
                    "adapted_hidden_sha256": tensor_sha256(changed_row),
                    "hidden_bit_identical": torch.equal(native_row, changed_row),
                }
            )
    if len(rows) != QUERY_COUNT:
        raise RuntimeError("E9 did not evaluate every query")
    return {"rows": rows, "summary": summarize_query_scan(rows)}


__all__ = [
    "GATE_BATCH",
    "KINDS",
    "MODEL_FORWARD_BATCH",
    "QUERY_CASE_COUNT",
    "QUERY_CASE_START",
    "QUERY_CASE_STOP",
    "QUERY_COUNT",
    "SCORE_THRESHOLDS",
    "TOP_K",
    "load_query_rows",
    "prepare_e8_adapter",
    "query_prompt_order",
    "query_specs",
    "reconstruct_e8_memory",
    "scan_anchor_keys",
    "scan_queries",
    "select_query_rows",
    "summarize_query_scan",
]

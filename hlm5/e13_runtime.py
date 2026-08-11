"""Terminal paired readout-conditioning runtime for HLM5 E13."""

from __future__ import annotations

import platform
from typing import Any, Iterable

import torch

from .e7_contract import stable_json_sha256, tensor_sha256
from .e7b_runtime import environment_record as base_environment_record
from .e7c_runtime import basis_record, normalize_rows, zca_basis
from .e8_runtime import (
    baseline_case_rows,
    direct_admission_rows,
    direct_summary,
    paired_geometry,
)
from .e9_runtime import KINDS, query_prompt_order
from .e10_runtime import (
    BOOST,
    CANDIDATE_COUNT,
    DEGREE,
    GATE_BATCH,
    GATE_THRESHOLD,
    KEY_FLOOR_FRACTION,
    MEMORY_SIZE,
    MODEL_FORWARD_BATCH,
    TEMPERATURE,
    TOP_K,
    atomic_create_bundle,
    build_capacity_memory,
    bundle_manifest,
    bundle_payload,
    linear_median,
    load_bundle,
    population_record,
    rollback_scan,
    scan_locality,
    scan_positives,
    shared_key_operator,
    summarize_locality,
)
from .e12_runtime import load_records, select_final_cases


BASELINE = "zca_half_raw_r1e2"
CANDIDATE = "mahalanobis_raw_r1e2"
BASELINE_TIERS = (1_024,)
CANDIDATE_TIERS = (256, 1_024)
FINAL_QUERY_CASE_START = 2_508
FINAL_QUERY_CASE_COUNT = 4_096
FINAL_QUERY_COUNT = FINAL_QUERY_CASE_COUNT * len(KINDS)
PRIOR_NODES = (
    "lrdn1455.leonardo.local",
    "lrdn2661.leonardo.local",
    "lrdn2974.leonardo.local",
)


def e13_environment_record(device: torch.device) -> dict[str, Any]:
    value = base_environment_record(device)
    value["node"] = platform.node()
    return value


def select_terminal_cases(
    records: list[dict[str, Any]],
    tokenizer: Any,
    excluded_prompts: Iterable[str],
) -> list[dict[str, Any]]:
    return select_final_cases(records, tokenizer, excluded_prompts)


def select_terminal_query_rows(
    records: list[dict[str, Any]],
    excluded_prompts: Iterable[str],
) -> list[dict[str, Any]]:
    excluded = set(excluded_prompts)
    used: set[str] = set()
    rows: list[dict[str, Any]] = []
    for record in records:
        case_id = int(record["case_id"])
        if case_id < FINAL_QUERY_CASE_START:
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
        if len(rows) == FINAL_QUERY_CASE_COUNT:
            break
    if len(rows) != FINAL_QUERY_CASE_COUNT:
        raise RuntimeError(
            f"E13 selected {len(rows)} locality cases, expected {FINAL_QUERY_CASE_COUNT}"
        )
    return rows


@torch.inference_mode()
def paired_directions(
    head_native: torch.Tensor,
    target_ids: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Construct the frozen half-ZCA and full-Mahalanobis value directions."""
    head64 = head_native.to(torch.float64)
    mean32, eigenvalues32, eigenvectors32, half_operator64, ridge = zca_basis(
        head_native
    )
    eigenvectors64 = eigenvectors32.to(torch.float64)
    spectrum64 = eigenvalues32.to(torch.float64) + ridge
    source = head64[target_ids]
    half = normalize_rows(
        (source @ eigenvectors64 * spectrum64.rsqrt()) @ eigenvectors64.T
    )
    full_operator64 = (eigenvectors64 / spectrum64) @ eigenvectors64.T
    full = normalize_rows((source @ eigenvectors64 / spectrum64) @ eigenvectors64.T)
    if not torch.isfinite(half).all() or not torch.isfinite(full).all():
        raise RuntimeError("E13 direction construction produced nonfinite values")
    record = basis_record(mean32, eigenvalues32, eigenvectors32, half_operator64, ridge)
    record.update(
        {
            "baseline_name": BASELINE,
            "candidate_name": CANDIDATE,
            "baseline_directions_sha256": tensor_sha256(half),
            "candidate_directions_sha256": tensor_sha256(full),
            "candidate_operator_sha256": tensor_sha256(full_operator64),
            "direction_dtype": "float64",
            "direction_shape": list(half.shape),
            "candidate_power": 1.0,
        }
    )
    return {BASELINE: half, CANDIDATE: full}, record


@torch.inference_mode()
def paired_direct_measurement(
    model: Any,
    cases: list[dict[str, Any]],
    exact_hidden: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], list[int]]:
    target_ids = torch.tensor(
        [row["target_id"] for row in cases],
        device=exact_hidden.device,
        dtype=torch.long,
    )
    head = model.get_output_embeddings().weight.detach()
    directions, basis = paired_directions(head, target_ids)
    baseline_rows = baseline_case_rows(model, exact_hidden, cases)
    arms: dict[str, Any] = {}
    for name in (BASELINE, CANDIDATE):
        geometry = paired_geometry(
            head.to(torch.float64), exact_hidden, target_ids, directions[name]
        )
        rows = direct_admission_rows(
            model, exact_hidden, directions[name], geometry, baseline_rows
        )
        arms[name] = {
            "direct_rows": rows,
            "direct_summary": direct_summary(rows),
        }
    selected = [
        index
        for index in range(len(cases))
        if all(
            arms[name]["direct_rows"][index]["deployment_admitted"]
            for name in (BASELINE, CANDIDATE)
        )
    ][:1_024]
    case_ids = [int(cases[index]["case_id"]) for index in selected]
    common = {
        "common_admitted_count": sum(
            all(
                arms[name]["direct_rows"][index]["deployment_admitted"]
                for name in (BASELINE, CANDIDATE)
            )
            for index in range(len(cases))
        ),
        "selected_count": len(selected),
        "selected_indices": selected,
        "selected_case_ids": case_ids,
        "selected_indices_sha256": stable_json_sha256(selected),
        "selected_case_ids_sha256": stable_json_sha256(case_ids),
    }
    return (
        {
            "basis_and_directions": basis,
            "baseline_rows": baseline_rows,
            "arms": arms,
            "common_selection": common,
        },
        directions,
        selected,
    )


@torch.inference_mode()
def terminal_measurement(
    model: Any,
    cases: list[dict[str, Any]],
    exact_hidden: torch.Tensor,
    whitening_hidden: torch.Tensor,
    query_rows: list[dict[str, Any]],
    locality_hidden: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    direct, directions, selected = paired_direct_measurement(model, cases, exact_hidden)
    key_mean, key_transform, key_record = shared_key_operator(
        exact_hidden, whitening_hidden
    )
    arms: dict[str, Any] = {BASELINE: {}, CANDIDATE: {}}
    portable: dict[str, Any] | None = None
    portable_manifest: dict[str, Any] | None = None
    for name, tiers in (
        (BASELINE, BASELINE_TIERS),
        (CANDIDATE, CANDIDATE_TIERS),
    ):
        direct_rows = direct["arms"][name]["direct_rows"]
        for tier in tiers:
            if len(selected) < tier:
                continue
            memory, adapter, slots, memory_record = build_capacity_memory(
                cases,
                exact_hidden,
                direct_rows,
                directions[name],
                selected,
                key_mean,
                key_transform,
                tier,
            )
            positives = scan_positives(
                model, adapter, memory, cases, exact_hidden, selected, slots
            )
            locality = scan_locality(
                adapter, memory, selected, cases, query_rows, locality_hidden
            )
            if name == CANDIDATE and tier == 1_024:
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
            arms[name][str(tier)] = {
                "memory_receipt": memory_record,
                "positive_scan": positives,
                "locality_scan": locality,
                "rollback": rollback,
            }
    return {
        "population": population_record(cases),
        "direct": direct,
        "key_whitening": key_record,
        "arms": arms,
        "bundle": (
            None
            if portable_manifest is None
            else {**portable_manifest, "file_sha256": None}
        ),
    }, portable


__all__ = [
    "BASELINE",
    "BASELINE_TIERS",
    "BOOST",
    "CANDIDATE",
    "CANDIDATE_COUNT",
    "CANDIDATE_TIERS",
    "DEGREE",
    "FINAL_QUERY_CASE_COUNT",
    "FINAL_QUERY_CASE_START",
    "FINAL_QUERY_COUNT",
    "GATE_BATCH",
    "GATE_THRESHOLD",
    "KEY_FLOOR_FRACTION",
    "MEMORY_SIZE",
    "MODEL_FORWARD_BATCH",
    "PRIOR_NODES",
    "TEMPERATURE",
    "TOP_K",
    "atomic_create_bundle",
    "e13_environment_record",
    "linear_median",
    "load_bundle",
    "load_records",
    "paired_directions",
    "population_record",
    "query_prompt_order",
    "select_terminal_cases",
    "select_terminal_query_rows",
    "summarize_locality",
    "terminal_measurement",
]

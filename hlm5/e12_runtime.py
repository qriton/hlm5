"""Untouched-population selection and E10 measurement reuse for HLM5 E12."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any, Iterable

import torch

from .e7b_runtime import environment_record as base_environment_record
from .e9_runtime import KINDS, query_prompt_order, query_specs
from .e10_runtime import (
    BOOST,
    CAPACITY_TIERS,
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
    capacity_measurement,
    linear_median,
    load_bundle,
    population_record,
    summarize_locality,
)


FINAL_CASE_START = 0
FINAL_QUERY_CASE_START = 4_000
FINAL_QUERY_CASE_COUNT = 4_096
FINAL_QUERY_COUNT = FINAL_QUERY_CASE_COUNT * len(KINDS)
PRIOR_NODES = (
    "lrdn1455.leonardo.local",
    "lrdn2661.leonardo.local",
)


def e12_environment_record(device: torch.device) -> dict[str, Any]:
    """Extend the frozen arithmetic environment with the physical node."""
    value = base_environment_record(device)
    value["node"] = platform.node()
    return value


def select_final_cases(
    records: list[dict[str, Any]],
    tokenizer: Any,
    excluded_prompts: Iterable[str],
) -> list[dict[str, Any]]:
    """Select the frozen E12 positive population without model inference."""
    excluded = set(excluded_prompts)
    used: set[str] = set()
    rows: list[dict[str, Any]] = []
    for record in records:
        case_id = int(record["case_id"])
        if case_id < FINAL_CASE_START:
            continue
        rewrite = record["requested_rewrite"]
        prompt = str(rewrite["prompt"]).format(rewrite["subject"])
        target = " " + str(rewrite["target_new"]["str"]).strip()
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if len(target_ids) != 1 or not prompt or prompt in excluded or prompt in used:
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
            f"E12 selected {len(rows)} positive cases, expected {CANDIDATE_COUNT}"
        )
    return rows


def select_final_query_rows(
    records: list[dict[str, Any]],
    excluded_prompts: Iterable[str],
) -> list[dict[str, Any]]:
    """Select 4,096 disjoint exact/paraphrase/neighborhood locality cases."""
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
            f"E12 selected {len(rows)} locality cases, expected {FINAL_QUERY_CASE_COUNT}"
        )
    return rows


def load_records(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError("CounterFact data must be a JSON list")
    return records


__all__ = [
    "BOOST",
    "CAPACITY_TIERS",
    "CANDIDATE_COUNT",
    "DEGREE",
    "FINAL_CASE_START",
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
    "build_capacity_memory",
    "bundle_manifest",
    "capacity_measurement",
    "e12_environment_record",
    "linear_median",
    "load_bundle",
    "load_records",
    "population_record",
    "query_prompt_order",
    "query_specs",
    "select_final_cases",
    "select_final_query_rows",
    "summarize_locality",
]

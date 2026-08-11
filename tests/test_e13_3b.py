"""Fast contract tests for the terminal HLM5 E13 experiment."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

import hlm5.e13_contract as contract
from hlm5.e7_contract import snapshot_path, stable_json_sha256
from hlm5.e7_runtime import load_pinned_tokenizer
from hlm5.e7c_runtime import normalize_rows, zca_basis
from hlm5.e13_runtime import (
    BASELINE,
    CANDIDATE,
    CANDIDATE_TIERS,
    PRIOR_NODES,
    paired_directions,
    population_record,
    select_terminal_cases,
    select_terminal_query_rows,
)


def test_protocol_dependencies_and_bound_evidence_are_frozen() -> None:
    contract.verify_protocol()
    assert contract.verify_dependencies() == contract.DEPENDENCY_SHA256
    bound = contract.verify_bound_inputs()
    assert bound["e10_verdict"] == "PASS_3B_CAPACITY_1024"
    assert bound["e11_verdict"] == "PASS_3B_CROSS_NODE_PORTABLE"
    assert bound["e12_verdict"] == "FAIL_3B_UNTOUCHED_FINAL"


def test_registered_population_matches_frozen_hashes_and_is_disjoint() -> None:
    population = contract.selected_population(load_pinned_tokenizer(snapshot_path()))
    assert population["record"]["first_case_id"] == 1_247
    assert population["record"]["last_case_id"] == 2_507
    assert population["record"]["case_sha256"] == contract.CANDIDATE_CASE_SHA256
    assert population["query_record"]["query_row_sha256"] == contract.QUERY_ROW_SHA256
    assert len(population["cases"]) == 1_200
    assert len(population["query_prompt_order"]) == 12_288
    assert not set(population["prompt_order"]) & set(population["query_prompt_order"])


def test_environment_rejects_all_three_spent_nodes() -> None:
    for node in PRIOR_NODES:
        value = {**contract.EXPECTED_ENVIRONMENT, "node": node}
        assert (
            "E13 physical node was already used by E10-E12"
            in contract.environment_failures(value)
        )


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [sum(text.encode("utf-8")) % 997]


def test_positive_selection_is_ordered_unique_and_excludes_prior() -> None:
    records = [
        {
            "case_id": index,
            "requested_rewrite": {
                "prompt": "{} fact " + str(index),
                "subject": "S",
                "target_new": {"str": f" target-{index}"},
                "relation_id": "P1",
            },
        }
        for index in range(1_247, 2_600)
    ]
    rows = select_terminal_cases(records, _Tokenizer(), {"S fact 1247"})
    assert len(rows) == 1_200
    assert rows[0]["case_id"] == 1_248
    assert rows[-1]["case_id"] == 2_447
    assert population_record(rows)["case_sha256"] == stable_json_sha256(rows)


def test_locality_selection_is_ordered_unique(monkeypatch: pytest.MonkeyPatch) -> None:
    import hlm5.e13_runtime as runtime

    monkeypatch.setattr(runtime, "FINAL_QUERY_CASE_COUNT", 2)
    records = [
        {
            "case_id": index,
            "requested_rewrite": {"prompt": "{} exact " + str(index), "subject": "S"},
            "paraphrase_prompts": [f"para {index}"],
            "neighborhood_prompts": [f"near {index}"],
        }
        for index in range(2_508, 2_512)
    ]
    rows = select_terminal_query_rows(records, {"S exact 2508"})
    assert [row["case_id"] for row in rows] == [2_509, 2_510]


def test_paired_directions_are_exact_frozen_half_and_full_covariance() -> None:
    generator = torch.Generator().manual_seed(17)
    head = torch.randn(31, 7, generator=generator)
    target_ids = torch.tensor([1, 8, 23])
    directions, receipt = paired_directions(head, target_ids)
    _mean, eigenvalues, eigenvectors, _operator, ridge = zca_basis(head)
    vectors = eigenvectors.to(torch.float64)
    spectrum = eigenvalues.to(torch.float64) + ridge
    source = head.to(torch.float64)[target_ids]
    expected_half = normalize_rows((source @ vectors * spectrum.rsqrt()) @ vectors.T)
    expected_full = normalize_rows((source @ vectors / spectrum) @ vectors.T)
    assert torch.equal(directions[BASELINE], expected_half)
    assert torch.equal(directions[CANDIDATE], expected_full)
    assert receipt["candidate_power"] == 1.0
    assert receipt["baseline_name"] == BASELINE
    assert receipt["candidate_name"] == CANDIDATE


def _tier(tier: int, strict: int | None = None) -> dict[str, object]:
    return {
        "positive_scan": {
            "summary": {
                "positive_count": tier,
                "gate_open_count": tier,
                "own_slot_count": tier,
                "strict_target_success_count": tier if strict is None else strict,
                "nonzero_delta_count": tier,
            }
        },
        "locality_scan": {
            "summary": {
                "query_count": 12_288,
                "gate_open_count": 0,
                "delta_zero_count": 12_288,
                "hidden_bit_identical_count": 12_288,
                "by_kind": {
                    kind: {"query_count": 4_096, "gate_open_count": 0}
                    for kind in ("exact", "paraphrase", "neighborhood")
                },
            }
        },
        "rollback": {
            "summary": {
                "positive_delta_zero_count": tier,
                "positive_hidden_bit_identical_count": tier,
                "locality_delta_zero_count": 12_288,
                "locality_hidden_bit_identical_count": 12_288,
            }
        },
    }


def _result(baseline_strict: int = 1_023) -> dict[str, object]:
    measurement = {
        "direct": {"common_selection": {"selected_count": 1_024}},
        "arms": {
            BASELINE: {"1024": _tier(1_024, baseline_strict)},
            CANDIDATE: {str(tier): _tier(tier) for tier in CANDIDATE_TIERS},
        },
        "bundle": {"file_sha256": "a" * 64},
    }
    return {"scientific": {"measurement": measurement}}


def test_scientific_pass_requires_both_candidate_tiers_and_not_baseline_strict() -> (
    None
):
    result = _result()
    assert contract.scientific_failures(result) == []
    assert contract.largest_passing_tier(result) == 1_024
    assert contract.baseline_strict_success_count(result) == 1_023
    broken = copy.deepcopy(result)
    del broken["scientific"]["measurement"]["arms"][CANDIDATE]["256"]
    assert "candidate tier 256 was not constructible" in contract.scientific_failures(
        broken
    )


def test_baseline_strict_miss_is_control_outcome_not_candidate_failure() -> None:
    result = _result(0)
    assert contract.scientific_failures(result) == []
    assert contract.baseline_strict_success_count(result) == 0
    result["scientific"]["measurement"]["arms"][BASELINE]["1024"]["positive_scan"][
        "summary"
    ]["own_slot_count"] = 1_023
    assert any(
        "own_slot_count" in value for value in contract.scientific_failures(result)
    )


def test_atomic_json_is_single_use_and_staging_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contract, "STAGING_DIR", tmp_path / "staging")
    path = contract.STAGING_DIR / "receipt.json"
    contract.atomic_create_json(path, {"x": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"x": 1}
    with pytest.raises(FileExistsError):
        contract.atomic_create_json(path, {"x": 2})
    with pytest.raises(ValueError):
        contract.atomic_create_json(tmp_path / "outside.json", {"x": 3})


def test_registered_source_manifest_covers_every_executable() -> None:
    paths = set(contract.REGISTERED_SOURCE_PATHS)
    assert "hlm5/e13_contract.py" in paths
    assert "hlm5/e13_runtime.py" in paths
    assert "tests/test_e13_3b.py" in paths
    assert {
        f"scripts/{name}_e13_3b.py"
        for name in ("preflight", "register", "admit", "replay", "verify")
    } <= paths

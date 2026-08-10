"""Fast CPU contract tests for the registered HLM5 E10 experiment."""

from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path

import pytest
import torch

import hlm5.e10_contract as contract
from hlm5.e7_contract import sha256_file, snapshot_path, stable_json_sha256
from hlm5.e7_runtime import load_pinned_tokenizer
from hlm5.e10_runtime import (
    CAPACITY_TIERS,
    MEMORY_SIZE,
    atomic_create_bundle,
    build_capacity_memory,
    bundle_manifest,
    linear_median,
    load_bundle,
    population_record,
    select_capacity_cases,
    summarize_locality,
)


def _script_globals(name: str) -> dict[str, object]:
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / name),
        run_name=f"e10_test_{name.replace('.', '_')}",
    )


def _patch_paths(
    globals_dict: dict[str, object],
    staging: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path]:
    paths = {
        "ATTEMPT_PATH": staging / "attempt.json",
        "BUNDLE_PATH": staging / "adapter_1024.pt",
        "ADMISSION_PATH": staging / "admission.json",
        "RESULT_PATH": staging / "result.json",
        "VERDICT_PATH": staging / "verdict.json",
    }
    monkeypatch.setattr(contract, "STAGING_DIR", staging)
    for name, path in paths.items():
        if name in globals_dict:
            monkeypatch.setitem(globals_dict, name, path)
    return paths


def _fake_receipt() -> dict[str, object]:
    return {
        "implementation_commit": "implementation",
        "native_runtime": dict(contract.EXPECTED_NATIVE_RUNTIME),
        "environment": {"registered": "environment"},
    }


def test_protocol_dependencies_and_bound_evidence_are_frozen() -> None:
    contract.verify_protocol()
    assert contract.verify_dependencies() == contract.DEPENDENCY_SHA256
    bound = contract.verify_bound_inputs()
    assert bound["e9_verdict"] == "PASS_3B_WIDE_LOCALITY"


def test_registered_candidate_population_matches_every_hash() -> None:
    tokenizer = load_pinned_tokenizer(snapshot_path())
    population = contract.selected_population(tokenizer)
    assert population["record"] == {
        "candidate_count": 1200,
        "first_case_id": 20077,
        "last_case_id": 21318,
        "case_sha256": contract.CANDIDATE_CASE_SHA256,
        "case_id_sha256": contract.CANDIDATE_CASE_ID_SHA256,
        "prompt_sha256": contract.CANDIDATE_PROMPT_SHA256,
        "target_id_sha256": contract.CANDIDATE_TARGET_ID_SHA256,
        "target_sha256": contract.CANDIDATE_TARGET_SHA256,
    }
    assert len(population["query_prompt_order"]) == 12_288
    assert not set(population["prompt_order"]) & set(population["query_prompt_order"])


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [sum(text.encode("utf-8")) % 97]


def test_selection_is_ordered_unique_and_exhaustive() -> None:
    records = []
    for case_id in range(20_070, 21_400):
        records.append(
            {
                "case_id": case_id,
                "requested_rewrite": {
                    "prompt": "{} fact " + str(case_id),
                    "subject": "Subject",
                    "target_new": {"str": f"target-{case_id}"},
                    "relation_id": "P1",
                },
            }
        )
    excluded = {"Subject fact 20077"}
    rows = select_capacity_cases(records, _Tokenizer(), excluded)
    assert len(rows) == 1200
    assert rows[0]["case_id"] == 20078
    assert rows[-1]["case_id"] == 21277
    assert len({row["prompt"] for row in rows}) == 1200
    assert population_record(rows)["case_sha256"] == stable_json_sha256(rows)


def test_registered_linear_median_has_one_estimator() -> None:
    assert linear_median([9, 1, 5]) == 5.0
    assert linear_median([1, 2, 9, 10]) == 5.5
    with pytest.raises(ValueError):
        linear_median([])
    with pytest.raises(ValueError):
        linear_median([float("nan")])


def _memory_inputs(count: int = 64, dim: int = 16):
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(count, dim, generator=generator, dtype=torch.float32)
    directions = torch.randn(count, dim, generator=generator, dtype=torch.float64)
    directions /= directions.norm(dim=1, keepdim=True)
    cases = [
        {"case_id": 30_000 + index, "target": f" t{index}", "target_id": index}
        for index in range(count)
    ]
    direct = [
        {"deployment_admitted": True, "decision": "ADMIT_NATIVE", "beta": 2.0}
        for _ in range(count)
    ]
    return cases, hidden, directions, direct


def test_memory_tier_changes_only_active_prefix() -> None:
    cases, hidden, directions, direct = _memory_inputs()
    mean = torch.zeros(hidden.shape[1])
    transform = torch.eye(hidden.shape[1])
    memory, _adapter, slots, receipt = build_capacity_memory(
        cases, hidden, direct, directions, list(range(64)), mean, transform, 64
    )
    assert memory.memory_size == MEMORY_SIZE
    assert int(memory.active.sum()) == 64
    assert torch.equal(memory.active[:64], torch.ones(64, dtype=torch.bool))
    assert not bool(memory.active[64:].any())
    assert slots == {index: index for index in range(64)}
    assert receipt["active_slots"] == list(range(64))
    assert receipt["memory_size"] == 1032
    assert torch.allclose(memory.values[:64], directions.float() * 2.0)


def test_locality_summary_tie_order_and_kind_denominators() -> None:
    rows = []
    for index, kind in enumerate(("exact", "paraphrase", "neighborhood") * 2):
        rows.append(
            {
                "pool_index": index,
                "case_id": index // 3,
                "kind": kind,
                "prompt_sha256": f"{index:064x}",
                "selected_score": 0.5 if index < 2 else 0.1,
                "selected_slot": 0,
                "selected_candidate_index": 3,
                "selected_case_id": 30_003,
                "selected_label": "x",
                "gate_open": False,
                "delta_zero": True,
                "hidden_bit_identical": True,
            }
        )
    summary = summarize_locality(rows)
    assert summary["top_20"][0]["pool_index"] == 0
    assert summary["top_20"][1]["pool_index"] == 1
    assert summary["score_threshold_counts"] == {
        "0.10": 6, "0.50": 2, "0.90": 0, "0.95": 0
    }
    assert all(summary["by_kind"][kind]["query_count"] == 2 for kind in summary["by_kind"])


def test_bundle_is_create_new_and_tensor_exact(tmp_path: Path) -> None:
    payload = {
        "keys": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "values": torch.eye(3, 4),
    }
    manifest = bundle_manifest(payload, ["a", "b", "c"])
    path = tmp_path / "adapter.pt"
    digest = atomic_create_bundle(path, payload)
    assert digest == sha256_file(path)
    loaded = load_bundle(path, manifest)
    assert all(torch.equal(loaded[name], payload[name]) for name in payload)
    with pytest.raises(FileExistsError):
        atomic_create_bundle(path, payload)
    bad = copy.deepcopy(manifest)
    bad["tensors"]["keys"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError):
        load_bundle(path, bad)


def _minimal_locality_rows(count: int = 12_288) -> list[dict[str, object]]:
    kinds = ("exact", "paraphrase", "neighborhood")
    rows = []
    for index in range(count):
        kind = kinds[index % 3]
        native = f"{index:064x}"[-64:]
        rows.append(
            {
                "pool_index": index,
                "case_id": index // 3,
                "kind": kind,
                "prompt_sha256": native,
                "selected_score": 0.1,
                "selected_slot": 0,
                "selected_candidate_index": 0,
                "selected_case_id": 20_077,
                "selected_label": "x",
                "gate_open": False,
                "delta_zero": True,
                "hidden_bit_identical": True,
                "native_hidden_sha256": native,
                "adapted_hidden_sha256": native,
            }
        )
    return rows


def _passing_tier(tier: int, locality_rows: list[dict[str, object]]) -> dict[str, object]:
    positive_rows = [
        {
            "candidate_index": index,
            "target_id": index,
            "prediction": index,
            "own_slot": index,
            "selected_slot": index,
            "selected_score": 1.0,
            "own_slot_attention": 1.0,
            "strongest_nonown_score": 0.0,
            "target_margin": 1.0,
            "gate_open": True,
            "strict_target_success": True,
            "delta_zero": False,
            "prompt_sha256": "1" * 64,
            "native_hidden_sha256": "2" * 64,
            "delta_sha256": "3" * 64,
            "adapted_hidden_sha256": "4" * 64,
            "logits_sha256": "5" * 64,
        }
        for index in range(tier)
    ]
    return {
        "memory_receipt": {
            "memory_size": 1032,
            "active_count": tier,
            "active_slots": list(range(tier)),
            "selected_candidate_indices": list(range(tier)),
            "key_mean_sha256": "a" * 64,
            "key_transform_sha256": "b" * 64,
            "active_labels": [f" t{index}" for index in range(tier)],
        },
        "positive_scan": {
            "rows": positive_rows,
            "summary": {
                "positive_count": tier,
                "gate_open_count": tier,
                "own_slot_count": tier,
                "strict_target_success_count": tier,
                "nonzero_delta_count": tier,
                "minimum_own_slot_attention": 1.0,
                "median_own_slot_attention": 1.0,
                "minimum_target_margin": 1.0,
                "maximum_nonown_score": 0.0,
                "key_coherence": 0.5,
                "degree_five_max_crosstalk": 0.5**5,
            },
        },
        "locality_scan": {
            "rows": locality_rows,
            "summary": summarize_locality(locality_rows),
        },
        "rollback": {
            "post_active_count": 0,
            "post_labels": [None] * 1032,
            "summary": {
                "positive_count": tier,
                "positive_delta_zero_count": tier,
                "positive_hidden_bit_identical_count": tier,
                "locality_count": 12_288,
                "locality_delta_zero_count": 12_288,
                "locality_hidden_bit_identical_count": 12_288,
            },
        },
    }


def test_scientific_bar_requires_every_nested_tier() -> None:
    locality = _minimal_locality_rows()
    measurement = {
        "direct": {"direct_summary": {"selected_count": 1024}},
        "tiers": {str(tier): _passing_tier(tier, locality) for tier in CAPACITY_TIERS},
        "bundle": {"file_sha256": "c" * 64},
    }
    result = {"scientific": {"measurement": measurement}}
    assert contract.scientific_failures(result) == []
    measurement["tiers"]["1024"]["positive_scan"]["summary"]["own_slot_count"] = 1023
    failures = contract.scientific_failures(result)
    assert "tier 1024 positive bar failed: own_slot_count" in failures
    assert contract.largest_passing_tier(result) == 256


def test_measurement_validator_binds_first_admitted_and_all_summaries() -> None:
    from hlm5.e8_runtime import direct_summary

    direct_rows = [
        {
            "case_index": index,
            "case_id": 20_077 + index,
            "target_id": index,
            "geometry_class": "GEOMETRIC_REACHABLE",
            "decision": "ADMIT_NATIVE",
            "baseline_correct": False,
            "eligible": True,
            "deployment_admitted": True,
            "baseline_logits_sha256": "6" * 64,
            "native_checked": True,
            "native_logits_sha256": "7" * 64,
            "native_prediction": index,
            "native_margin": 1.0,
        }
        for index in range(1200)
    ]
    selected = list(range(1024))
    selected_case_ids = [20_077 + index for index in selected]
    summary = direct_summary(direct_rows)
    summary.update(
        {
            "selected_count": 1024,
            "selected_indices": selected,
            "selected_case_ids": selected_case_ids,
            "selected_indices_sha256": stable_json_sha256(selected),
            "selected_case_ids_sha256": stable_json_sha256(selected_case_ids),
        }
    )
    locality = _minimal_locality_rows()
    tiers = {str(tier): _passing_tier(tier, locality) for tier in CAPACITY_TIERS}
    manifest = {
        name: {"shape": shape, "sha256": "8" * 64}
        for name, shape in {
            "keys": [1024, 2048], "values": [1024, 2048],
            "alphas": [1024], "key_mean": [2048],
            "key_transform": [2048, 2048], "selected_indices": [1024],
            "selected_case_ids": [1024],
        }.items()
    }
    measurement = {
        "direct": {"direct_rows": direct_rows, "direct_summary": summary},
        "key_whitening": {
            "exact_key_count": 1200,
            "whitening_prompt_count": 18,
            "floor_fraction": 0.01,
            "mean_sha256": "a" * 64,
            "transform_sha256": "b" * 64,
        },
        "tiers": tiers,
        "bundle": {
            "file_sha256": "9" * 64,
            "tensors": manifest,
            "active_labels": tiers["1024"]["memory_receipt"]["active_labels"],
        },
    }
    assert contract.measurement_validity_failures(measurement) == []
    measurement["direct"]["direct_summary"]["selected_indices"][0] = 99
    assert "E10 did not select the first admitted prefix" in contract.measurement_validity_failures(measurement)


def test_atomic_json_is_single_use_and_staging_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract, "STAGING_DIR", tmp_path / "staging")
    path = contract.STAGING_DIR / "receipt.json"
    contract.atomic_create_json(path, {"x": 1})
    assert json.loads(path.read_text()) == {"x": 1}
    with pytest.raises(FileExistsError):
        contract.atomic_create_json(path, {"x": 2})
    with pytest.raises(ValueError):
        contract.atomic_create_json(tmp_path / "outside.json", {"x": 3})


def test_admission_post_attempt_exception_is_durable_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = _script_globals("admit_e10_3b.py")
    globals_dict = namespace["main"].__globals__
    paths = _patch_paths(globals_dict, tmp_path / "admit", monkeypatch)
    receipt = _fake_receipt()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setitem(
        globals_dict,
        "configure_native_runtime",
        lambda: dict(contract.EXPECTED_NATIVE_RUNTIME),
    )
    monkeypatch.setitem(
        globals_dict,
        "e10_environment_record",
        lambda _device: dict(receipt["environment"]),
    )
    monkeypatch.setitem(
        globals_dict, "load_execution_receipt", lambda: (receipt, "e" * 64)
    )
    monkeypatch.setitem(globals_dict, "prepare_before_attempt", lambda _device: {})

    def fail_after_burn(**_kwargs: object) -> None:
        raise RuntimeError("known E10 post-attempt failure")

    monkeypatch.setitem(globals_dict, "run_after_attempt", fail_after_burn)
    with pytest.raises(RuntimeError, match="known E10 post-attempt failure"):
        namespace["main"]()
    assert paths["ATTEMPT_PATH"].is_file()
    invalid = json.loads(paths["ADMISSION_PATH"].read_text(encoding="utf-8"))
    assert invalid["status"] == "IMPLEMENTATION_INVALID"


def test_admission_environment_mismatch_is_pre_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = _script_globals("admit_e10_3b.py")
    globals_dict = namespace["main"].__globals__
    paths = _patch_paths(globals_dict, tmp_path / "admit-env", monkeypatch)
    receipt = _fake_receipt()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setitem(
        globals_dict,
        "configure_native_runtime",
        lambda: dict(contract.EXPECTED_NATIVE_RUNTIME),
    )
    monkeypatch.setitem(globals_dict, "e10_environment_record", lambda _device: {"drift": 1})
    monkeypatch.setitem(
        globals_dict, "load_execution_receipt", lambda: (receipt, "e" * 64)
    )
    with pytest.raises(RuntimeError, match="environment differs"):
        namespace["main"]()
    assert not paths["ATTEMPT_PATH"].exists()


def test_replay_environment_mismatch_is_durable_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = _script_globals("replay_e10_3b.py")
    globals_dict = namespace["main"].__globals__
    paths = _patch_paths(globals_dict, tmp_path / "replay-env", monkeypatch)
    receipt = _fake_receipt()
    admission = {"scientific_sha256": "science", "scientific": {}}
    monkeypatch.setitem(globals_dict, "load_execution_receipt", lambda: (receipt, "e" * 64))
    monkeypatch.setitem(globals_dict, "load_admission", lambda: (admission, "a" * 64))
    monkeypatch.setitem(globals_dict, "load_attempt", lambda: ({}, "b" * 64))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setitem(
        globals_dict,
        "configure_native_runtime",
        lambda: dict(contract.EXPECTED_NATIVE_RUNTIME),
    )
    monkeypatch.setitem(globals_dict, "e10_environment_record", lambda _device: {"drift": 1})
    with pytest.raises(RuntimeError, match="environment differs"):
        namespace["main"]()
    invalid = json.loads(paths["RESULT_PATH"].read_text(encoding="utf-8"))
    assert invalid["status"] == "IMPLEMENTATION_INVALID"


def test_verifier_incomplete_and_invalid_result_preserve_live_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incomplete_ns = _script_globals("verify_e10_3b.py")
    incomplete_globals = incomplete_ns["main"].__globals__
    paths = _patch_paths(incomplete_globals, tmp_path / "incomplete", monkeypatch)
    paths["ATTEMPT_PATH"].parent.mkdir(parents=True, exist_ok=True)
    paths["ATTEMPT_PATH"].write_text("{}", encoding="utf-8")
    paths["ADMISSION_PATH"].write_text("{}", encoding="utf-8")
    monkeypatch.setitem(incomplete_globals, "load_attempt", lambda: ({}, "attempt"))
    monkeypatch.setitem(
        incomplete_globals,
        "load_admission",
        lambda: ({"scientific_sha256": "science"}, "admission"),
    )
    assert incomplete_ns["main"]() == 3
    value = json.loads(paths["VERDICT_PATH"].read_text(encoding="utf-8"))
    assert value["verdict"] == "INCOMPLETE"

    invalid_ns = _script_globals("verify_e10_3b.py")
    invalid_globals = invalid_ns["main"].__globals__
    paths = _patch_paths(invalid_globals, tmp_path / "invalid", monkeypatch)
    paths["ATTEMPT_PATH"].parent.mkdir(parents=True, exist_ok=True)
    paths["ATTEMPT_PATH"].write_text('{"live":"attempt"}', encoding="utf-8")
    paths["ADMISSION_PATH"].write_text('{"live":"admission"}', encoding="utf-8")
    bogus = {
        "attempt_sha256": "bogus-attempt",
        "admission_sha256": "bogus-admission",
        "admission_scientific_sha256": "bogus-admission-science",
        "scientific_sha256": "bogus-result-science",
    }
    paths["RESULT_PATH"].write_text(json.dumps(bogus), encoding="utf-8")
    monkeypatch.setitem(invalid_globals, "load_attempt", lambda: ({}, "live-attempt"))
    monkeypatch.setitem(
        invalid_globals,
        "load_admission",
        lambda: ({"scientific_sha256": "live-science"}, "live-admission"),
    )

    def invalid_result() -> tuple[dict[str, object], str]:
        raise RuntimeError("invalid E10 result provenance")

    monkeypatch.setitem(invalid_globals, "load_result", invalid_result)
    assert invalid_ns["main"]() == 2
    verdict = json.loads(paths["VERDICT_PATH"].read_text(encoding="utf-8"))
    assert verdict["attempt_sha256"] == "live-attempt"
    assert verdict["admission_sha256"] == "live-admission"
    assert verdict["invalid_result_claims"]["result_scientific_sha256"] == "bogus-result-science"

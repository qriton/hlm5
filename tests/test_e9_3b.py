"""Fast contract, routing, and provenance checks for registered E9."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
import torch

import hlm5.e9_contract as contract
import hlm5.e9_runtime as runtime
from hlm5.e7_contract import stable_json_sha256
from hlm5.memory import EditableHLM5Memory
from hlm5.public_adapter import HLM5PreHeadAdapter


def _script_globals(name: str) -> dict[str, object]:
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / name),
        run_name=f"e9_test_{name.replace('.', '_')}",
    )


def _patch_paths(
    globals_dict: dict[str, object],
    staging: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path]:
    paths = {
        "ATTEMPT_PATH": staging / "attempt.json",
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
        "basis_and_directions": {"basis": 1},
        "e8_memory_receipt": {"memory": 1},
        "key_whitening": {"key": 1},
        "e8_anchor_scan": {"anchor": 1},
    }


def test_protocol_dependencies_and_e8_evidence_are_frozen() -> None:
    contract.verify_protocol()
    assert contract.verify_dependencies() == contract.DEPENDENCY_SHA256
    evidence = contract.verify_bound_inputs()
    assert evidence["e8_verdict"] == "PASS_3B_FULL_PATH_ZCA"
    assert evidence["e8_memory_sha256"] == contract.E8_MEMORY_SHA256
    assert contract.REGISTERED_COMMANDS == (
        "python scripts/preflight_e9_3b.py",
        "python scripts/register_e9_3b.py",
        "python scripts/admit_e9_3b.py",
        "python scripts/replay_e9_3b.py",
        "python scripts/verify_e9_3b.py",
    )


def test_pre_outcome_memory_freeze_allows_only_registered_key_hash_drift() -> None:
    frozen = contract.frozen_e8_measurement()
    expected_memory = frozen["arms"]["candidate"]["memory_receipt"]
    memory = json.loads(json.dumps(expected_memory))
    memory["active_keys_sha256"] = "a" * 64
    memory["key_mean_sha256"] = "b" * 64
    memory["key_transform_sha256"] = "c" * 64
    key_whitening = json.loads(json.dumps(frozen["key_whitening"]))
    key_whitening["mean_sha256"] = memory["key_mean_sha256"]
    key_whitening["transform_sha256"] = memory["key_transform_sha256"]

    assert contract.e8_memory_compatibility_failures(memory, expected_memory) == []
    assert contract.key_whitening_validity_failures(key_whitening, memory) == []

    memory["active_values_sha256"] = "d" * 64
    assert contract.e8_memory_compatibility_failures(memory, expected_memory)


def test_e9_environment_binds_the_physical_node() -> None:
    environment = {**contract.EXPECTED_ENVIRONMENT, "node": "lrdn-test"}
    assert contract.environment_validity_failures(environment) == []
    assert (
        contract.environment_validity_failures(environment, expected_node="lrdn-test")
        == []
    )
    assert contract.environment_validity_failures(
        environment, expected_node="lrdn-other"
    )


def test_registered_query_population_matches_every_hash() -> None:
    population = contract.selected_population()
    rows = population["query_rows"]
    prompts = population["prompt_order"]
    assert len(rows) == contract.QUERY_CASE_COUNT
    assert len(prompts) == contract.QUERY_COUNT
    assert len(set(prompts)) == contract.QUERY_COUNT
    assert rows[0]["case_id"] == 10_000
    assert rows[-1]["case_id"] == 14_607
    assert stable_json_sha256(rows) == contract.QUERY_ROW_SHA256
    assert stable_json_sha256(prompts) == contract.QUERY_PROMPT_SHA256
    assert stable_json_sha256(population["case_ids"]) == (contract.QUERY_CASE_ID_SHA256)
    assert population["kind_sha256"] == contract.QUERY_KIND_SHA256


def test_query_selection_is_disjoint_stable_and_exhaustive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = []
    for case_id in range(runtime.QUERY_CASE_START, runtime.QUERY_CASE_START + 10):
        records.append(
            {
                "case_id": case_id,
                "requested_rewrite": {
                    "prompt": "{} is",
                    "subject": f"subject-{case_id}",
                },
                "paraphrase_prompts": [f"p-{case_id}", f"p-{case_id}"],
                "neighborhood_prompts": [f"n-{case_id}"],
            }
        )
    monkeypatch.setattr(runtime, "QUERY_CASE_COUNT", 3)
    monkeypatch.setattr(runtime, "QUERY_COUNT", 9)
    selected = runtime.select_query_rows(records, ["subject-10000 is"])
    assert [row["case_id"] for row in selected] == [10001, 10002, 10003]
    assert runtime.query_specs(selected)[0] == {
        "pool_index": 0,
        "case_index": 0,
        "case_id": 10001,
        "kind": "exact",
        "prompt": "subject-10001 is",
    }


def test_summary_thresholds_ties_and_top_order_are_exact() -> None:
    rows = []
    scores = [0.95, 0.95, 0.50, 0.10, 0.09, 0.0]
    kinds = ["exact", "paraphrase", "neighborhood"] * 2
    for index, (score, kind) in enumerate(zip(scores, kinds, strict=True)):
        rows.append(
            {
                "pool_index": index,
                "case_id": index,
                "kind": kind,
                "prompt_sha256": f"p{index}",
                "selected_score": score,
                "selected_slot": 0,
                "selected_e8_case_id": 20000,
                "selected_label": " target",
                "gate_open": score >= 0.95,
                "delta_zero": score < 0.95,
                "hidden_bit_identical": score < 0.95,
            }
        )
    summary = runtime.summarize_query_scan(rows)
    assert summary["maximum"]["pool_index"] == 0
    assert [row["pool_index"] for row in summary["top_20"]] == [0, 1, 2, 3, 4, 5]
    assert summary["score_threshold_counts"] == {
        "0.10": 4,
        "0.50": 3,
        "0.90": 2,
        "0.95": 2,
    }


def test_released_adapter_scan_can_detect_closed_and_open_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dim = 4
    memory = EditableHLM5Memory(
        dim=dim,
        memory_size=4,
        degree=5,
        temperature=0.10,
        learnable_memory=False,
    )
    memory.inject(
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        value=torch.tensor([0.0, 0.0, 1.0, 0.0]),
        label=" A",
    )
    memory.inject(
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
        value=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        label=" B",
    )
    adapter = HLM5PreHeadAdapter(
        memory, torch.zeros(dim), torch.eye(dim), boost=1.0, gate_thresh=0.95
    )
    e8_cases = [{"case_id": 20000}, {"case_id": 20001}]
    query_rows = [
        {"case_id": 1, "exact": "a", "paraphrase": "b", "neighborhood": "c"},
        {"case_id": 2, "exact": "d", "paraphrase": "e", "neighborhood": "f"},
    ]
    hidden = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, -1.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.bfloat16,
    )
    monkeypatch.setattr(runtime, "QUERY_COUNT", 6)
    monkeypatch.setattr(runtime, "GATE_BATCH", 4)
    scan = runtime.scan_queries(
        adapter, memory, e8_cases, {0: 0, 1: 1}, query_rows, hidden
    )
    assert [row["gate_batch_size"] for row in scan["rows"]] == [4, 4, 4, 4, 2, 2]
    assert scan["rows"][0]["gate_open"] is False
    assert scan["rows"][0]["delta_zero"] is True
    assert scan["rows"][0]["hidden_bit_identical"] is True
    assert (
        scan["rows"][0]["native_hidden_sha256"]
        == scan["rows"][0]["adapted_hidden_sha256"]
    )
    assert scan["rows"][2]["gate_open"] is True
    assert scan["rows"][2]["delta_zero"] is False
    assert scan["rows"][2]["hidden_bit_identical"] is False
    assert (
        scan["rows"][2]["native_hidden_sha256"]
        != scan["rows"][2]["adapted_hidden_sha256"]
    )
    assert scan["rows"][5]["gate_open"] is True
    assert scan["summary"]["gate_open_count"] == 2


def test_scientific_bar_rejects_any_global_or_per_kind_open() -> None:
    def result(global_count: int, exact_count: int = 0) -> dict:
        return {
            "scientific": {
                "measurement": {
                    "query_scan": {
                        "summary": {
                            "query_count": contract.QUERY_COUNT,
                            "gate_open_count": global_count,
                            "by_kind": {
                                "exact": {
                                    "query_count": contract.QUERY_CASE_COUNT,
                                    "gate_open_count": exact_count,
                                },
                                "paraphrase": {
                                    "query_count": contract.QUERY_CASE_COUNT,
                                    "gate_open_count": 0,
                                },
                                "neighborhood": {
                                    "query_count": contract.QUERY_CASE_COUNT,
                                    "gate_open_count": 0,
                                },
                            },
                        }
                    }
                }
            }
        }

    assert contract.scientific_failures(result(0)) == []
    assert contract.scientific_failures(result(1))
    assert contract.scientific_failures(result(0, exact_count=1))


def test_atomic_create_is_single_use_and_staging_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "e9"
    monkeypatch.setattr(contract, "STAGING_DIR", staging)
    output = staging / "receipt.json"
    contract.atomic_create_json(output, {"a": 1})
    assert contract.load_json_object(output) == {"a": 1}
    with pytest.raises(FileExistsError):
        contract.atomic_create_json(output, {"a": 2})
    with pytest.raises(ValueError):
        contract.atomic_create_json(tmp_path / "outside.json", {})


def test_result_exact_replay_rejects_measurement_or_hash_drift() -> None:
    admission = {
        "scientific_sha256": "admission-science",
        "scientific": {
            "model_invariants": {"model": 1},
            "basis_and_directions": {"basis": 1},
            "measurement": {"value": 1},
        },
    }
    scientific = {
        "admission_scientific_sha256": "admission-science",
        "model_invariants": {"model": 1},
        "basis_and_directions": {"basis": 1},
        "measurement": {"value": 1},
        "measurement_exact_match": True,
        "all_tensor_hashes_match": True,
    }
    result = {
        "scientific": scientific,
        "scientific_sha256": contract.scientific_sha256(scientific),
    }
    assert contract.result_validity_failures(result, admission) == []
    result["scientific"]["all_tensor_hashes_match"] = False
    result["scientific_sha256"] = contract.scientific_sha256(result["scientific"])
    assert contract.result_validity_failures(result, admission)


def test_admission_post_attempt_exception_is_durable_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = _script_globals("admit_e9_3b.py")
    globals_dict = namespace["main"].__globals__
    paths = _patch_paths(globals_dict, tmp_path / "admit", monkeypatch)
    receipt = _fake_receipt()
    prepared = {
        "reconstruction": {
            "basis_and_directions": receipt["basis_and_directions"],
            "memory_receipt": receipt["e8_memory_receipt"],
            "key_whitening": receipt["key_whitening"],
            "anchor_scan": receipt["e8_anchor_scan"],
        }
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setitem(
        globals_dict,
        "configure_native_runtime",
        lambda: dict(contract.EXPECTED_NATIVE_RUNTIME),
    )
    monkeypatch.setitem(
        globals_dict,
        "e9_environment_record",
        lambda _device: dict(receipt["environment"]),
    )
    monkeypatch.setitem(
        globals_dict, "load_execution_receipt", lambda: (receipt, "execution")
    )
    monkeypatch.setitem(
        globals_dict, "prepare_before_attempt", lambda _device: prepared
    )

    def fail_after_burn(**_kwargs: object) -> None:
        raise RuntimeError("known E9 post-attempt failure")

    monkeypatch.setitem(globals_dict, "run_after_attempt", fail_after_burn)
    with pytest.raises(RuntimeError, match="known E9 post-attempt failure"):
        namespace["main"]()
    assert paths["ATTEMPT_PATH"].is_file()
    invalid = json.loads(paths["ADMISSION_PATH"].read_text(encoding="utf-8"))
    assert invalid["status"] == "IMPLEMENTATION_INVALID"


def test_admission_environment_mismatch_is_pre_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = _script_globals("admit_e9_3b.py")
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
    monkeypatch.setitem(
        globals_dict, "e9_environment_record", lambda _device: {"drift": 1}
    )
    monkeypatch.setitem(
        globals_dict, "load_execution_receipt", lambda: (receipt, "execution")
    )
    with pytest.raises(RuntimeError, match="environment differs"):
        namespace["main"]()
    assert not paths["ATTEMPT_PATH"].exists()


def test_replay_environment_mismatch_is_durable_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = _script_globals("replay_e9_3b.py")
    globals_dict = namespace["main"].__globals__
    paths = _patch_paths(globals_dict, tmp_path / "replay-env", monkeypatch)
    receipt = _fake_receipt()
    admission = {"scientific_sha256": "admission-science", "scientific": {}}
    monkeypatch.setitem(
        globals_dict, "load_execution_receipt", lambda: (receipt, "execution")
    )
    monkeypatch.setitem(
        globals_dict, "load_admission", lambda: (admission, "admission")
    )
    monkeypatch.setitem(globals_dict, "load_attempt", lambda: ({}, "attempt"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setitem(
        globals_dict,
        "configure_native_runtime",
        lambda: dict(contract.EXPECTED_NATIVE_RUNTIME),
    )
    monkeypatch.setitem(
        globals_dict, "e9_environment_record", lambda _device: {"drift": 1}
    )
    with pytest.raises(RuntimeError, match="environment differs"):
        namespace["main"]()
    invalid = json.loads(paths["RESULT_PATH"].read_text(encoding="utf-8"))
    assert invalid["status"] == "IMPLEMENTATION_INVALID"
    assert invalid["attempt_sha256"] == "attempt"


def test_verifier_incomplete_and_invalid_result_preserves_live_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incomplete_ns = _script_globals("verify_e9_3b.py")
    incomplete_globals = incomplete_ns["main"].__globals__
    incomplete_paths = _patch_paths(
        incomplete_globals, tmp_path / "incomplete", monkeypatch
    )
    incomplete_paths["ATTEMPT_PATH"].parent.mkdir(parents=True, exist_ok=True)
    incomplete_paths["ATTEMPT_PATH"].write_text("{}", encoding="utf-8")
    incomplete_paths["ADMISSION_PATH"].write_text("{}", encoding="utf-8")
    monkeypatch.setitem(incomplete_globals, "load_attempt", lambda: ({}, "attempt"))
    monkeypatch.setitem(
        incomplete_globals,
        "load_admission",
        lambda: ({"scientific_sha256": "science"}, "admission"),
    )
    assert incomplete_ns["main"]() == 3
    incomplete = json.loads(
        incomplete_paths["VERDICT_PATH"].read_text(encoding="utf-8")
    )
    assert incomplete["verdict"] == "INCOMPLETE"

    invalid_ns = _script_globals("verify_e9_3b.py")
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
        raise RuntimeError("invalid E9 result provenance")

    monkeypatch.setitem(invalid_globals, "load_result", invalid_result)
    assert invalid_ns["main"]() == 2
    verdict = json.loads(paths["VERDICT_PATH"].read_text(encoding="utf-8"))
    assert verdict["verdict"] == "IMPLEMENTATION_INVALID"
    assert verdict["attempt_sha256"] == "live-attempt"
    assert verdict["admission_sha256"] == "live-admission"
    assert verdict["invalid_result_claims"] == {
        "attempt_sha256": "bogus-attempt",
        "admission_sha256": "bogus-admission",
        "admission_scientific_sha256": "bogus-admission-science",
        "result_scientific_sha256": "bogus-result-science",
    }

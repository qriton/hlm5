"""Fast CPU contract tests for HLM5 E11 cross-node portability."""

from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path

import pytest
import torch

import hlm5.e11_contract as contract
from hlm5.e11_runtime import E10_NODE, adapter_from_bundle, load_bundle


def _script_globals(name: str) -> dict[str, object]:
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / name),
        run_name=f"e11_test_{name.replace('.', '_')}",
    )


def _patch_paths(
    namespace: dict[str, object], staging: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path]:
    globals_dict = namespace["main"].__globals__
    paths = {
        name: staging / filename
        for name, filename in {
            "ATTEMPT_PATH": "attempt.json",
            "ADMISSION_PATH": "admission.json",
            "RESULT_PATH": "result.json",
            "VERDICT_PATH": "verdict.json",
        }.items()
    }
    monkeypatch.setattr(contract, "STAGING_DIR", staging)
    for name, path in paths.items():
        if name in globals_dict:
            monkeypatch.setitem(globals_dict, name, path)
    return paths


def test_protocol_dependencies_and_e10_evidence_are_frozen() -> None:
    contract.verify_protocol()
    assert contract.verify_dependencies() == contract.DEPENDENCY_SHA256
    evidence = contract.frozen_e10()
    assert len(evidence["selected_indices"]) == 1024
    assert (
        evidence["tier"]["positive_scan"]["summary"]["strict_target_success_count"]
        == 1024
    )
    assert evidence["tier"]["locality_scan"]["summary"]["gate_open_count"] == 0


def test_environment_requires_a_different_physical_node() -> None:
    valid = {**contract.EXPECTED_ENVIRONMENT, "node": "lrdn-other.leonardo.local"}
    assert contract.environment_failures(valid) == []
    same = {**contract.EXPECTED_ENVIRONMENT, "node": E10_NODE}
    assert "E11 physical node equals E10 node" in contract.environment_failures(same)
    assert contract.environment_failures(valid, expected_node="another")


def test_bundle_reconstructs_the_exact_e10_memory_receipt() -> None:
    evidence = contract.frozen_e10()
    bundle = load_bundle(contract.BUNDLE_PATH, evidence["bundle"])
    memory, _adapter, slots, receipt = adapter_from_bundle(
        bundle,
        evidence["bundle"]["active_labels"],
        evidence["selected_indices"],
        torch.device("cpu"),
    )
    assert receipt == evidence["tier"]["memory_receipt"]
    assert slots == {
        candidate: slot for slot, candidate in enumerate(evidence["selected_indices"])
    }
    assert int(memory.active.sum()) == 1024
    assert not bool(memory.active[1024:].any())


def test_bundle_rejects_selected_index_substitution() -> None:
    evidence = contract.frozen_e10()
    bundle = load_bundle(contract.BUNDLE_PATH, evidence["bundle"])
    substituted = list(evidence["selected_indices"])
    substituted[0], substituted[1] = substituted[1], substituted[0]
    with pytest.raises(RuntimeError, match="selected indices"):
        adapter_from_bundle(
            bundle,
            evidence["bundle"]["active_labels"],
            substituted,
            torch.device("cpu"),
        )


def test_structural_and_scientific_mismatch_are_separate() -> None:
    evidence = contract.frozen_e10()
    measurement = copy.deepcopy(evidence["tier"])
    assert contract.structural_failures(measurement) == []
    result = {"scientific": {"measurement": measurement}}
    assert contract.scientific_failures(result) == []
    measurement["positive_scan"]["rows"][0]["selected_score"] -= 1e-7
    assert contract.structural_failures(measurement) == []
    assert contract.scientific_failures(result) == [
        "cross-node E11 tier differs from frozen E10 tier"
    ]
    measurement["positive_scan"]["rows"].pop()
    assert "E11 positive denominator mismatch" in contract.structural_failures(
        measurement
    )


def test_atomic_json_is_single_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(contract, "STAGING_DIR", tmp_path / "staging")
    path = contract.STAGING_DIR / "value.json"
    contract.atomic_create_json(path, {"x": 1})
    assert json.loads(path.read_text()) == {"x": 1}
    with pytest.raises(FileExistsError):
        contract.atomic_create_json(path, {"x": 2})
    with pytest.raises(ValueError):
        contract.atomic_create_json(tmp_path / "outside.json", {})


def test_admission_environment_mismatch_is_pre_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = _script_globals("admit_e11_3b.py")
    paths = _patch_paths(namespace, tmp_path / "admit-env", monkeypatch)
    globals_dict = namespace["main"].__globals__
    receipt = {
        "implementation_commit": "impl",
        "native_runtime": dict(contract.EXPECTED_NATIVE_RUNTIME),
        "environment": {"node": "registered"},
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setitem(
        globals_dict,
        "configure_native_runtime",
        lambda: dict(contract.EXPECTED_NATIVE_RUNTIME),
    )
    monkeypatch.setitem(
        globals_dict, "e11_environment_record", lambda _device: {"node": "drift"}
    )
    monkeypatch.setitem(
        globals_dict, "load_execution_receipt", lambda: (receipt, "e" * 64)
    )
    with pytest.raises(RuntimeError, match="environment differs"):
        namespace["main"]()
    assert not paths["ATTEMPT_PATH"].exists()


def test_admission_post_burn_exception_is_durable_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = _script_globals("admit_e11_3b.py")
    paths = _patch_paths(namespace, tmp_path / "admit", monkeypatch)
    globals_dict = namespace["main"].__globals__
    receipt = {
        "implementation_commit": "impl",
        "native_runtime": dict(contract.EXPECTED_NATIVE_RUNTIME),
        "environment": {"node": "registered"},
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setitem(
        globals_dict,
        "configure_native_runtime",
        lambda: dict(contract.EXPECTED_NATIVE_RUNTIME),
    )
    monkeypatch.setitem(
        globals_dict, "e11_environment_record", lambda _device: {"node": "registered"}
    )
    monkeypatch.setitem(
        globals_dict, "load_execution_receipt", lambda: (receipt, "e" * 64)
    )
    monkeypatch.setitem(globals_dict, "prepare_before_attempt", lambda _device: {})
    monkeypatch.setitem(
        globals_dict,
        "run_after_attempt",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("known post-burn")),
    )
    with pytest.raises(RuntimeError, match="known post-burn"):
        namespace["main"]()
    assert paths["ATTEMPT_PATH"].is_file()
    assert (
        json.loads(paths["ADMISSION_PATH"].read_text())["status"]
        == "IMPLEMENTATION_INVALID"
    )

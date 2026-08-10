"""Fast contract, math, and full-path checks for the registered E8 study."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
import torch

import hlm5.e8_contract as contract
from hlm5.e7_contract import sha256_file, snapshot_path, stable_json_sha256
from hlm5.e7_runtime import load_pinned_tokenizer
from hlm5.e7c_runtime import geometry_for_directions, raw_and_zca_directions
from hlm5.e8_runtime import (
    build_memory,
    evaluate_memory_arm,
    paired_geometry,
    prompt_order,
    rollback_memory,
    select_cases,
    strongest_competitor,
)
from hlm5.public_adapter import HLM5PreHeadAdapter


def _script_globals(name: str) -> dict[str, object]:
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / name),
        run_name=f"e8_test_{name.replace('.', '_')}",
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
    }


class _TinyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [len(text)] if not text.endswith(" reject") else [1, 2]


def test_protocol_inputs_dependencies_and_commands_are_frozen() -> None:
    contract.verify_protocol()
    assert contract.verify_dependencies() == contract.DEPENDENCY_SHA256
    evidence = contract.verify_bound_inputs()
    assert evidence["e7c_verdict"] == "PASS_3B_ZCA_GEOMETRY"
    assert sha256_file(contract.REPO_ROOT / contract.PROTOCOL_PATH) == (
        contract.PROTOCOL_SHA256
    )
    assert contract.REGISTERED_COMMANDS == (
        "python scripts/preflight_e8_3b.py",
        "python scripts/register_e8_3b.py",
        "python scripts/admit_e8_3b.py",
        "python scripts/replay_e8_3b.py",
        "python scripts/verify_e8_3b.py",
    )


def test_registered_population_and_prompt_order_match_hashes() -> None:
    tokenizer = load_pinned_tokenizer(snapshot_path())
    population = contract.selected_population(tokenizer)
    assert [case["case_id"] for case in population["cases"]] == contract.CASE_IDS
    assert stable_json_sha256(population["cases"]) == contract.CASE_SHA256
    assert len(population["prompt_order"]) == contract.PROMPT_COUNT
    assert stable_json_sha256(population["prompt_order"]) == contract.PROMPT_SHA256


def test_selection_is_structural_and_prompt_order_is_stable_deduped() -> None:
    records = []
    for case_id in range(contract.CASE_START, contract.CASE_START + 70):
        target = "reject" if case_id == contract.CASE_START else f"target-{case_id}"
        records.append(
            {
                "case_id": case_id,
                "requested_rewrite": {
                    "target_new": {"str": target},
                    "prompt": "{} is",
                    "subject": f"subject-{case_id}",
                    "relation_id": "P1",
                },
                "paraphrase_prompts": ["shared", f"p-{case_id}", "shared"],
                "neighborhood_prompts": [f"n-{case_id}", f"n2-{case_id}"],
            }
        )
    selected = select_cases(records, _TinyTokenizer())
    assert len(selected) == contract.CASE_COUNT
    assert selected[0]["case_id"] == contract.CASE_START + 1
    ordered = prompt_order(selected[:1], ["shared", "neutral"], ["neutral", "white"])
    assert ordered == [
        selected[0]["prompt"],
        "shared",
        selected[0]["paraphrases"][1],
        *selected[0]["neighborhood_prompts"],
        "neutral",
        "white",
    ]


def test_paired_geometry_is_the_e7c_geometry_at_each_own_hidden() -> None:
    generator = torch.Generator(device="cpu").manual_seed(41)
    head = torch.randn((48, 8), generator=generator, dtype=torch.float64)
    hidden = torch.randn((5, 8), generator=generator).to(torch.bfloat16)
    targets = torch.tensor([1, 3, 7, 11, 19], dtype=torch.long)
    directions = torch.randn((5, 8), generator=generator, dtype=torch.float64)
    directions /= directions.norm(dim=1, keepdim=True) + 1e-12
    actual = paired_geometry(head, hidden, targets, directions)
    expected = [
        geometry_for_directions(
            head,
            hidden[index],
            targets[index : index + 1],
            directions[index : index + 1],
        )[0]
        for index in range(len(targets))
    ]
    discrete_fields = (
        "target_id",
        "hard_blocker",
        "hard_blocker_id",
        "geometry_class",
        "decision",
    )
    assert [[row.get(field) for field in discrete_fields] for row in actual] == [
        [row.get(field) for field in discrete_fields] for row in expected
    ]
    for row, reference in zip(actual, expected, strict=True):
        for field in ("L", "U", "beta", "float64_margin"):
            if reference[field] is None:
                assert row[field] is None
            else:
                assert row[field] == pytest.approx(
                    reference[field], rel=1e-12, abs=1e-12
                )


def test_zca_direction_is_distinct_and_finite_on_a_fixed_synthetic_head() -> None:
    head = torch.tensor(
        [
            [20.0, 0.0, 0.0],
            [19.0, 0.1, 0.0],
            [18.0, 0.0, 0.1],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=torch.bfloat16,
    )
    targets = torch.tensor([3, 4], dtype=torch.long)
    raw, zca, _record = raw_and_zca_directions(head, targets)
    assert not torch.equal(raw, zca)
    raw_slopes = (head.float()[targets] * raw.float()).sum(dim=1)
    zca_slopes = (head.float()[targets] * zca.float()).sum(dim=1)
    assert torch.all(zca_slopes > 0)
    assert torch.isfinite(raw_slopes).all()


def test_strongest_competitor_uses_lowest_id_on_a_tie() -> None:
    logits = torch.tensor([3.0, 5.0, 5.0, 9.0], dtype=torch.bfloat16)
    competitor_id, value = strongest_competitor(logits, target_id=3)
    assert competitor_id == 1
    assert value == 5.0


class _FakeModel:
    def __init__(self, weight: torch.Tensor) -> None:
        self.head = torch.nn.Linear(
            weight.shape[1], weight.shape[0], bias=False, dtype=torch.bfloat16
        )
        self.head.weight.data.copy_(weight.to(torch.bfloat16))

    def get_output_embeddings(self) -> torch.nn.Linear:
        return self.head


def test_released_memory_gate_and_complete_rollback_are_exact() -> None:
    dim = 4
    cases = [
        {
            "case_id": 1,
            "prompt": "exact-a",
            "target": " A",
            "target_id": 2,
            "paraphrases": ["pa1", "pa2"],
            "neighborhood_prompts": ["na1", "na2"],
        },
        {
            "case_id": 2,
            "prompt": "exact-b",
            "target": " B",
            "target_id": 3,
            "paraphrases": ["pb1", "pb2"],
            "neighborhood_prompts": ["nb1", "nb2"],
        },
    ]
    exact = torch.eye(dim, dtype=torch.bfloat16)[:2]
    directions = torch.eye(dim, dtype=torch.float64)[2:4]
    direct = [
        {
            "deployment_admitted": True,
            "beta": 2.0,
        },
        {
            "deployment_admitted": True,
            "beta": 2.0,
        },
    ]
    key_mean = torch.zeros(dim)
    key_transform = torch.eye(dim)
    memory, slots, receipt = build_memory(
        cases, exact, direct, directions, key_mean, key_transform
    )
    assert receipt["active_count"] == 2
    adapter = HLM5PreHeadAdapter(
        memory, key_mean, key_transform, boost=1.0, gate_thresh=0.95
    )
    weight = torch.zeros((8, dim), dtype=torch.bfloat16)
    weight[2, 2] = 1
    weight[3, 3] = 1
    model = _FakeModel(weight)
    off = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.bfloat16)
    prompt_list = prompt_order(cases, ["neutral"], [])
    hidden_by_prompt = {prompt: off.clone() for prompt in prompt_list}
    hidden_by_prompt["exact-a"] = exact[0]
    hidden_by_prompt["exact-b"] = exact[1]
    rows, summary = evaluate_memory_arm(
        model, adapter, hidden_by_prompt, cases, ["neutral"], direct, slots
    )
    exact_rows = [row for row in rows if row["kind"] == "exact"]
    assert all(row["gate_open"] for row in exact_rows)
    assert all(row["selected_slot"] == row["own_slot"] for row in exact_rows)
    assert all(row["target_success_strict"] for row in exact_rows)
    assert summary["false_gate_applications"] == 0
    closed = [row for row in rows if not row["gate_open"]]
    assert closed and all(row["logits_bit_identical"] for row in closed)
    rollback = rollback_memory(
        model, memory, adapter, prompt_list, hidden_by_prompt, receipt["active_slots"]
    )
    assert rollback["summary"] == {
        "prompt_count": len(prompt_list),
        "gate_open_count": 0,
        "delta_zero_count": len(prompt_list),
        "bit_identical_count": len(prompt_list),
    }


def test_scientific_bars_use_exact_integer_arithmetic() -> None:
    def result(eligible: int, raw: int, candidate: int, lost: int) -> dict:
        return {
            "scientific": {
                "measurement": {
                    "summary": {
                        "eligible_count": eligible,
                        "raw_admitted": raw,
                        "candidate_admitted": candidate,
                        "candidate_lost_raw_count": lost,
                        "candidate_gate": {
                            "admitted_exact_checked": candidate,
                            "admitted_exact_gate_open": candidate,
                            "admitted_exact_own_slot": candidate,
                            "admitted_exact_target_success": candidate,
                            "false_gate_applications": 0,
                        },
                        "rollback": {
                            "prompt_count": contract.PROMPT_COUNT,
                            "gate_open_count": 0,
                            "delta_zero_count": contract.PROMPT_COUNT,
                            "bit_identical_count": contract.PROMPT_COUNT,
                        },
                    }
                }
            }
        }

    assert contract.scientific_failures(result(60, 56, 57, 1)) == []
    assert contract.scientific_failures(result(60, 56, 56, 0))
    assert contract.scientific_failures(result(47, 45, 45, 0))
    assert contract.scientific_failures(result(60, 58, 57, 0))
    assert contract.scientific_failures(result(60, 55, 57, 2))


def test_atomic_create_is_single_use_and_staging_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "e8"
    monkeypatch.setattr(contract, "STAGING_DIR", staging)
    output = staging / "receipt.json"
    contract.atomic_create_json(output, {"a": 1})
    assert contract.load_json_object(output) == {"a": 1}
    with pytest.raises(FileExistsError):
        contract.atomic_create_json(output, {"a": 2})
    with pytest.raises(ValueError):
        contract.atomic_create_json(tmp_path / "outside.json", {})


def test_expected_basis_rejects_shape_or_basis_drift() -> None:
    record = {
        **contract.EXPECTED_BASIS,
        "raw_directions_sha256": "a" * 64,
        "zca_directions_sha256": "b" * 64,
        "operator_dtype": "float64",
        "operator_shape": [2048, 2048],
        "direction_dtype": "float64",
        "direction_shape": [64, 2048],
    }
    contract._verify_basis(record)
    record["direction_shape"] = [63, 2048]
    with pytest.raises(RuntimeError, match="direction tensor"):
        contract._verify_basis(record)


def test_result_exact_replay_rejects_any_measurement_drift() -> None:
    measurement = {"value": 1}
    admission = {
        "scientific_sha256": "admission-science",
        "scientific": {
            "model_invariants": {"model": 1},
            "basis_and_directions": {"basis": 1},
            "measurement": measurement,
        },
    }
    scientific = {
        "admission_scientific_sha256": "admission-science",
        "model_invariants": {"model": 1},
        "basis_and_directions": {"basis": 1},
        "measurement": measurement,
        "measurement_exact_match": True,
        "raw_full_logit_hashes_match": True,
        "candidate_full_logit_hashes_match": True,
        "rollback_full_logit_hashes_match": True,
    }
    result = {
        "scientific": scientific,
        "scientific_sha256": contract.scientific_sha256(scientific),
    }
    assert contract.result_validity_failures(result, admission) == []
    result["scientific"]["measurement"] = {"value": 2}
    result["scientific_sha256"] = contract.scientific_sha256(result["scientific"])
    assert contract.result_validity_failures(result, admission)


def test_admission_post_attempt_exception_is_durable_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = _script_globals("admit_e8_3b.py")
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
        "environment_record",
        lambda _device: dict(receipt["environment"]),
    )
    monkeypatch.setitem(
        globals_dict, "load_execution_receipt", lambda: (receipt, "execution")
    )

    def fail_after_burn(**_kwargs: object) -> None:
        raise RuntimeError("known post-attempt failure")

    monkeypatch.setitem(globals_dict, "run_after_attempt", fail_after_burn)
    with pytest.raises(RuntimeError, match="known post-attempt failure"):
        namespace["main"]()
    assert paths["ATTEMPT_PATH"].is_file()
    invalid = json.loads(paths["ADMISSION_PATH"].read_text(encoding="utf-8"))
    assert invalid["status"] == "IMPLEMENTATION_INVALID"


def test_admission_environment_mismatch_is_pre_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = _script_globals("admit_e8_3b.py")
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
        globals_dict, "environment_record", lambda _device: {"drift": 1}
    )
    monkeypatch.setitem(
        globals_dict, "load_execution_receipt", lambda: (receipt, "execution")
    )
    with pytest.raises(RuntimeError, match="environment differs"):
        namespace["main"]()
    assert not paths["ATTEMPT_PATH"].exists()
    assert not paths["ADMISSION_PATH"].exists()


def test_replay_exception_is_durable_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = _script_globals("replay_e8_3b.py")
    globals_dict = namespace["main"].__globals__
    paths = _patch_paths(globals_dict, tmp_path / "replay", monkeypatch)
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
        globals_dict, "environment_record", lambda _device: {"drift": 1}
    )
    with pytest.raises(RuntimeError, match="environment differs"):
        namespace["main"]()
    invalid = json.loads(paths["RESULT_PATH"].read_text(encoding="utf-8"))
    assert invalid["status"] == "IMPLEMENTATION_INVALID"
    assert invalid["attempt_sha256"] == "attempt"


def test_verifier_incomplete_and_invalid_result_claims_keep_live_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incomplete_namespace = _script_globals("verify_e8_3b.py")
    incomplete_globals = incomplete_namespace["main"].__globals__
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
    assert incomplete_namespace["main"]() == 3
    incomplete = json.loads(
        incomplete_paths["VERDICT_PATH"].read_text(encoding="utf-8")
    )
    assert incomplete["verdict"] == "INCOMPLETE"

    invalid_namespace = _script_globals("verify_e8_3b.py")
    invalid_globals = invalid_namespace["main"].__globals__
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
        raise RuntimeError("invalid result provenance")

    monkeypatch.setitem(invalid_globals, "load_result", invalid_result)
    assert invalid_namespace["main"]() == 2
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

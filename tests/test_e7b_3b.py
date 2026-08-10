"""Fast contract and math checks for the registered E7b study."""

from __future__ import annotations

import json
import runpy
from copy import deepcopy
from pathlib import Path

import pytest
import torch

import hlm5.e7b_contract as contract
from hlm5.e7_contract import stable_json_sha256, tensor_sha256
from hlm5.e7b_runtime import (
    GEOMETRIC_REACHABLE,
    NATIVE_ADMIT,
    REFUSE_HARD,
    REFUSE_INTERVAL,
    REFUSE_TIE,
    REFUSE_WRONG,
    geometry_rows,
    native_decision,
    native_rows,
    summarize_rows,
)


_E7_REFERENCE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "run_e7_3b.py")
)
e7_build_slope_matrix = _E7_REFERENCE["build_slope_matrix"]
e7_envelope_for_key = _E7_REFERENCE["envelope_for_key"]
e7_grid_dose = _E7_REFERENCE["grid_dose"]


def _script_globals(name: str) -> dict[str, object]:
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / name),
        run_name=f"e7b_test_{name.replace('.', '_')}",
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


class _FakeTokenizer:
    def __init__(self, vocab: dict[str, int]) -> None:
        self._vocab = vocab

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)


def test_eligible_token_rule_is_sorted_and_filtered() -> None:
    tokenizer = _FakeTokenizer(
        {
            "Ġzebra": 9,
            "Ġan": 1,
            "plain": 7,
            "ĠAlpha": 5,
            "Ġabc123": 4,
            "Ġbeta": 3,
        }
    )
    assert contract.eligible_token_ids(tokenizer) == [3, 5, 9]


def test_native_rule_refuses_ties_and_wrong_argmax() -> None:
    assert (
        native_decision(target_id=7, prediction=2, native_margin=0.0)
        == REFUSE_TIE
    )
    assert (
        native_decision(target_id=7, prediction=2, native_margin=-1.0)
        == REFUSE_WRONG
    )
    assert (
        native_decision(target_id=7, prediction=7, native_margin=0.5)
        == NATIVE_ADMIT
    )


def test_e7_spent_sentinel_is_refused_by_new_rule() -> None:
    evidence = contract.verify_e7_inputs()
    assert evidence["sentinel"]["target_id"] == 922
    assert evidence["sentinel"]["e7b_rule_decision"] == REFUSE_TIE


def test_geometry_exactly_matches_e7_reference() -> None:
    generator = torch.Generator(device="cpu").manual_seed(42)
    head = torch.randn((48, 12), generator=generator, dtype=torch.float64)
    hidden = torch.randn((12,), generator=generator).to(torch.bfloat16)
    targets = torch.tensor([1, 3, 7, 9, 12, 17, 23, 31, 37, 44])

    rows, directions, slopes = geometry_rows(head, hidden, targets)
    e7_directions, e7_slopes = e7_build_slope_matrix(head, targets)
    assert torch.equal(directions, e7_directions)
    assert torch.equal(slopes, e7_slopes)

    base = head @ hidden.to(torch.float64)
    _summary, arrays = e7_envelope_for_key(
        base,
        e7_slopes,
        targets,
        keep_arrays=True,
    )
    assert arrays is not None
    for column, row in enumerate(rows):
        assert (row["geometry_class"] == GEOMETRIC_REACHABLE) == bool(
            arrays["reach"][column]
        )
        assert row["hard_blocker"] == bool(arrays["hard"][column])
        assert row["L"] == float(arrays["lower"][column])
        expected_upper = float(arrays["upper"][column])
        assert row["U"] == (expected_upper if torch.isfinite(arrays["upper"][column]) else None)
        if row["geometry_class"] != GEOMETRIC_REACHABLE:
            continue
        target_id = int(targets[column])
        intercept = base[target_id] - base
        intercept[target_id] = float("inf")
        beta, margin = e7_grid_dose(
            intercept,
            e7_slopes[:, column],
            row["L"],
            expected_upper,
        )
        assert row["beta"] == beta
        assert row["float64_margin"] == margin


def test_summary_partitions_every_row() -> None:
    rows = [
        {"decision": REFUSE_HARD},
        {"decision": REFUSE_INTERVAL},
        {"decision": REFUSE_TIE},
        {"decision": REFUSE_WRONG},
        {"decision": NATIVE_ADMIT},
    ]
    summary = summarize_rows(rows)
    assert summary["target_count"] == 5
    assert summary["geometrically_reachable"] == 3
    assert summary["native_admitted"] == 1
    assert sum(summary["decision_counts"].values()) == 5


def test_scientific_bar_rejects_vacuous_and_rounding_shortfall() -> None:
    vacuous = {
        "scientific": {
            "summary": {"geometrically_reachable": 1, "native_admitted": 1},
            "rows": [],
        }
    }
    assert any("below" in failure for failure in contract.scientific_failures(vacuous))

    short = {
        "scientific": {
            "summary": {
                "geometrically_reachable": 1000,
                "native_admitted": 989,
            },
            "rows": [],
        }
    }
    assert any("99%" in failure for failure in contract.scientific_failures(short))

    exact = {
        "scientific": {
            "summary": {
                "geometrically_reachable": 1000,
                "native_admitted": 990,
            },
            "rows": [],
        }
    }
    assert contract.scientific_failures(exact) == []


def test_scientific_hash_is_canonical_and_detects_change() -> None:
    first = {"b": [2, 3], "a": "value"}
    second = {"a": "value", "b": [2, 3]}
    assert contract.scientific_sha256(first) == contract.scientific_sha256(second)
    payload = {
        "scientific": first,
        "scientific_sha256": contract.scientific_sha256(first),
    }
    contract.verify_scientific_hash(payload, "test")
    payload["scientific"]["a"] = "changed"
    with pytest.raises(RuntimeError, match="scientific hash mismatch"):
        contract.verify_scientific_hash(payload, "test")


def test_atomic_create_is_single_use_and_staging_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    monkeypatch.setattr(contract, "STAGING_DIR", staging)
    output = staging / "receipt.json"
    contract.atomic_create_json(output, {"status": "READY"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "READY"}
    with pytest.raises(FileExistsError, match="already exists"):
        contract.atomic_create_json(output, {"status": "CHANGED"})
    with pytest.raises(ValueError, match="must stay under"):
        contract.atomic_create_json(tmp_path / "outside.json", {})


def test_protocol_and_dependency_hashes_are_frozen() -> None:
    contract.verify_protocol()
    assert contract.verify_dependencies() == contract.DEPENDENCY_SHA256
    assert stable_json_sha256([922]) != contract.TARGET_POOL_SHA256


@pytest.mark.parametrize("relative", sorted(contract.DEPENDENCY_SHA256))
def test_every_eager_import_dependency_drift_fails(
    relative: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_require = contract.require_file_hash

    def reject_one(path: Path, expected: str, label: str) -> str:
        if path == contract.REPO_ROOT / relative:
            raise RuntimeError(f"simulated drift: {relative}")
        return real_require(path, expected, label)

    monkeypatch.setattr(contract, "require_file_hash", reject_one)
    with pytest.raises(RuntimeError, match="simulated drift"):
        contract.verify_dependencies()


@pytest.mark.parametrize(
    "field",
    [
        "attempt_sha256",
        "execution_receipt_sha256",
        "implementation_commit",
        "native_runtime",
    ],
)
def test_admission_provenance_rejects_every_chain_mutation(field: str) -> None:
    receipt = {
        "implementation_commit": "implementation",
        "native_runtime": dict(contract.EXPECTED_NATIVE_RUNTIME),
    }
    payload = {
        "attempt_sha256": "attempt",
        "execution_receipt_sha256": "execution",
        "implementation_commit": "implementation",
        "native_runtime": dict(contract.EXPECTED_NATIVE_RUNTIME),
    }
    if field == "native_runtime":
        payload[field] = {**payload[field], "cuda_matmul_allow_tf32": True}
    else:
        payload[field] = "mutated"
    with pytest.raises(RuntimeError, match="admission provenance mismatch"):
        contract.verify_admission_provenance(
            payload,
            receipt=receipt,
            execution_sha="execution",
            attempt_sha="attempt",
        )


@pytest.mark.parametrize(
    "field",
    [
        "attempt_sha256",
        "admission_sha256",
        "admission_scientific_sha256",
        "native_runtime",
    ],
)
def test_result_provenance_rejects_every_chain_mutation(field: str) -> None:
    receipt = {"native_runtime": dict(contract.EXPECTED_NATIVE_RUNTIME)}
    admission = {"scientific_sha256": "admission-science"}
    payload = {
        "attempt_sha256": "attempt",
        "admission_sha256": "admission",
        "admission_scientific_sha256": "admission-science",
        "native_runtime": dict(contract.EXPECTED_NATIVE_RUNTIME),
    }
    if field == "native_runtime":
        payload[field] = {**payload[field], "cuda_matmul_allow_tf32": True}
    else:
        payload[field] = "mutated"
    with pytest.raises(RuntimeError, match="result provenance mismatch"):
        contract.verify_result_provenance(
            payload,
            receipt=receipt,
            admission=admission,
            admission_sha="admission",
            attempt_sha="attempt",
        )


def test_admission_post_attempt_exception_is_durable_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _script_globals("admit_e7b_3b.py")
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
        globals_dict,
        "load_execution_receipt",
        lambda: (receipt, "execution"),
    )

    def fail_after_burn(**_kwargs: object) -> None:
        raise RuntimeError("known post-attempt failure")

    monkeypatch.setitem(globals_dict, "run_after_attempt", fail_after_burn)
    with pytest.raises(RuntimeError, match="known post-attempt failure"):
        namespace["main"]()
    assert paths["ATTEMPT_PATH"].is_file()
    invalid = json.loads(paths["ADMISSION_PATH"].read_text(encoding="utf-8"))
    assert invalid["status"] == "IMPLEMENTATION_INVALID"
    assert invalid["attempt_sha256"] == contract.sha256_file(paths["ATTEMPT_PATH"])


def test_admission_environment_mismatch_is_pre_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _script_globals("admit_e7b_3b.py")
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
    monkeypatch.setitem(globals_dict, "environment_record", lambda _device: {"drift": 1})
    monkeypatch.setitem(
        globals_dict,
        "load_execution_receipt",
        lambda: (receipt, "execution"),
    )
    with pytest.raises(RuntimeError, match="environment differs"):
        namespace["main"]()
    assert not paths["ATTEMPT_PATH"].exists()
    assert not paths["ADMISSION_PATH"].exists()


@pytest.mark.parametrize("failure_kind", ["environment", "runtime"])
def test_replay_post_admission_exception_is_durable_invalid(
    failure_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _script_globals("replay_e7b_3b.py")
    globals_dict = namespace["main"].__globals__
    paths = _patch_paths(globals_dict, tmp_path / f"replay-{failure_kind}", monkeypatch)
    receipt = _fake_receipt()
    admission = {"scientific_sha256": "admission-science", "scientific": {}}
    monkeypatch.setitem(
        globals_dict,
        "load_execution_receipt",
        lambda: (receipt, "execution"),
    )
    monkeypatch.setitem(
        globals_dict,
        "load_admission",
        lambda: (admission, "admission"),
    )
    monkeypatch.setitem(globals_dict, "load_attempt", lambda: ({}, "attempt"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    runtime = dict(contract.EXPECTED_NATIVE_RUNTIME)
    if failure_kind == "runtime":
        runtime["cuda_matmul_allow_tf32"] = True
    monkeypatch.setitem(globals_dict, "configure_native_runtime", lambda: runtime)
    environment = (
        {"drift": 1}
        if failure_kind == "environment"
        else dict(receipt["environment"])
    )
    monkeypatch.setitem(
        globals_dict,
        "environment_record",
        lambda _device: environment,
    )
    with pytest.raises(RuntimeError):
        namespace["main"]()
    invalid = json.loads(paths["RESULT_PATH"].read_text(encoding="utf-8"))
    assert invalid["status"] == "IMPLEMENTATION_INVALID"
    assert invalid["attempt_sha256"] == "attempt"
    assert invalid["admission_sha256"] == "admission"


def test_verifier_incomplete_and_invalid_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run_case(case: str, admission_error: bool) -> dict[str, object]:
        namespace = _script_globals("verify_e7b_3b.py")
        globals_dict = namespace["main"].__globals__
        paths = _patch_paths(globals_dict, tmp_path / case, monkeypatch)
        paths["ATTEMPT_PATH"].parent.mkdir(parents=True, exist_ok=True)
        paths["ATTEMPT_PATH"].write_text("{}", encoding="utf-8")
        paths["ADMISSION_PATH"].write_text("{}", encoding="utf-8")
        monkeypatch.setitem(globals_dict, "load_attempt", lambda: ({}, "attempt"))
        if admission_error:
            def invalid_admission() -> tuple[dict[str, object], str]:
                raise RuntimeError("known invalid admission")

            monkeypatch.setitem(globals_dict, "load_admission", invalid_admission)
        else:
            monkeypatch.setitem(
                globals_dict,
                "load_admission",
                lambda: ({"scientific_sha256": "science"}, "admission"),
            )
        exit_code = namespace["main"]()
        verdict = json.loads(paths["VERDICT_PATH"].read_text(encoding="utf-8"))
        verdict["exit_code"] = exit_code
        return verdict

    incomplete = run_case("incomplete", admission_error=False)
    assert incomplete["verdict"] == "INCOMPLETE"
    assert incomplete["exit_code"] == 3
    invalid = run_case("invalid", admission_error=True)
    assert invalid["verdict"] == "IMPLEMENTATION_INVALID"
    assert invalid["exit_code"] == 2


def test_invalid_result_claims_cannot_replace_live_verdict_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _script_globals("verify_e7b_3b.py")
    globals_dict = namespace["main"].__globals__
    paths = _patch_paths(globals_dict, tmp_path / "invalid-chain", monkeypatch)
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
    monkeypatch.setitem(
        globals_dict,
        "load_attempt",
        lambda: ({}, "live-attempt"),
    )
    monkeypatch.setitem(
        globals_dict,
        "load_admission",
        lambda: ({"scientific_sha256": "live-admission-science"}, "live-admission"),
    )

    def invalid_result() -> tuple[dict[str, object], str]:
        raise RuntimeError("invalid result provenance")

    monkeypatch.setitem(globals_dict, "load_result", invalid_result)
    assert namespace["main"]() == 2
    verdict = json.loads(paths["VERDICT_PATH"].read_text(encoding="utf-8"))
    assert verdict["verdict"] == "IMPLEMENTATION_INVALID"
    assert verdict["attempt_sha256"] == "live-attempt"
    assert verdict["admission_sha256"] == "live-admission"
    assert verdict["admission_scientific_sha256"] == "live-admission-science"
    assert verdict["result_sha256"] == contract.sha256_file(paths["RESULT_PATH"])
    assert verdict["invalid_result_claims"] == {
        "attempt_sha256": "bogus-attempt",
        "admission_sha256": "bogus-admission",
        "admission_scientific_sha256": "bogus-admission-science",
        "result_scientific_sha256": "bogus-result-science",
    }


def test_native_rows_preserve_full_reachable_batch_context_and_hashes() -> None:
    vocab = contract.MODEL_VOCAB_SIZE
    hidden_size = 4
    target_ids = list(range(100, 120))
    geometry: list[dict[str, object]] = []
    for index, target_id in enumerate(target_ids):
        reachable = index < 18
        geometry.append(
            {
                "pool_index": index,
                "target_id": target_id,
                "L": 0.0,
                "U": None,
                "hard_blocker": not reachable,
                "hard_blocker_id": 0 if not reachable else None,
                "beta": 1.0 if reachable else None,
                "float64_margin": 1.0 if reachable else None,
                "geometry_class": GEOMETRIC_REACHABLE if reachable else REFUSE_HARD,
                "decision": None if reachable else REFUSE_HARD,
            }
        )

    class FakeHead:
        def __init__(self) -> None:
            self.position = 0
            self.batch_sizes: list[int] = []
            self.rows: list[torch.Tensor] = []

        def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
            batch = hidden.shape[0]
            self.batch_sizes.append(batch)
            logits = torch.zeros((batch, vocab), dtype=torch.bfloat16)
            for offset in range(batch):
                position = self.position + offset
                target_id = target_ids[position]
                logits[offset, 1] = 1.0
                logits[offset, 2] = 1.0
                logits[offset, target_id] = 1.0 if position == 3 else 2.0
                self.rows.append(logits[offset].clone())
            self.position += batch
            return logits

    class FakeModel:
        def __init__(self) -> None:
            self.head = FakeHead()

        def get_output_embeddings(self) -> FakeHead:
            return self.head

    model = FakeModel()
    hidden = torch.zeros(hidden_size, dtype=torch.bfloat16)
    directions = torch.zeros((20, hidden_size), dtype=torch.float64)
    rows = native_rows(model, hidden, directions, deepcopy(geometry))
    assert model.head.batch_sizes == [16, 2]
    assert rows[3]["decision"] == REFUSE_TIE
    assert rows[3]["native_competitor_id"] == 1
    assert rows[16]["native_batch_start"] == 16
    assert rows[16]["native_batch_size"] == 2
    assert rows[17]["native_batch_offset"] == 1
    assert rows[4]["native_logits_shape"] == [vocab]
    assert rows[4]["native_logits_dtype"] == "bfloat16"
    assert rows[4]["native_logits_sha256"] == tensor_sha256(model.head.rows[4])
    assert sum(row.get("decision") == NATIVE_ADMIT for row in rows) == 17

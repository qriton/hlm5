from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch

import hlm5.e7c_contract as contract
from hlm5.e7_contract import REPO_ROOT, sha256_file
from hlm5.e7b_runtime import (
    GEOMETRIC_REACHABLE,
    NATIVE_ADMIT,
    REFUSE_HARD,
    geometry_rows,
)
from hlm5.e7c_runtime import (
    geometry_for_directions,
    normalize_rows,
    raw_and_zca_directions,
    zca_basis,
)


SCREEN = runpy.run_path(str(REPO_ROOT / "scripts" / "analyze_e7c_spent_geometry.py"))


def test_protocol_and_development_evidence_are_hash_pinned() -> None:
    assert sha256_file(REPO_ROOT / contract.PROTOCOL_PATH) == contract.PROTOCOL_SHA256
    assert (
        sha256_file(REPO_ROOT / contract.E7B_RESULT_PATH) == contract.E7B_RESULT_SHA256
    )
    assert (
        sha256_file(REPO_ROOT / contract.E7B_VERDICT_PATH)
        == contract.E7B_VERDICT_SHA256
    )
    assert (
        sha256_file(REPO_ROOT / contract.SPENT_SCREEN_PATH)
        == contract.SPENT_SCREEN_SHA256
    )
    evidence = contract.verify_development_evidence()
    assert evidence["selected_candidate"] == "zca_half_raw_r1e2"


def test_every_frozen_dependency_matches_its_registered_hash() -> None:
    assert contract.verify_dependencies() == contract.DEPENDENCY_SHA256


def test_registered_commands_are_exact_and_single_attempt() -> None:
    assert contract.REGISTERED_COMMANDS == (
        "python scripts/preflight_e7c_3b.py",
        "python scripts/register_e7c_3b.py",
        "python scripts/admit_e7c_3b.py",
        "python scripts/replay_e7c_3b.py",
        "python scripts/verify_e7c_3b.py",
    )
    assert len(set(contract.NAMED_OUTPUTS)) == 6


def test_normalize_rows_uses_the_frozen_denominator() -> None:
    rows = torch.tensor([[3.0, 4.0], [5.0, 12.0]], dtype=torch.float64)
    expected = rows / (rows.norm(dim=1, keepdim=True) + 1e-12)
    assert torch.equal(normalize_rows(rows), expected)


def test_zca_basis_exactly_matches_the_spent_screen_implementation() -> None:
    torch.manual_seed(123)
    head = torch.randn(40, 8, dtype=torch.float32)
    expected_mean, expected_values, expected_vectors, expected_ridge = SCREEN[
        "covariance_basis"
    ](head)
    mean, values, vectors, operator, ridge = zca_basis(head)
    assert torch.equal(mean, expected_mean)
    assert torch.equal(values, expected_values)
    assert torch.equal(vectors, expected_vectors)
    assert ridge == expected_ridge
    expected_operator = (
        vectors.to(torch.float64) * (values.to(torch.float64) + ridge).rsqrt()
    ) @ vectors.to(torch.float64).T
    assert torch.equal(operator, expected_operator)


def test_raw_and_zca_directions_match_the_frozen_selected_candidate() -> None:
    torch.manual_seed(456)
    head = torch.randn(48, 7, dtype=torch.float32)
    target_ids = torch.tensor([1, 7, 11, 31], dtype=torch.long)
    raw, candidate, record = raw_and_zca_directions(head, target_ids)
    mean, values, vectors, ridge = SCREEN["covariance_basis"](head)
    head64 = head.to(torch.float64)
    raw_expected = SCREEN["directions_for_candidate"](
        head64,
        target_ids,
        SCREEN["CANDIDATE_SPECS"][0],
        mean32=mean,
        eigenvalues32=values,
        eigenvectors32=vectors,
        ridge=ridge,
    )
    zca_spec = next(
        spec
        for spec in SCREEN["CANDIDATE_SPECS"]
        if spec["name"] == "zca_half_raw_r1e2"
    )
    candidate_expected = SCREEN["directions_for_candidate"](
        head64,
        target_ids,
        zca_spec,
        mean32=mean,
        eigenvalues32=values,
        eigenvectors32=vectors,
        ridge=ridge,
    )
    assert torch.equal(raw, raw_expected)
    assert torch.equal(candidate, candidate_expected)
    assert not torch.equal(raw, candidate)
    assert record["direction_shape"] == [4, 7]
    assert record["direction_dtype"] == "float64"


def test_raw_geometry_matches_the_unchanged_e7b_implementation() -> None:
    head64 = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-0.5, -0.25, 0.1],
        ],
        dtype=torch.float64,
    )
    hidden = torch.tensor([0.2, -0.1, 0.05], dtype=torch.bfloat16)
    targets = torch.tensor([0, 1, 2], dtype=torch.long)
    expected, directions, _ = geometry_rows(head64, hidden, targets)
    actual = geometry_for_directions(head64, hidden, targets, directions)
    fields = (
        "target_id",
        "L",
        "U",
        "beta",
        "float64_margin",
        "geometry_class",
        "hard_blocker",
        "hard_blocker_id",
    )
    assert [tuple(row.get(field) for field in fields) for row in actual] == [
        tuple(row.get(field) for field in fields) for row in expected
    ]


def test_scientific_bars_use_exact_integer_arithmetic() -> None:
    def result(raw: int, geometric: int, admitted: int, lost: int) -> dict:
        return {
            "scientific": {
                "summary": {
                    "raw": {"geometrically_reachable": raw},
                    "candidate": {
                        "geometrically_reachable": geometric,
                        "native_admitted": admitted,
                    },
                    "lost_raw_reachable_count": lost,
                }
            }
        }

    assert contract.scientific_failures(result(1008, 1188, 1188, 12)) == []
    assert contract.scientific_failures(result(1008, 1187, 1187, 12))
    assert contract.scientific_failures(result(1007, 1188, 1176, 12))
    assert contract.scientific_failures(result(1009, 1188, 1188, 12))
    assert contract.scientific_failures(result(1008, 1188, 1188, 13))


def test_dual_summary_binds_gain_and_raw_losses() -> None:
    raw = [
        {
            "target_id": 1,
            "geometry_class": GEOMETRIC_REACHABLE,
            "decision": NATIVE_ADMIT,
        },
        {
            "target_id": 2,
            "geometry_class": GEOMETRIC_REACHABLE,
            "decision": NATIVE_ADMIT,
        },
        {"target_id": 3, "geometry_class": REFUSE_HARD, "decision": REFUSE_HARD},
    ]
    candidate = [
        {
            "target_id": 1,
            "geometry_class": GEOMETRIC_REACHABLE,
            "decision": NATIVE_ADMIT,
        },
        {"target_id": 2, "geometry_class": REFUSE_HARD, "decision": REFUSE_HARD},
        {
            "target_id": 3,
            "geometry_class": GEOMETRIC_REACHABLE,
            "decision": NATIVE_ADMIT,
        },
    ]
    summary = contract.dual_summary(raw, candidate)
    assert summary["reachability_gain"] == 0
    assert summary["gained_reachable_target_ids"] == [3]
    assert summary["lost_raw_reachable_target_ids"] == [2]
    assert summary["lost_raw_reachable_count"] == 1


def test_atomic_create_is_staging_bound_and_never_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "e7c"
    monkeypatch.setattr(contract, "STAGING_DIR", staging)
    output = staging / "receipt.json"
    contract.atomic_create_json(output, {"a": 1})
    assert contract.load_json_object(output) == {"a": 1}
    with pytest.raises(FileExistsError):
        contract.atomic_create_json(output, {"a": 2})
    with pytest.raises(ValueError):
        contract.atomic_create_json(tmp_path / "outside.json", {"a": 3})


def test_expected_basis_rejects_direction_or_operator_drift() -> None:
    record = {
        **contract.EXPECTED_BASIS,
        "operator_sha256": "a" * 64,
        "raw_directions_sha256": "b" * 64,
        "zca_directions_sha256": "c" * 64,
        "operator_dtype": "float64",
        "operator_shape": [2048, 2048],
        "direction_dtype": "float64",
        "direction_shape": [1200, 2048],
    }
    contract._verify_basis(record)
    record["ridge"] = 0.1
    with pytest.raises(RuntimeError, match="ridge"):
        contract._verify_basis(record)


def test_admission_provenance_binds_the_registered_direction_hashes() -> None:
    basis = {
        **contract.EXPECTED_BASIS,
        "operator_sha256": "a" * 64,
        "raw_directions_sha256": "b" * 64,
        "zca_directions_sha256": "c" * 64,
        "operator_dtype": "float64",
        "operator_shape": [2048, 2048],
        "direction_dtype": "float64",
        "direction_shape": [1200, 2048],
    }
    receipt = {
        "implementation_commit": "deadbeef",
        "native_runtime": contract.EXPECTED_NATIVE_RUNTIME,
        "basis_and_directions": basis,
    }
    payload = {
        "attempt_sha256": "1" * 64,
        "execution_receipt_sha256": "2" * 64,
        "implementation_commit": "deadbeef",
        "native_runtime": contract.EXPECTED_NATIVE_RUNTIME,
        "scientific": {"basis_and_directions": basis},
    }
    contract.verify_admission_provenance(
        payload,
        receipt=receipt,
        execution_sha="2" * 64,
        attempt_sha="1" * 64,
    )
    payload["scientific"]["basis_and_directions"] = {**basis, "ridge": 0.2}
    with pytest.raises(RuntimeError, match="basis differs"):
        contract.verify_admission_provenance(
            payload,
            receipt=receipt,
            execution_sha="2" * 64,
            attempt_sha="1" * 64,
        )

"""Fast CPU contract tests for HLM5 E14."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import hlm5.e14_contract as contract
import hlm5.e14_runtime as runtime


def unit(rows: torch.Tensor) -> torch.Tensor:
    return rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-8)


def test_router_initialization_is_deterministic_identity_and_exact_size() -> None:
    first = runtime.ResidualRouter()
    second = runtime.ResidualRouter()
    query = unit(torch.randn(7, runtime.HIDDEN_SIZE))
    assert torch.equal(first.a, second.a)
    assert torch.equal(first.b, second.b)
    assert torch.allclose(first(query), query, atol=1e-7, rtol=0)
    assert runtime.router_record(first)["parameter_count"] == 262_144


@pytest.mark.parametrize("arm", [runtime.HLM, runtime.COSINE])
def test_matched_router_training_is_finite_and_learns_synthetic_slots(arm: str) -> None:
    generator = torch.Generator().manual_seed(71)
    exact = unit(torch.randn(runtime.TRAIN_COUNT, 32, generator=generator))
    paraphrase = unit(exact + 0.08 * torch.randn(exact.shape, generator=generator))
    router, record = runtime.train_router(arm, exact, paraphrase)
    telemetry = record["telemetry"]
    assert [row["step"] for row in telemetry] == list(runtime.TELEMETRY_STEPS)
    assert telemetry[-1]["own_slot_accuracy"] == runtime.TRAIN_COUNT
    assert telemetry[-1]["parameter_update_norm"] > 0
    assert telemetry[0]["gradient_norm"] is not None
    assert all(torch.isfinite(parameter).all() for parameter in router.parameters())


def test_cosine_and_hlm_training_share_exact_initial_state() -> None:
    first = runtime.ResidualRouter(32)
    second = runtime.ResidualRouter(32)
    assert torch.equal(first.a, second.a)
    assert torch.equal(first.b, second.b)


@pytest.mark.parametrize("arm", runtime.ARMS)
def test_routed_memory_exact_gate_locality_and_rollback(arm: str) -> None:
    dim = 8
    keys = torch.eye(dim)[:4]
    values = torch.flip(keys, dims=[1])
    router = None if arm == runtime.IDENTITY else runtime.ResidualRouter(dim, 2)
    memory = runtime.RoutedMemory(
        arm=arm,
        router=router,
        key_mean=torch.zeros(dim),
        key_transform=torch.eye(dim),
        base_keys=keys,
        values=values,
    )
    changed, audit = memory.apply(keys)
    assert int(audit["gate_open"].sum()) == 4
    assert torch.equal(audit["selected_slot"], torch.arange(4))
    assert int(torch.count_nonzero(changed - keys)) > 0
    locality = -keys
    unchanged, local_audit = memory.apply(locality)
    assert int(local_audit["gate_open"].sum()) == 0
    assert torch.equal(unchanged, locality)
    rollback = runtime.rollback_scan(memory, keys, locality)
    assert rollback["exact_bit_identical"]
    assert rollback["locality_bit_identical"]
    assert rollback["exact_nonzero_delta"] == 0
    assert rollback["locality_nonzero_delta"] == 0


def test_bundle_roundtrip_binds_every_tensor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract, "STAGING_DIR", tmp_path)
    routers = {
        runtime.HLM: runtime.ResidualRouter(8, 2),
        runtime.COSINE: runtime.ResidualRouter(8, 2),
    }
    payload = runtime.bundle_tensors(routers, torch.zeros(8), torch.eye(8))
    manifest = runtime.bundle_manifest(payload)
    path = tmp_path / "routers.pt"
    digest = contract.atomic_create_bundle(path, payload)
    assert len(digest) == 64
    restored = contract.load_bundle(path, manifest)
    assert all(torch.equal(restored[name], payload[name]) for name in payload)
    with pytest.raises(FileExistsError):
        contract.atomic_create_bundle(path, payload)


def test_atomic_json_is_create_new_and_confined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract, "STAGING_DIR", tmp_path)
    path = tmp_path / "receipt.json"
    contract.atomic_create_json(path, {"b": 2, "a": 1})
    assert json.loads(path.read_text()) == {"a": 1, "b": 2}
    with pytest.raises(FileExistsError):
        contract.atomic_create_json(path, {})
    with pytest.raises(ValueError):
        contract.atomic_create_json(tmp_path.parent / "escape.json", {})


def test_model_parameter_hash_is_stable_and_detects_mutation() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    before = runtime.model_parameter_sha256(model)
    assert runtime.model_parameter_sha256(model) == before
    with torch.no_grad():
        model[0].weight[0, 0] += 1
    assert runtime.model_parameter_sha256(model) != before


def test_fact_and_locality_records_bind_order_and_fields() -> None:
    facts = [
        {
            "case_id": 1,
            "exact": "a",
            "paraphrase": "b",
            "target": " c",
            "target_id": 3,
            "relation_id": "r",
            "subject": "s",
        }
    ]
    locality = [{"case_id": 2, "exact": "d", "paraphrase": "e", "neighborhood": "f"}]
    fact = runtime.fact_record(facts)
    local = runtime.locality_record(locality)
    assert fact["count"] == 1
    assert local["prompt_count"] == 3
    changed = [dict(facts[0], exact="different")]
    assert runtime.fact_record(changed)["row_sha256"] != fact["row_sha256"]


def test_scientific_summary_preserves_all_binding_counts() -> None:
    rollback = {
        "exact_count": 1024,
        "locality_count": 2994,
        "exact_gate_open_count": 0,
        "locality_gate_open_count": 0,
        "exact_bit_identical": True,
        "locality_bit_identical": True,
        "exact_nonzero_delta": 0,
        "locality_nonzero_delta": 0,
    }
    arms = {}
    training = {}
    for arm in runtime.ARMS:
        summary = {
            "gate_open_count": 1024,
            "own_slot_count": 1024,
            "strict_target_success_count": 1024,
            "nonzero_delta_count": 1024,
        }
        arms[arm] = {
            "exact": {"summary": summary},
            "paraphrase": {"summary": summary},
            "locality": {
                "summary": {
                    "gate_open_count": 0,
                    "nonzero_delta_count": 0,
                    "changed_hidden_count": 0,
                }
            },
            "rollback": rollback,
        }
        if arm != runtime.IDENTITY:
            training[arm] = {"telemetry": [{"own_slot_accuracy": 256}]}
    measurement = {
        "evaluation": {
            "direct": {"admitted_count": 1100, "selected_count": 1024},
            "arms": arms,
        },
        "training": training,
        "trunk_parameter_sha256_before": "a" * 64,
        "trunk_parameter_sha256_after": "a" * 64,
    }
    summary = runtime.scientific_summary(measurement)
    assert summary["trunk_unchanged"]
    assert summary["arms"][runtime.HLM]["train_correct"] == 256
    assert summary["arms"][runtime.HLM]["paraphrase_strict"] == 1024


def test_environment_failure_detects_node_or_runtime_drift() -> None:
    valid = dict(contract.EXPECTED_ENVIRONMENT)
    valid["node"] = "node-a"
    assert contract.environment_failures(valid, node="node-a") == []
    assert contract.environment_failures(valid, node="node-b") == [
        "E14 node differs from registration"
    ]


def _valid_measurement() -> dict:
    selected = list(range(runtime.ACTIVE_COUNT))
    direct_rows = [
        {
            "case_index": index,
            "baseline_correct": index >= runtime.ACTIVE_COUNT,
            "eligible": index < runtime.ACTIVE_COUNT,
            "decision": "ADMIT_NATIVE" if index < runtime.ACTIVE_COUNT else None,
            "deployment_admitted": index < runtime.ACTIVE_COUNT,
        }
        for index in range(runtime.EVAL_CANDIDATE_COUNT)
    ]
    telemetry = [
        {
            "step": step,
            "loss": 1.0,
            "cross_entropy": 1.0,
            "identity_displacement_mse": 0.0,
            "gradient_norm": 1.0,
            "parameter_update_norm": 0.0 if step == 0 else 1.0,
            "own_slot_accuracy": runtime.TRAIN_COUNT,
        }
        for step in runtime.TELEMETRY_STEPS
    ]
    bundle = {
        "hlm_a": {"sha256": "1" * 64},
        "hlm_b": {"sha256": "2" * 64},
        "cosine_a": {"sha256": "3" * 64},
        "cosine_b": {"sha256": "4" * 64},
        "key_mean": {"sha256": "5" * 64},
        "key_transform": {"sha256": "6" * 64},
    }
    training = {
        runtime.HLM: {
            "telemetry": telemetry,
            "router": {"a_sha256": "1" * 64, "b_sha256": "2" * 64},
        },
        runtime.COSINE: {
            "telemetry": telemetry,
            "router": {"a_sha256": "3" * 64, "b_sha256": "4" * 64},
        },
    }
    fact_rows = [
        {
            "candidate_index": index,
            "own_slot": index,
            "selected_slot": index,
            "gate_open": True,
            "strict_target_success": True,
            "delta_zero": False,
            "target_margin": 1.0,
        }
        for index in selected
    ]
    fact_summary = {
        "count": runtime.ACTIVE_COUNT,
        "gate_open_count": runtime.ACTIVE_COUNT,
        "own_slot_count": runtime.ACTIVE_COUNT,
        "strict_target_success_count": runtime.ACTIVE_COUNT,
        "nonzero_delta_count": runtime.ACTIVE_COUNT,
        "minimum_margin": 1.0,
    }
    local_rows = [
        {
            "pool_index": index,
            "kind": runtime.KINDS[index % 3],
            "gate_open": False,
            "delta_zero": True,
            "hidden_bit_identical": True,
        }
        for index in range(runtime.LOCALITY_COUNT)
    ]
    local_summary = {
        "count": runtime.LOCALITY_COUNT,
        "gate_open_count": 0,
        "nonzero_delta_count": 0,
        "changed_hidden_count": 0,
        "by_kind": {
            kind: {
                "count": runtime.LOCALITY_CASE_COUNT,
                "gate_open_count": 0,
                "nonzero_delta_count": 0,
                "changed_hidden_count": 0,
            }
            for kind in runtime.KINDS
        },
    }
    rollback = {
        "exact_count": runtime.ACTIVE_COUNT,
        "locality_count": runtime.LOCALITY_COUNT,
        "exact_gate_open_count": 0,
        "locality_gate_open_count": 0,
        "exact_bit_identical": True,
        "locality_bit_identical": True,
        "exact_nonzero_delta": 0,
        "locality_nonzero_delta": 0,
    }
    arms = {
        arm: {
            "memory": {
                "physical_slots": runtime.MEMORY_SIZE,
                "active_slots": runtime.ACTIVE_COUNT,
            },
            "exact": {"rows": fact_rows, "summary": fact_summary},
            "paraphrase": {"rows": fact_rows, "summary": fact_summary},
            "locality": {"rows": local_rows, "summary": local_summary},
            "rollback": rollback,
        }
        for arm in runtime.ARMS
    }
    return {
        "population": {
            "train": contract.TRAIN_EXPECTED,
            "evaluation": contract.EVAL_EXPECTED,
            "locality": contract.LOCALITY_EXPECTED,
        },
        "key_whitening": {
            "fit_count": runtime.TRAIN_COUNT + runtime.EVAL_CANDIDATE_COUNT + 18,
            "mean_sha256": "5" * 64,
            "transform_sha256": "6" * 64,
        },
        "training": training,
        "bundle": {"tensors": bundle},
        "evaluation": {
            "direct": {
                "rows": direct_rows,
                "admitted_count": runtime.ACTIVE_COUNT,
                "selected_count": runtime.ACTIVE_COUNT,
                "selected_indices": selected,
                "selected_indices_sha256": contract.stable_json_sha256(selected),
            },
            "arms": arms,
        },
        "trunk_parameter_sha256_before": "a" * 64,
        "trunk_parameter_sha256_after": "a" * 64,
    }


def test_measurement_validator_recomputes_binding_summaries() -> None:
    measurement = _valid_measurement()
    assert contract.measurement_failures(measurement) == []
    measurement["evaluation"]["arms"][runtime.HLM]["locality"]["summary"][
        "gate_open_count"
    ] = 1
    assert "E14 hlm_degree5 locality summary mismatch" in contract.measurement_failures(
        measurement
    )

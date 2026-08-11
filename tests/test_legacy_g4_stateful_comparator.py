"""CPU contracts for the distributed state-complete checkpoint comparator."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
COMPARATOR = (
    ROOT / "research" / "legacy_g4_recovery" / "compare_stateful_shards.py"
)


def load_comparator():
    spec = importlib.util.spec_from_file_location("legacy_g4_comparator_test", COMPARATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def complete_payload(rank: int = 0, world: int = 1, step: int = 9) -> dict:
    return {
        "model": {"weight": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)},
        "opt": {
            "state": {0: {"step": torch.tensor(10), "exp_avg": torch.ones(4)}},
            "param_groups": [{"params": [0], "lr": 2e-4}],
        },
        "runtime_state": {
            "schema": "hlm5-g4-state-complete-v1",
            "sample_generator_state": torch.arange(8, dtype=torch.uint8),
            "torch_cpu_rng_state": torch.arange(9, dtype=torch.uint8),
            "torch_cuda_rng_state": torch.arange(10, dtype=torch.uint8),
        },
        "sharded": True,
        "rank": rank,
        "world_size": world,
        "step": step,
        "checkpoint_schema": "hlm5-g4-state-complete-v1",
        "optimizer_state_saved": True,
        "runtime_state_saved": True,
        "best": 15.5,
    }


def test_exact_payload_has_matching_semantic_digests() -> None:
    module = load_comparator()
    left = complete_payload()
    right = complete_payload()
    comparison = module.Comparison()
    comparison.compare(left, right)
    result = comparison.result()
    assert result["exact"] is True
    assert result["mismatch_count"] == 0
    assert result["left_semantic_sha256"] == result["right_semantic_sha256"]


def test_tensor_and_rng_drift_are_detected_with_paths() -> None:
    module = load_comparator()
    left = complete_payload()
    right = complete_payload()
    right["model"]["weight"][2, 3] += 1
    right["runtime_state"]["torch_cuda_rng_state"][4] += 1
    comparison = module.Comparison()
    comparison.compare(left, right)
    result = comparison.result()
    assert result["exact"] is False
    assert result["mismatch_count"] == 2
    paths = {item["path"] for item in result["mismatches"]}
    assert "$['model']['weight']" in paths
    assert "$['runtime_state']['torch_cuda_rng_state']" in paths


def test_mapping_order_is_semantically_irrelevant_but_type_is_not() -> None:
    module = load_comparator()
    left = {"a": 1, "b": 2}
    right = {"b": 2, "a": 1}
    comparison = module.Comparison()
    comparison.compare(left, right)
    assert comparison.result()["exact"] is True

    typed = module.Comparison()
    typed.compare(left, OrderedDict(left))
    assert typed.result()["exact"] is False


def test_payload_validation_requires_complete_rank_local_state() -> None:
    module = load_comparator()
    payload = complete_payload(rank=3, world=96, step=522499)
    module.validate_payload(payload, rank=3, world=96, expected_step=522499)
    del payload["runtime_state"]["torch_cuda_rng_state"]
    try:
        module.validate_payload(payload, rank=3, world=96, expected_step=522499)
    except RuntimeError as exc:
        assert "torch_cuda_rng_state" in str(exc)
    else:  # pragma: no cover - explicit fail-closed assertion
        raise AssertionError("missing CUDA RNG state was accepted")


def test_comparator_uses_original_world_process_group_and_create_new_reports() -> None:
    text = COMPARATOR.read_text(encoding="utf-8")
    assert 'dist.init_process_group(backend="nccl"' in text
    assert 'f"shard_rank{rank}.pt"' in text
    assert 'with path.open("x"' in text
    assert "torch.equal(left, right)" in text
    assert "local_shards()" in text
    assert '"model", "opt", "runtime_state"' in text
    assert '"PASS_EXACT"' in text
    assert '"FAIL_MISMATCH"' in text
    assert '"IMPLEMENTATION_INVALID"' in text


def test_world_one_cli_writes_exact_and_mismatch_aggregate_reports(tmp_path: Path) -> None:
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    left = complete_payload()
    right = complete_payload()
    torch.save(left, left_dir / "shard_rank0.pt")
    torch.save(right, right_dir / "shard_rank0.pt")

    exact_report = tmp_path / "exact-report"
    command = [
        sys.executable,
        str(COMPARATOR),
        "--left",
        str(left_dir),
        "--right",
        str(right_dir),
        "--report-dir",
        str(exact_report),
        "--label",
        "cpu_exact",
        "--expected-world",
        "1",
        "--expected-step",
        "9",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "STATEFUL_COMPARISON_PASS_EXACT" in completed.stdout
    exact = json.loads((exact_report / "aggregate.json").read_text(encoding="utf-8"))
    assert exact["status"] == "PASS_EXACT"
    assert exact["mismatching_ranks"] == []

    right["opt"]["state"][0]["exp_avg"][0] = 3
    torch.save(right, right_dir / "shard_rank0.pt")
    mismatch_report = tmp_path / "mismatch-report"
    mismatch_command = command.copy()
    mismatch_command[mismatch_command.index(str(exact_report))] = str(mismatch_report)
    mismatch = subprocess.run(mismatch_command, check=False, capture_output=True, text=True)
    assert mismatch.returncode == 1
    aggregate = json.loads(
        (mismatch_report / "aggregate.json").read_text(encoding="utf-8")
    )
    assert aggregate["status"] == "FAIL_MISMATCH"
    assert aggregate["mismatching_ranks"] == [0]

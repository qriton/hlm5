"""CPU and static contracts for the R2d one-update determinism diagnostic."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "research" / "legacy_g4_recovery" / "probe_one_update_determinism.py"
COMPARATOR = ROOT / "research" / "legacy_g4_recovery" / "compare_one_update_reports.py"
LAUNCHER = ROOT / "slurm" / "legacy_g4_hybrid_522k_one_update_determinism.slurm"
PROTOCOL = ROOT / "docs" / "legacy-g4-hybrid-r2d-one-update-determinism-contract-2026-08-11.md"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_semantic_hash_is_exact_and_mapping_order_independent() -> None:
    probe = load_module(PROBE, "legacy_g4_r2d_probe_test")
    left = {"b": [torch.arange(4, dtype=torch.bfloat16)], "a": 2}
    right = {"a": 2, "b": [torch.arange(4, dtype=torch.bfloat16)]}
    assert probe.semantic_sha256(left) == probe.semantic_sha256(right)
    right["b"][0][3] += 1
    assert probe.semantic_sha256(left) != probe.semantic_sha256(right)


def test_scheduled_lr_uses_the_original_610k_horizon() -> None:
    probe = load_module(PROBE, "legacy_g4_r2d_lr_test")
    expected = 2e-4 * 0.5 * (
        1
        + __import__("math").cos(
            __import__("math").pi * (522000 - 2000) / (610000 - 2000)
        )
    )
    assert probe.scheduled_lr(522000) == expected


def report(rank: int = 0) -> dict:
    stages = {
        "loaded": {"model_sha256": "m0", "optimizer_sha256": "o0", "runtime": "r0"},
        "sampled": {"x_sha256": "x", "y_sha256": "y", "runtime": "r1"},
        "forward": {"loss_sha256": "l", "logits_probe_sha256": "p", "runtime": "r2"},
        "backward": {"gradient_sha256": "g", "runtime": "r3"},
        "clipped": {"gradient_sha256": "gc", "gradient_norm": 1.0, "runtime": "r4"},
        "stepped": {"model_sha256": "m1", "optimizer_sha256": "o1", "runtime": "r5"},
    }
    return {
        "schema": "hlm5-g4-one-update-determinism-rank-v1",
        "run_label": "a",
        "rank": rank,
        "world_size": 96,
        "local_rank": rank % 4,
        "hostname": f"node{rank // 4:02d}",
        "device_name": "A100",
        "torch": "2.0",
        "cuda": "12.1",
        "cudnn": 8907,
        "source_step": 522000,
        "scheduled_lr": 1e-5,
        "stages": stages,
    }


def test_classifier_selects_the_earliest_divergent_stage() -> None:
    comparator = load_module(COMPARATOR, "legacy_g4_r2d_compare_test")
    left = [report(0)]
    right = [report(0)]
    right[0]["stages"]["backward"]["gradient_sha256"] = "different"
    right[0]["stages"]["stepped"]["model_sha256"] = "also-different"
    result = comparator.classify(left, right)
    assert result["status"] == "VALID_DIAGNOSIS"
    assert result["classification"] == "BACKWARD_REDUCTION_DIVERGENCE"
    assert result["first_divergent_stage"] == "backward"


def test_classifier_rejects_topology_drift_before_arithmetic_claim() -> None:
    comparator = load_module(COMPARATOR, "legacy_g4_r2d_topology_test")
    left = [report(0)]
    right = [report(0)]
    right[0]["hostname"] = "other-node"
    result = comparator.classify(left, right)
    assert result["status"] == "IMPLEMENTATION_INVALID"
    assert result["classification"] == "TOPOLOGY_MISMATCH"


def write_run(directory: Path, label: str, world: int = 2) -> None:
    directory.mkdir()
    for rank in range(world):
        item = report(rank)
        item["run_label"] = label
        item["world_size"] = world
        (directory / f"rank{rank:03d}.json").write_text(
            json.dumps(item, sort_keys=True) + "\n", encoding="utf-8"
        )
    (directory / "complete.json").write_text(
        json.dumps({"rank_reports": world}) + "\n", encoding="utf-8"
    )


def test_comparator_cli_writes_a_create_new_valid_diagnosis(tmp_path: Path) -> None:
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    write_run(run_a, "a")
    write_run(run_b, "b")
    result_path = tmp_path / "result.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(COMPARATOR),
            "--run-a",
            str(run_a),
            "--run-b",
            str(run_b),
            "--result",
            str(result_path),
            "--expected-world",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["classification"] == "ONE_UPDATE_EXACT"
    assert "ONE_UPDATE_DIAGNOSIS" in completed.stdout


def test_probe_executes_one_step_and_never_saves_a_checkpoint() -> None:
    text = PROBE.read_text(encoding="utf-8")
    assert text.count("optimizer.step()") == 1
    assert "sample_batch(train_buffer, 1, CONTEXT" in text
    assert 'stages["loaded"]' in text
    assert 'stages["sampled"]' in text
    assert 'stages["forward"]' in text
    assert 'stages["backward"]' in text
    assert 'stages["clipped"]' in text
    assert 'stages["stepped"]' in text
    assert "torch.save" not in text
    assert "save_fsdp_sharded_checkpoint" not in text


def test_launcher_is_bounded_single_use_and_pins_current_tools() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    probe_sha = hashlib.sha256(PROBE.read_bytes()).hexdigest()
    comparator_sha = hashlib.sha256(COMPARATOR.read_bytes()).hexdigest()
    assert "#SBATCH --time=00:30:00" in text
    assert "R2D_REGISTERED_LAUNCHER_SHA256_REQUIRED" in text
    assert "R2D_REGISTERED_PROTOCOL_SHA256_REQUIRED" in text
    assert "R2D_WORK_ROOT_ALREADY_EXISTS" in text
    assert f"probe_sha_expected={probe_sha}" in text
    assert f"comparator_sha_expected={comparator_sha}" in text
    assert text.index('create_new_json "$attempt"') < text.index(
        'run_probe a "$run_a"'
    )
    assert 'run_probe a "$run_a"' in text
    assert 'run_probe b "$run_b"' in text
    assert "SOURCE_CHANGED_DURING_R2D" in text


def test_protocol_preserves_the_exact_r2c_bar_and_paid_boundary() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "No" in text and "R2d outcome has been accessed" in text
    assert "does not rerun R2c, change its bar" in text
    assert "may not be weakened" in text
    assert "may not be submitted without a new explicit user approval" in text

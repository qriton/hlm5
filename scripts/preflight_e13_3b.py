"""Outcome-blind preflight for registered HLM5 E13."""

# ruff: noqa: E402
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from hlm5.e7_contract import (
    MODEL_FILE_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    snapshot_path,
    verify_snapshot,
)
from hlm5.e7_runtime import (
    assert_model_invariants,
    load_pinned_model,
    load_pinned_tokenizer,
)
from hlm5.e7b_runtime import configure_native_runtime
from hlm5.e13_contract import (
    DEPENDENCY_SHA256,
    NAMED_OUTPUTS,
    PREFLIGHT_PATH,
    PREFLIGHT_SCHEMA,
    PROTOCOL_COMMIT,
    PROTOCOL_SHA256,
    REGISTERED_COMMANDS,
    assert_outputs_absent,
    atomic_create_json,
    environment_failures,
    git_output,
    registered_source_sha256,
    selected_population,
    verify_bound_inputs,
    verify_dependencies,
    verify_protocol,
)
from hlm5.e13_runtime import (
    BASELINE,
    BASELINE_TIERS,
    CANDIDATE,
    CANDIDATE_TIERS,
    GATE_BATCH,
    MEMORY_SIZE,
    MODEL_FORWARD_BATCH,
    e13_environment_record,
    paired_directions,
)


TEST_COMMAND = [sys.executable, "-m", "pytest", "tests/test_e13_3b.py", "-q"]


def main() -> None:
    assert_outputs_absent(NAMED_OUTPUTS)
    torch.manual_seed(0)
    native_runtime = configure_native_runtime()
    verify_protocol()
    verify_dependencies()
    bound_inputs = verify_bound_inputs()
    source_sha256 = registered_source_sha256(require_clean=True)
    implementation_commit = git_output("rev-parse", "HEAD")
    tracked = set(git_output("ls-files", *source_sha256).splitlines())
    if tracked != set(source_sha256):
        raise RuntimeError("every registered E13 source must be tracked")
    tests = subprocess.run(TEST_COMMAND, cwd=ROOT, capture_output=True, text=True)
    if tests.returncode:
        raise RuntimeError(
            "registered E13 tests failed:\n" + tests.stdout + tests.stderr
        )
    snapshot = snapshot_path()
    model_files = verify_snapshot(snapshot)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("registered E13 preflight requires exactly one GPU")
    device = torch.device("cuda", 0)
    tokenizer = load_pinned_tokenizer(snapshot)
    population = selected_population(tokenizer)
    model = load_pinned_model(snapshot, device)
    invariants = assert_model_invariants(model)
    target_ids = torch.tensor(
        [row["target_id"] for row in population["cases"]],
        dtype=torch.long,
        device=device,
    )
    _directions, basis = paired_directions(
        model.get_output_embeddings().weight.detach(), target_ids
    )
    environment = e13_environment_record(device)
    failures = environment_failures(environment)
    if failures:
        raise RuntimeError(
            "registered E13 environment mismatch: " + "; ".join(failures)
        )
    payload = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": implementation_commit,
        "branch": git_output("branch", "--show-current"),
        "source_sha256": source_sha256,
        "dependency_sha256": dict(DEPENDENCY_SHA256),
        "bound_inputs": bound_inputs,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_file_sha256": dict(MODEL_FILE_SHA256),
        "model_files": model_files,
        "snapshot_selector": "HLM5_E7_SNAPSHOT"
        if os.environ.get("HLM5_E7_SNAPSHOT")
        else "HF_HOME",
        "candidate_record": population["record"],
        "query_record": population["query_record"],
        "baseline": BASELINE,
        "candidate": CANDIDATE,
        "baseline_tiers": list(BASELINE_TIERS),
        "candidate_tiers": list(CANDIDATE_TIERS),
        "memory_size": MEMORY_SIZE,
        "model_forward_batch": MODEL_FORWARD_BATCH,
        "gate_batch": GATE_BATCH,
        "basis_and_directions": basis,
        "model_invariants": invariants,
        "environment": environment,
        "native_runtime": native_runtime,
        "test_command": "python -m pytest tests/test_e13_3b.py -q",
        "tests_returncode": tests.returncode,
        "tests_stdout": tests.stdout.strip(),
        "tests_stderr": tests.stderr.strip(),
        "e13_outcomes_computed": False,
        "population_materialized_without_model": len(population["cases"]),
        "commands": list(REGISTERED_COMMANDS),
    }
    atomic_create_json(PREFLIGHT_PATH, payload)
    print(f"READY preflight: {PREFLIGHT_PATH}")


if __name__ == "__main__":
    main()

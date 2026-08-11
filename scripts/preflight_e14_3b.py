"""Outcome-blind preflight for the single-use HLM5 E14 run."""

# ruff: noqa: E402
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from hlm5.e7_contract import snapshot_path
from hlm5.e7_runtime import assert_model_invariants, load_pinned_model, load_pinned_tokenizer
from hlm5.e7b_runtime import configure_native_runtime
from hlm5.e14_contract import (
    DEPENDENCY_PATHS,
    EVAL_EXPECTED,
    LOCALITY_EXPECTED,
    NAMED_OUTPUTS,
    PREFLIGHT_PATH,
    PREFLIGHT_SCHEMA,
    PROTOCOL_COMMIT,
    PROTOCOL_SHA256,
    REGISTERED_COMMANDS,
    REGISTERED_SOURCE_PATHS,
    TEST_COMMAND,
    TRAIN_EXPECTED,
    assert_outputs_absent,
    atomic_create_json,
    git_output,
    registered_hashes,
    selected_population,
    training_config,
    verify_bound_inputs,
    verify_protocol,
)
from hlm5.e14_runtime import ResidualRouter, environment_record, router_record


def main() -> None:
    started = time.time()
    assert_outputs_absent(NAMED_OUTPUTS)
    verify_protocol()
    bound = verify_bound_inputs()
    implementation_commit = git_output("rev-parse", "HEAD")
    source_hashes = registered_hashes(REGISTERED_SOURCE_PATHS, require_clean=True)
    dependency_hashes = registered_hashes(DEPENDENCY_PATHS, require_clean=True)
    completed = subprocess.run(TEST_COMMAND.split(), cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError("E14 targeted tests failed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("E14 preflight requires exactly one GPU")
    device = torch.device("cuda", 0)
    native_runtime = configure_native_runtime()
    tokenizer = load_pinned_tokenizer(snapshot_path())
    population = selected_population(tokenizer)
    model = load_pinned_model(snapshot_path(), device)
    router = ResidualRouter().to(device)
    payload = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": implementation_commit,
        "source_sha256": source_hashes,
        "dependency_sha256": dependency_hashes,
        "model_invariants": assert_model_invariants(model),
        "environment": environment_record(device),
        "native_runtime": native_runtime,
        "train_population": population["train_record"],
        "evaluation_population": population["evaluation_record"],
        "locality_population": population["locality_record"],
        "training_config": training_config(),
        "router_initialization": router_record(router),
        "registered_commands": list(REGISTERED_COMMANDS),
        "test_command": TEST_COMMAND,
        "tests_returncode": completed.returncode,
        "outcomes_computed": False,
        "bound_inputs": bound,
        "runtime_seconds": time.time() - started,
    }
    if payload["train_population"] != TRAIN_EXPECTED:
        raise RuntimeError("E14 train population changed during preflight")
    if payload["evaluation_population"] != EVAL_EXPECTED:
        raise RuntimeError("E14 evaluation population changed during preflight")
    if payload["locality_population"] != LOCALITY_EXPECTED:
        raise RuntimeError("E14 locality population changed during preflight")
    atomic_create_json(PREFLIGHT_PATH, payload)
    print(f"READY preflight: {PREFLIGHT_PATH}")


if __name__ == "__main__":
    main()

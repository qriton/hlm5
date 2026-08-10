"""Outcome-blind preflight for the registered E9 wide-locality scan."""

# ruff: noqa: E402 -- direct script execution bootstraps the repository root.
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

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
    final_hidden_batch,
    load_pinned_model,
    load_pinned_tokenizer,
)
from hlm5.e7b_runtime import configure_native_runtime, environment_record
from hlm5.e8_contract import selected_population as selected_e8_population
from hlm5.e9_contract import (
    DEPENDENCY_SHA256,
    EXPECTED_ENVIRONMENT,
    GATE_BATCH,
    MODEL_FORWARD_BATCH,
    NAMED_OUTPUTS,
    PREFLIGHT_PATH,
    PREFLIGHT_SCHEMA,
    PROTOCOL_COMMIT,
    PROTOCOL_SHA256,
    QUERY_CASE_COUNT,
    QUERY_CASE_ID_SHA256,
    QUERY_CASE_START,
    QUERY_CASE_STOP,
    QUERY_COUNT,
    QUERY_KIND_SHA256,
    QUERY_PROMPT_SHA256,
    QUERY_ROW_SHA256,
    REGISTERED_COMMANDS,
    SEED,
    assert_outputs_absent,
    atomic_create_json,
    frozen_e8_measurement,
    git_output,
    registered_source_sha256,
    selected_population,
    verify_bound_inputs,
    verify_dependencies,
    verify_protocol,
)
from hlm5.e9_runtime import prepare_e8_adapter


TEST_COMMAND = [sys.executable, "-m", "pytest", "tests/test_e9_3b.py", "-q"]


def main() -> None:
    assert_outputs_absent(NAMED_OUTPUTS)
    torch.manual_seed(SEED)
    native_runtime = configure_native_runtime()
    verify_protocol()
    verify_dependencies()
    bound_inputs = verify_bound_inputs()
    query_population = selected_population()
    source_sha256 = registered_source_sha256(require_clean=True)
    implementation_commit = git_output("rev-parse", "HEAD")
    tracked = set(git_output("ls-files", *source_sha256).splitlines())
    if tracked != set(source_sha256):
        raise RuntimeError("every registered E9 source must be tracked by git")

    snapshot = snapshot_path()
    model_files = verify_snapshot(snapshot)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("registered E9 preflight requires exactly one visible GPU")
    device = torch.device("cuda", 0)

    tests = subprocess.run(
        TEST_COMMAND,
        cwd=REPO_ROOT_BOOTSTRAP,
        capture_output=True,
        text=True,
        check=False,
    )
    if tests.returncode != 0:
        raise RuntimeError(
            "registered E9 tests failed before preflight:\n"
            + tests.stdout
            + tests.stderr
        )

    tokenizer = load_pinned_tokenizer(snapshot)
    e8_population = selected_e8_population(tokenizer)
    model = load_pinned_model(snapshot, device)
    invariants = assert_model_invariants(model)
    e8_hiddens = final_hidden_batch(
        model,
        tokenizer,
        e8_population["prompt_order"],
        device,
        batch_size=MODEL_FORWARD_BATCH,
    )
    e8_hidden_by_prompt = {
        prompt: e8_hiddens[index]
        for index, prompt in enumerate(e8_population["prompt_order"])
    }
    frozen = frozen_e8_measurement()
    reconstruction = prepare_e8_adapter(
        model, e8_population, frozen, e8_hidden_by_prompt
    )
    frozen_result = json.loads(
        (REPO_ROOT_BOOTSTRAP / "results/e8_3b_evidence/result.json").read_text(
            encoding="utf-8"
        )
    )
    frozen_basis = frozen_result["scientific"]["basis_and_directions"]
    if reconstruction["basis_and_directions"] != frozen_basis:
        raise RuntimeError("E9 preflight basis differs from E8")
    if (
        reconstruction["memory_receipt"]
        != frozen["arms"]["candidate"]["memory_receipt"]
    ):
        raise RuntimeError("E9 preflight memory reconstruction differs from E8")
    if reconstruction["key_whitening"] != frozen["key_whitening"]:
        raise RuntimeError("E9 preflight key whitening differs from E8")
    if reconstruction["anchor_scan"]["summary"] != {
        "anchor_count": 64,
        "gate_open_count": 64,
        "own_slot_count": 64,
        "nonzero_delta_count": 64,
    }:
        raise RuntimeError("E9 preflight exact-key non-vacuity control failed")
    environment = environment_record(device)
    if environment != EXPECTED_ENVIRONMENT:
        raise RuntimeError(f"registered E9 environment mismatch: {environment}")

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
        "snapshot_selector": (
            "HLM5_E7_SNAPSHOT" if os.environ.get("HLM5_E7_SNAPSHOT") else "HF_HOME"
        ),
        "query_case_start": QUERY_CASE_START,
        "query_case_stop": QUERY_CASE_STOP,
        "query_case_count": QUERY_CASE_COUNT,
        "query_count": QUERY_COUNT,
        "query_row_sha256": QUERY_ROW_SHA256,
        "query_case_id_sha256": QUERY_CASE_ID_SHA256,
        "query_prompt_sha256": QUERY_PROMPT_SHA256,
        "query_kind_sha256": dict(QUERY_KIND_SHA256),
        "model_forward_batch": MODEL_FORWARD_BATCH,
        "gate_batch": GATE_BATCH,
        "basis_and_directions": reconstruction["basis_and_directions"],
        "e8_memory_receipt": reconstruction["memory_receipt"],
        "e8_anchor_scan": reconstruction["anchor_scan"],
        "model_invariants": invariants,
        "environment": environment,
        "native_runtime": native_runtime,
        "preflight_test_command": "python -m pytest tests/test_e9_3b.py -q",
        "preflight_tests_returncode": tests.returncode,
        "preflight_tests_stdout": tests.stdout.strip(),
        "preflight_tests_stderr": tests.stderr.strip(),
        "e9_query_outcomes_computed": False,
        "query_population_materialized_without_model": len(
            query_population["prompt_order"]
        ),
        "commands": list(REGISTERED_COMMANDS),
    }
    atomic_create_json(PREFLIGHT_PATH, payload)
    print(f"READY preflight: {PREFLIGHT_PATH}")


if __name__ == "__main__":
    main()

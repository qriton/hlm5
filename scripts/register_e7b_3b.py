"""Create the explicit pre-outcome execution receipt for E7b."""

# ruff: noqa: E402 -- direct script execution bootstraps the repository root.
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

import torch

from hlm5.e7_contract import MODEL_FILE_SHA256, sha256_file
from hlm5.e7b_contract import (
    ADMISSION_PATH,
    ATTEMPT_PATH,
    DEPENDENCY_SHA256,
    EXECUTION_RECEIPT_PATH,
    EXECUTION_SCHEMA,
    PREFLIGHT_PATH,
    PROMPT_SHA256,
    PROTOCOL_COMMIT,
    PROTOCOL_SHA256,
    REGISTERED_COMMANDS,
    RESULT_PATH,
    TARGET_POOL_SHA256,
    VERDICT_PATH,
    assert_outputs_absent,
    atomic_create_json,
    load_preflight,
)
from hlm5.e7b_runtime import configure_native_runtime, environment_record


TEST_COMMAND = [sys.executable, "-m", "pytest", "tests/test_e7b_3b.py", "-q"]


def main() -> None:
    assert_outputs_absent(
        [
            EXECUTION_RECEIPT_PATH,
            ATTEMPT_PATH,
            ADMISSION_PATH,
            RESULT_PATH,
            VERDICT_PATH,
        ]
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("registered E7b registration requires CUDA")
    device = torch.device("cuda", 0)
    native_runtime = configure_native_runtime()
    preflight, preflight_sha = load_preflight()
    environment = environment_record(device)
    if environment != preflight.get("environment"):
        raise RuntimeError("E7b registration environment differs from preflight")
    if native_runtime != preflight.get("native_runtime"):
        raise RuntimeError("E7b registration arithmetic differs from preflight")

    tests = subprocess.run(
        TEST_COMMAND,
        cwd=REPO_ROOT_BOOTSTRAP,
        capture_output=True,
        text=True,
        check=False,
    )
    if tests.returncode != 0:
        raise RuntimeError(
            "registered E7b tests failed at registration:\n"
            + tests.stdout
            + tests.stderr
        )
    payload = {
        "schema": EXECUTION_SCHEMA,
        "status": "READY",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": preflight["implementation_commit"],
        "preflight_sha256": preflight_sha,
        "source_sha256": preflight["source_sha256"],
        "dependency_sha256": dict(DEPENDENCY_SHA256),
        "model_file_sha256": dict(MODEL_FILE_SHA256),
        "prompt_sha256": PROMPT_SHA256,
        "target_pool_sha256": TARGET_POOL_SHA256,
        "environment": environment,
        "native_runtime": native_runtime,
        "registered_commands": list(REGISTERED_COMMANDS),
        "test_command": "python -m pytest tests/test_e7b_3b.py -q",
        "tests_returncode": tests.returncode,
        "tests_stdout": tests.stdout.strip(),
        "tests_stderr": tests.stderr.strip(),
        "candidate_outputs_computed_before_receipt": False,
        "pre_attempt_outputs_absent": True,
        "preflight_file_sha256_recomputed": sha256_file(PREFLIGHT_PATH),
    }
    if payload["preflight_file_sha256_recomputed"] != preflight_sha:
        raise RuntimeError("E7b preflight changed during registration")
    atomic_create_json(EXECUTION_RECEIPT_PATH, payload)
    print(f"READY execution receipt: {EXECUTION_RECEIPT_PATH}")


if __name__ == "__main__":
    main()

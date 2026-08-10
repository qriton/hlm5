"""Create the pre-outcome execution receipt for HLM5 E11."""

# ruff: noqa: E402
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from hlm5.e7_contract import sha256_file
from hlm5.e7b_runtime import configure_native_runtime
from hlm5.e11_contract import (
    ADMISSION_PATH,
    ATTEMPT_PATH,
    DEPENDENCY_SHA256,
    EXECUTION_RECEIPT_PATH,
    EXECUTION_SCHEMA,
    PREFLIGHT_PATH,
    PROTOCOL_COMMIT,
    PROTOCOL_SHA256,
    REGISTERED_COMMANDS,
    RESULT_PATH,
    VERDICT_PATH,
    assert_outputs_absent,
    atomic_create_json,
    load_preflight,
)
from hlm5.e11_runtime import e11_environment_record

TEST_COMMAND = [sys.executable, "-m", "pytest", "tests/test_e11_3b.py", "-q"]


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
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("registered E11 registration requires exactly one GPU")
    device = torch.device("cuda", 0)
    native = configure_native_runtime()
    preflight, preflight_sha = load_preflight()
    environment = e11_environment_record(device)
    if environment != preflight["environment"] or native != preflight["native_runtime"]:
        raise RuntimeError("E11 registration runtime differs from preflight")
    tests = subprocess.run(TEST_COMMAND, cwd=ROOT, capture_output=True, text=True)
    if tests.returncode:
        raise RuntimeError(
            "registered E11 tests failed:\n" + tests.stdout + tests.stderr
        )
    payload = {
        "schema": EXECUTION_SCHEMA,
        "status": "READY",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": preflight["implementation_commit"],
        "preflight_sha256": preflight_sha,
        "preflight_file_sha256_recomputed": sha256_file(PREFLIGHT_PATH),
        "source_sha256": preflight["source_sha256"],
        "dependency_sha256": dict(DEPENDENCY_SHA256),
        "e10_raw_sha256": preflight["e10_raw_sha256"],
        "e10_bundle_manifest": preflight["e10_bundle_manifest"],
        "environment": environment,
        "native_runtime": native,
        "registered_commands": list(REGISTERED_COMMANDS),
        "test_command": "python -m pytest tests/test_e11_3b.py -q",
        "tests_returncode": tests.returncode,
        "tests_stdout": tests.stdout.strip(),
        "tests_stderr": tests.stderr.strip(),
        "e11_outcomes_computed_before_receipt": False,
        "pre_attempt_outputs_absent": True,
    }
    atomic_create_json(EXECUTION_RECEIPT_PATH, payload)
    print(f"READY execution receipt: {EXECUTION_RECEIPT_PATH}")


if __name__ == "__main__":
    main()

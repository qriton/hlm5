"""Create the immutable pre-outcome execution receipt for HLM5 E14."""

# ruff: noqa: E402
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hlm5.e14_contract import (
    ADMISSION_PATH,
    ATTEMPT_PATH,
    BUNDLE_PATH,
    EXECUTION_RECEIPT_PATH,
    EXECUTION_SCHEMA,
    REGISTERED_COMMANDS,
    RESULT_PATH,
    VERDICT_PATH,
    assert_outputs_absent,
    atomic_create_json,
    load_preflight,
)


def main() -> None:
    preflight, preflight_sha = load_preflight()
    assert_outputs_absent(
        [EXECUTION_RECEIPT_PATH, ATTEMPT_PATH, BUNDLE_PATH, ADMISSION_PATH, RESULT_PATH, VERDICT_PATH]
    )
    payload = {
        "schema": EXECUTION_SCHEMA,
        "status": "READY",
        "preflight_sha256": preflight_sha,
        "implementation_commit": preflight["implementation_commit"],
        "source_sha256": preflight["source_sha256"],
        "dependency_sha256": preflight["dependency_sha256"],
        "environment": preflight["environment"],
        "native_runtime": preflight["native_runtime"],
        "registered_commands": list(REGISTERED_COMMANDS),
        "pre_attempt_outputs_absent": True,
        "registered_unix_seconds": time.time(),
    }
    atomic_create_json(EXECUTION_RECEIPT_PATH, payload)
    print(f"READY execution receipt: {EXECUTION_RECEIPT_PATH}")


if __name__ == "__main__":
    main()

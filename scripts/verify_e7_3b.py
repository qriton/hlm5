"""Emit the formal fail-closed verdict for the frozen-3B E7 study."""

# ruff: noqa: E402 -- direct script execution bootstraps the repository root.
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from hlm5.e7_contract import (
    RESULT_PATH,
    VERDICT_PATH,
    VERDICT_SCHEMA,
    atomic_write_json,
    load_execution_receipt,
    load_json_object,
    result_contract,
    result_validity_failures,
    scientific_failures,
    sha256_file,
)


PRODUCER = REPO_ROOT_BOOTSTRAP / "scripts" / "run_e7_3b.py"


def main() -> int:
    if not RESULT_PATH.is_file():
        verdict = {
            "schema": VERDICT_SCHEMA,
            "verdict": "INCOMPLETE",
            "validity_failures": [],
            "scientific_failures": [],
            "reason": "registered E7 result is absent",
        }
        atomic_write_json(VERDICT_PATH, verdict)
        print("INCOMPLETE: registered E7 result is absent")
        return 3

    validity_failures: list[str] = []
    scientific: list[str] = []
    try:
        load_execution_receipt()
        result = load_json_object(RESULT_PATH)
        expected_contract = result_contract(PRODUCER)
        if result.get("certificate_contract") != expected_contract:
            validity_failures.append("result certificate contract mismatch")
        validity_failures.extend(result_validity_failures(result))
        if not validity_failures:
            scientific = scientific_failures(result)
    except Exception as error:  # fail closed and preserve the formal verdict
        result = None
        validity_failures.append(f"verification exception: {type(error).__name__}: {error}")

    if validity_failures:
        verdict_name = "IMPLEMENTATION_INVALID"
        exit_code = 2
    elif scientific:
        verdict_name = "FAIL_3B_PORTABILITY"
        exit_code = 0
    else:
        verdict_name = "PASS_3B_PORTABILITY"
        exit_code = 0
    verdict = {
        "schema": VERDICT_SCHEMA,
        "verdict": verdict_name,
        "result_sha256": sha256_file(RESULT_PATH),
        "validity_failures": validity_failures,
        "scientific_failures": scientific,
        "binding_bars_passed": not validity_failures,
        "claim_boundary": (
            "A pass applies only to the pinned SmolLM3-3B-Base revision, "
            "single-token targets, and registered exact-key evaluation."
        ),
    }
    atomic_write_json(VERDICT_PATH, verdict)
    print(verdict_name)
    for failure in validity_failures + scientific:
        print(f"- {failure}")
    print(f"verdict -> {VERDICT_PATH}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

"""Emit the formal fail-closed E9 wide-locality verdict."""

# ruff: noqa: E402 -- direct script execution bootstraps the repository root.
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from hlm5.e7_contract import sha256_file
from hlm5.e9_contract import (
    ADMISSION_PATH,
    ATTEMPT_PATH,
    RESULT_PATH,
    VERDICT_PATH,
    VERDICT_SCHEMA,
    atomic_create_json,
    load_admission,
    load_attempt,
    load_json_object,
    load_result,
    scientific_failures,
    scientific_sha256,
    verdict_scientific_payload,
)


def optional_hash(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def main() -> int:
    if VERDICT_PATH.exists():
        raise FileExistsError(f"E9 verdict already exists: {VERDICT_PATH}")
    validity_failures: list[str] = []
    scientific_failure_list: list[str] = []
    result: dict[str, object] | None = None
    admission: dict[str, object] | None = None
    attempt_sha: str | None = None
    admission_sha: str | None = None
    result_sha: str | None = None

    if not RESULT_PATH.is_file():
        verdict_name = "INCOMPLETE"
        exit_code = 3
        try:
            if ATTEMPT_PATH.is_file():
                _attempt, attempt_sha = load_attempt()
            if ADMISSION_PATH.is_file():
                admission, admission_sha = load_admission()
        except Exception as error:
            validity_failures.append(
                f"verification exception: {type(error).__name__}: {error}"
            )
            verdict_name = "IMPLEMENTATION_INVALID"
            exit_code = 2
    else:
        try:
            _attempt, attempt_sha = load_attempt()
            admission, admission_sha = load_admission()
            result, result_sha = load_result()
            scientific_failure_list = scientific_failures(result)
        except Exception as error:
            validity_failures.append(
                f"verification exception: {type(error).__name__}: {error}"
            )
        if validity_failures:
            verdict_name = "IMPLEMENTATION_INVALID"
            exit_code = 2
        elif scientific_failure_list:
            verdict_name = "FAIL_3B_WIDE_LOCALITY"
            exit_code = 0
        else:
            verdict_name = "PASS_3B_WIDE_LOCALITY"
            exit_code = 0

    scientific = verdict_scientific_payload(
        verdict=verdict_name,
        result=result,
        validity_failures=validity_failures,
        scientific_failures_value=scientific_failure_list,
    )
    payload = {
        "schema": VERDICT_SCHEMA,
        "verdict": verdict_name,
        "attempt_sha256": attempt_sha or optional_hash(ATTEMPT_PATH),
        "admission_sha256": admission_sha or optional_hash(ADMISSION_PATH),
        "admission_scientific_sha256": (
            None if admission is None else admission.get("scientific_sha256")
        ),
        "result_sha256": result_sha or optional_hash(RESULT_PATH),
        "result_scientific_sha256": (
            None if result is None else result.get("scientific_sha256")
        ),
        "scientific": scientific,
        "scientific_sha256": scientific_sha256(scientific),
    }
    if result is None and RESULT_PATH.is_file():
        try:
            raw_result = load_json_object(RESULT_PATH)
        except Exception:
            raw_result = {}
        payload["invalid_result_claims"] = {
            "attempt_sha256": raw_result.get("attempt_sha256"),
            "admission_sha256": raw_result.get("admission_sha256"),
            "admission_scientific_sha256": raw_result.get(
                "admission_scientific_sha256"
            ),
            "result_scientific_sha256": raw_result.get("scientific_sha256"),
        }
    atomic_create_json(VERDICT_PATH, payload)
    print(verdict_name)
    for failure in validity_failures + scientific_failure_list:
        print(f"- {failure}")
    print(f"verdict -> {VERDICT_PATH}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

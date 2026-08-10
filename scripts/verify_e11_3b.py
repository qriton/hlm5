"""Emit the formal HLM5 E11 cross-node portability verdict."""

# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hlm5.e7_contract import sha256_file
from hlm5.e11_contract import (
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
        raise FileExistsError(VERDICT_PATH)
    validity: list[str] = []
    science_fail: list[str] = []
    result = admission = None
    attempt_sha = admission_sha = result_sha = None
    if not RESULT_PATH.is_file():
        verdict, code = "INCOMPLETE", 3
        try:
            if ATTEMPT_PATH.is_file():
                _attempt, attempt_sha = load_attempt()
            if ADMISSION_PATH.is_file():
                admission, admission_sha = load_admission()
        except Exception as error:
            validity.append(f"verification exception: {type(error).__name__}: {error}")
            verdict, code = "IMPLEMENTATION_INVALID", 2
    else:
        try:
            _attempt, attempt_sha = load_attempt()
            admission, admission_sha = load_admission()
            result, result_sha = load_result()
            science_fail = scientific_failures(result)
        except Exception as error:
            validity.append(f"verification exception: {type(error).__name__}: {error}")
        if validity:
            verdict, code = "IMPLEMENTATION_INVALID", 2
        elif science_fail:
            verdict, code = "FAIL_3B_CROSS_NODE_PORTABILITY", 0
        else:
            verdict, code = "PASS_3B_CROSS_NODE_PORTABLE", 0
    scientific = verdict_scientific_payload(
        verdict=verdict,
        result=result,
        validity_failures=validity,
        scientific_failures_value=science_fail,
    )
    payload = {
        "schema": VERDICT_SCHEMA,
        "verdict": verdict,
        "attempt_sha256": attempt_sha or optional_hash(ATTEMPT_PATH),
        "admission_sha256": admission_sha or optional_hash(ADMISSION_PATH),
        "admission_scientific_sha256": None
        if admission is None
        else admission.get("scientific_sha256"),
        "result_sha256": result_sha or optional_hash(RESULT_PATH),
        "result_scientific_sha256": None
        if result is None
        else result.get("scientific_sha256"),
        "scientific": scientific,
        "scientific_sha256": scientific_sha256(scientific),
    }
    if result is None and RESULT_PATH.is_file():
        try:
            raw = load_json_object(RESULT_PATH)
        except Exception:
            raw = {}
        payload["invalid_result_claims"] = {
            "attempt_sha256": raw.get("attempt_sha256"),
            "admission_sha256": raw.get("admission_sha256"),
            "admission_scientific_sha256": raw.get("admission_scientific_sha256"),
            "result_scientific_sha256": raw.get("scientific_sha256"),
        }
    atomic_create_json(VERDICT_PATH, payload)
    print(verdict)
    for failure in validity + science_fail:
        print(f"- {failure}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

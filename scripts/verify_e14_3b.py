"""Emit the formal fail-closed HLM5 E14 verdict."""

# ruff: noqa: E402
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hlm5.e7_contract import sha256_file
from hlm5.e14_contract import (
    ADMISSION_PATH,
    ATTEMPT_PATH,
    RESULT_PATH,
    VERDICT_PATH,
    VERDICT_SCHEMA,
    atomic_create_json,
    binding_verdict,
    load_admission,
    load_attempt,
    load_result,
)


def main() -> None:
    if VERDICT_PATH.exists():
        raise FileExistsError(VERDICT_PATH)
    verdict = "INCOMPLETE"
    reasons: list[str] = []
    summary = None
    attempt_sha = sha256_file(ATTEMPT_PATH) if ATTEMPT_PATH.is_file() else None
    admission_sha = sha256_file(ADMISSION_PATH) if ADMISSION_PATH.is_file() else None
    result_sha = sha256_file(RESULT_PATH) if RESULT_PATH.is_file() else None
    try:
        if not ATTEMPT_PATH.is_file():
            reasons.append("attempt was never spent")
        else:
            load_attempt()
            if not ADMISSION_PATH.is_file():
                reasons.append("admission absent after spent attempt")
            else:
                admission_raw = __import__("json").loads(ADMISSION_PATH.read_text())
                if admission_raw.get("status") == "IMPLEMENTATION_INVALID":
                    verdict = "IMPLEMENTATION_INVALID"
                    reasons.append(admission_raw.get("failure_message", "invalid admission"))
                elif not RESULT_PATH.is_file():
                    load_admission()
                    reasons.append("replay result absent")
                else:
                    result_raw = __import__("json").loads(RESULT_PATH.read_text())
                    if result_raw.get("status") == "IMPLEMENTATION_INVALID":
                        verdict = "IMPLEMENTATION_INVALID"
                        reasons.append(result_raw.get("failure_message", "invalid replay"))
                    else:
                        result, _ = load_result()
                        verdict, reasons, summary = binding_verdict(
                            result["scientific"]["measurement"]
                        )
    except Exception as error:
        verdict = "IMPLEMENTATION_INVALID"
        reasons.append(f"{type(error).__name__}: {error}")
    payload = {
        "schema": VERDICT_SCHEMA,
        "verdict": verdict,
        "reasons": reasons,
        "attempt_sha256": attempt_sha,
        "admission_sha256": admission_sha,
        "result_sha256": result_sha,
        "summary": summary,
        "verified_unix_seconds": time.time(),
    }
    atomic_create_json(VERDICT_PATH, payload)
    print(verdict)
    for reason in reasons:
        print(f"- {reason}")


if __name__ == "__main__":
    main()

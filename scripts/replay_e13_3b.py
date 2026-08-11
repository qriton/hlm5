"""New-process exact replay for registered HLM5 E13."""

# ruff: noqa: E402
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from hlm5.e7_contract import sha256_file, snapshot_path
from hlm5.e7_runtime import (
    assert_model_invariants,
    final_hidden_batch,
    load_pinned_model,
    load_pinned_tokenizer,
)
from hlm5.e7b_runtime import configure_native_runtime
from hlm5.e13_contract import (
    BUNDLE_PATH,
    RESULT_PATH,
    RESULT_SCHEMA,
    SEED,
    VERDICT_PATH,
    assert_outputs_absent,
    atomic_create_json,
    load_admission,
    load_attempt,
    load_execution_receipt,
    result_validity_failures,
    scientific_sha256,
    selected_population,
)
from hlm5.e13_runtime import (
    MODEL_FORWARD_BATCH,
    e13_environment_record,
    load_bundle,
    terminal_measurement,
)


def run_replay(
    *,
    started: float,
    admission: dict[str, Any],
    admission_sha: str,
    attempt_sha: str,
    native_runtime: dict[str, Any],
    device: torch.device,
) -> None:
    snapshot = snapshot_path()
    tokenizer = load_pinned_tokenizer(snapshot)
    population = selected_population(tokenizer)
    model = load_pinned_model(snapshot, device)
    invariants = assert_model_invariants(model)
    prompts = population["prompt_order"] + population["whitening_prompts"]
    hidden = final_hidden_batch(
        model, tokenizer, prompts, device, batch_size=MODEL_FORWARD_BATCH
    )
    exact_hidden = hidden[: len(population["cases"])]
    whitening_hidden = hidden[len(population["cases"]) :]
    locality_hidden = final_hidden_batch(
        model,
        tokenizer,
        population["query_prompt_order"],
        device,
        batch_size=MODEL_FORWARD_BATCH,
    )
    measurement, portable = terminal_measurement(
        model,
        population["cases"],
        exact_hidden,
        whitening_hidden,
        population["query_rows"],
        locality_hidden,
    )
    measurement["e13_population"] = {
        "positive": population["record"],
        "locality": population["query_record"],
    }
    frozen_measurement = admission["scientific"]["measurement"]
    frozen_bundle = frozen_measurement.get("bundle")
    bundle_exact = portable is None and frozen_bundle is None
    if frozen_bundle is not None:
        if portable is None:
            raise RuntimeError("E13 replay did not reconstruct the portable bundle")
        loaded = load_bundle(BUNDLE_PATH, frozen_bundle)
        bundle_exact = all(torch.equal(loaded[name], portable[name]) for name in loaded)
        if sha256_file(BUNDLE_PATH) != frozen_bundle["file_sha256"]:
            bundle_exact = False
        measurement["bundle"] = frozen_bundle
    scientific = {
        "model_invariants": invariants,
        "admission_scientific_sha256": admission["scientific_sha256"],
        "measurement_exact_match": measurement == frozen_measurement,
        "bundle_exact_match": bundle_exact,
        "measurement": measurement,
    }
    payload = {
        "schema": RESULT_SCHEMA,
        "status": "COMPLETE",
        "attempt_sha256": attempt_sha,
        "admission_sha256": admission_sha,
        "admission_scientific_sha256": admission["scientific_sha256"],
        "native_runtime": native_runtime,
        "scientific": scientific,
        "scientific_sha256": scientific_sha256(scientific),
        "runtime_seconds": time.time() - started,
    }
    failures = result_validity_failures(payload, admission)
    atomic_create_json(RESULT_PATH, payload)
    print(f"COMPLETE replay: {RESULT_PATH}")
    if failures:
        print("KNOWN VALIDITY FAILURES:")
        for failure in failures:
            print(f"- {failure}")
    else:
        print("replay validity checks: PASS")


def main() -> None:
    started = time.time()
    assert_outputs_absent([RESULT_PATH, VERDICT_PATH])
    receipt, _receipt_sha = load_execution_receipt()
    admission, admission_sha = load_admission()
    _attempt, attempt_sha = load_attempt()
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("registered E13 replay requires exactly one GPU")
        torch.manual_seed(SEED)
        device = torch.device("cuda", 0)
        native_runtime = configure_native_runtime()
        if native_runtime != receipt["native_runtime"]:
            raise RuntimeError("E13 replay arithmetic differs from registration")
        if e13_environment_record(device) != receipt["environment"]:
            raise RuntimeError("E13 replay environment differs from registration")
        run_replay(
            started=started,
            admission=admission,
            admission_sha=admission_sha,
            attempt_sha=attempt_sha,
            native_runtime=native_runtime,
            device=device,
        )
    except Exception as error:
        failure = {
            "schema": RESULT_SCHEMA,
            "status": "IMPLEMENTATION_INVALID",
            "attempt_sha256": attempt_sha,
            "admission_sha256": admission_sha,
            "admission_scientific_sha256": admission.get("scientific_sha256"),
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "runtime_seconds": time.time() - started,
        }
        if not RESULT_PATH.exists():
            atomic_create_json(RESULT_PATH, failure)
        raise


if __name__ == "__main__":
    main()

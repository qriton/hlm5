"""Run the single-use E10 nested-capacity admission."""

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

from hlm5.e7_contract import snapshot_path
from hlm5.e7_runtime import assert_model_invariants, final_hidden_batch, load_pinned_model, load_pinned_tokenizer
from hlm5.e7b_runtime import configure_native_runtime
from hlm5.e10_contract import (
    ADMISSION_PATH, ADMISSION_SCHEMA, ATTEMPT_PATH, ATTEMPT_SCHEMA, BUNDLE_PATH,
    CANDIDATE_CASE_SHA256, PROTOCOL_COMMIT, PROTOCOL_SHA256, RESULT_PATH,
    SEED, VERDICT_PATH, admission_validity_failures, assert_outputs_absent,
    atomic_create_json, load_execution_receipt, scientific_sha256,
    selected_population,
)
from hlm5.e10_runtime import (
    MODEL_FORWARD_BATCH, atomic_create_bundle, capacity_measurement,
    e10_environment_record,
)


def prepare_before_attempt(device: torch.device) -> dict[str, Any]:
    snapshot = snapshot_path()
    tokenizer = load_pinned_tokenizer(snapshot)
    population = selected_population(tokenizer)
    model = load_pinned_model(snapshot, device)
    return {
        "tokenizer": tokenizer,
        "population": population,
        "model": model,
        "model_invariants": assert_model_invariants(model),
    }


def run_after_attempt(
    *, started: float, prepared: dict[str, Any], receipt: dict[str, Any],
    receipt_sha: str, attempt_sha: str, native_runtime: dict[str, Any],
    device: torch.device,
) -> None:
    population = prepared["population"]
    prompts = population["prompt_order"] + population["whitening_prompts"]
    candidate_and_whitening = final_hidden_batch(
        prepared["model"], prepared["tokenizer"], prompts, device,
        batch_size=MODEL_FORWARD_BATCH,
    )
    exact_hidden = candidate_and_whitening[: len(population["cases"])]
    whitening_hidden = candidate_and_whitening[len(population["cases"]):]
    locality_hidden = final_hidden_batch(
        prepared["model"], prepared["tokenizer"], population["query_prompt_order"],
        device, batch_size=MODEL_FORWARD_BATCH,
    )
    measurement, bundle = capacity_measurement(
        prepared["model"], population["cases"], exact_hidden, whitening_hidden,
        population["query_rows"], locality_hidden,
    )
    if bundle is not None:
        file_sha = atomic_create_bundle(BUNDLE_PATH, bundle)
        measurement["bundle"]["file_sha256"] = file_sha
    scientific = {
        "model_invariants": prepared["model_invariants"],
        "measurement": measurement,
    }
    payload = {
        "schema": ADMISSION_SCHEMA,
        "status": "COMPLETE",
        "attempt_sha256": attempt_sha,
        "execution_receipt_sha256": receipt_sha,
        "implementation_commit": receipt["implementation_commit"],
        "native_runtime": native_runtime,
        "scientific": scientific,
        "scientific_sha256": scientific_sha256(scientific),
        "runtime_seconds": time.time() - started,
    }
    failures = admission_validity_failures(payload)
    atomic_create_json(ADMISSION_PATH, payload)
    print(f"COMPLETE admission: {ADMISSION_PATH}")
    if failures:
        print("KNOWN VALIDITY FAILURES:")
        for failure in failures:
            print(f"- {failure}")
    else:
        print("admission validity checks: PASS")


def main() -> None:
    started = time.time()
    assert_outputs_absent([ATTEMPT_PATH, BUNDLE_PATH, ADMISSION_PATH, RESULT_PATH, VERDICT_PATH])
    receipt, receipt_sha = load_execution_receipt()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("registered E10 admission requires exactly one GPU")
    torch.manual_seed(SEED)
    device = torch.device("cuda", 0)
    native_runtime = configure_native_runtime()
    if native_runtime != receipt["native_runtime"]:
        raise RuntimeError("E10 admission arithmetic differs from registration")
    if e10_environment_record(device) != receipt["environment"]:
        raise RuntimeError("E10 admission environment differs from registration")
    prepared = prepare_before_attempt(device)
    attempt = {
        "schema": ATTEMPT_SCHEMA,
        "status": "POOL_SPENT",
        "execution_receipt_sha256": receipt_sha,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": receipt["implementation_commit"],
        "candidate_case_sha256": CANDIDATE_CASE_SHA256,
        "pre_admission_outputs_absent": True,
        "burn_unix_seconds": time.time(),
    }
    atomic_create_json(ATTEMPT_PATH, attempt)
    from hlm5.e7_contract import sha256_file
    attempt_sha = sha256_file(ATTEMPT_PATH)
    try:
        run_after_attempt(
            started=started, prepared=prepared, receipt=receipt,
            receipt_sha=receipt_sha, attempt_sha=attempt_sha,
            native_runtime=native_runtime, device=device,
        )
    except Exception as error:
        failure = {
            "schema": ADMISSION_SCHEMA,
            "status": "IMPLEMENTATION_INVALID",
            "attempt_sha256": attempt_sha,
            "execution_receipt_sha256": receipt_sha,
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "runtime_seconds": time.time() - started,
        }
        if not ADMISSION_PATH.exists():
            atomic_create_json(ADMISSION_PATH, failure)
        raise


if __name__ == "__main__":
    main()

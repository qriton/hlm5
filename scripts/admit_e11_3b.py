"""Run the single-use HLM5 E11 cross-node admission."""

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
from hlm5.e10_runtime import MODEL_FORWARD_BATCH
from hlm5.e11_contract import (
    ADMISSION_PATH,
    ADMISSION_SCHEMA,
    ATTEMPT_PATH,
    ATTEMPT_SCHEMA,
    BUNDLE_PATH,
    E10_BUNDLE_SHA256,
    PROTOCOL_COMMIT,
    PROTOCOL_SHA256,
    RESULT_PATH,
    SEED,
    VERDICT_PATH,
    assert_outputs_absent,
    atomic_create_json,
    frozen_e10,
    load_execution_receipt,
    scientific_sha256,
    selected_population,
    structural_failures,
)
from hlm5.e11_runtime import cross_node_measurement, e11_environment_record, load_bundle


def prepare_before_attempt(device: torch.device) -> dict[str, Any]:
    snapshot = snapshot_path()
    tokenizer = load_pinned_tokenizer(snapshot)
    population = selected_population(tokenizer)
    evidence = frozen_e10()
    bundle = load_bundle(BUNDLE_PATH, evidence["bundle"])
    model = load_pinned_model(snapshot, device)
    return {
        "tokenizer": tokenizer,
        "population": population,
        "evidence": evidence,
        "bundle": bundle,
        "model": model,
        "model_invariants": assert_model_invariants(model),
    }


def run_after_attempt(
    *,
    started: float,
    prepared: dict[str, Any],
    receipt: dict[str, Any],
    receipt_sha: str,
    attempt_sha: str,
    native_runtime: dict[str, Any],
    device: torch.device,
) -> None:
    population = prepared["population"]
    exact_hidden = final_hidden_batch(
        prepared["model"],
        prepared["tokenizer"],
        population["prompt_order"],
        device,
        batch_size=MODEL_FORWARD_BATCH,
    )
    locality_hidden = final_hidden_batch(
        prepared["model"],
        prepared["tokenizer"],
        population["query_prompt_order"],
        device,
        batch_size=MODEL_FORWARD_BATCH,
    )
    evidence = prepared["evidence"]
    measurement = cross_node_measurement(
        prepared["model"],
        population["cases"],
        exact_hidden,
        population["query_rows"],
        locality_hidden,
        prepared["bundle"],
        evidence["bundle"],
        evidence["selected_indices"],
        device,
    )
    failures = structural_failures(measurement)
    if failures:
        raise RuntimeError("E11 structural measurement failure: " + "; ".join(failures))
    scientific = {
        "model_invariants": prepared["model_invariants"],
        "e10_tier_sha256": scientific_sha256(evidence["tier"]),
        "cross_node_exact_match": measurement == evidence["tier"],
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
    atomic_create_json(ADMISSION_PATH, payload)
    print(f"COMPLETE admission: {ADMISSION_PATH}")


def main() -> None:
    started = time.time()
    assert_outputs_absent([ATTEMPT_PATH, ADMISSION_PATH, RESULT_PATH, VERDICT_PATH])
    receipt, receipt_sha = load_execution_receipt()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("registered E11 admission requires exactly one GPU")
    torch.manual_seed(SEED)
    device = torch.device("cuda", 0)
    native = configure_native_runtime()
    if native != receipt["native_runtime"]:
        raise RuntimeError("E11 admission arithmetic differs from registration")
    if e11_environment_record(device) != receipt["environment"]:
        raise RuntimeError("E11 admission environment differs from registration")
    prepared = prepare_before_attempt(device)
    attempt = {
        "schema": ATTEMPT_SCHEMA,
        "status": "PORTABILITY_ATTEMPT_SPENT",
        "execution_receipt_sha256": receipt_sha,
        "implementation_commit": receipt["implementation_commit"],
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "bundle_sha256": E10_BUNDLE_SHA256,
        "pre_admission_outputs_absent": True,
        "burn_unix_seconds": time.time(),
    }
    atomic_create_json(ATTEMPT_PATH, attempt)
    attempt_sha = sha256_file(ATTEMPT_PATH)
    try:
        run_after_attempt(
            started=started,
            prepared=prepared,
            receipt=receipt,
            receipt_sha=receipt_sha,
            attempt_sha=attempt_sha,
            native_runtime=native,
            device=device,
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

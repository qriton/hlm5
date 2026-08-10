"""Run the single-use E9 wide exact-key locality admission."""

# ruff: noqa: E402 -- direct script execution bootstraps the repository root.
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

import torch

from hlm5.e7_contract import sha256_file, snapshot_path
from hlm5.e7_runtime import (
    assert_model_invariants,
    final_hidden_batch,
    load_pinned_model,
    load_pinned_tokenizer,
)
from hlm5.e7b_runtime import configure_native_runtime, environment_record
from hlm5.e8_contract import selected_population as selected_e8_population
from hlm5.e9_contract import (
    ADMISSION_PATH,
    ADMISSION_SCHEMA,
    ATTEMPT_PATH,
    ATTEMPT_SCHEMA,
    MODEL_FORWARD_BATCH,
    PROTOCOL_COMMIT,
    PROTOCOL_SHA256,
    QUERY_CASE_COUNT,
    QUERY_CASE_ID_SHA256,
    QUERY_CASE_START,
    QUERY_CASE_STOP,
    QUERY_COUNT,
    QUERY_KIND_SHA256,
    QUERY_PROMPT_SHA256,
    QUERY_ROW_SHA256,
    RESULT_PATH,
    SEED,
    VERDICT_PATH,
    admission_validity_failures,
    assert_outputs_absent,
    atomic_create_json,
    frozen_e8_measurement,
    load_execution_receipt,
    scientific_sha256,
    selected_population,
)
from hlm5.e9_runtime import GATE_BATCH, prepare_e8_adapter, scan_queries


def prepare_before_attempt(device: torch.device) -> dict[str, Any]:
    snapshot = snapshot_path()
    tokenizer = load_pinned_tokenizer(snapshot)
    e8_population = selected_e8_population(tokenizer)
    query_population = selected_population()
    model = load_pinned_model(snapshot, device)
    invariants = assert_model_invariants(model)
    e8_hiddens = final_hidden_batch(
        model,
        tokenizer,
        e8_population["prompt_order"],
        device,
        batch_size=MODEL_FORWARD_BATCH,
    )
    e8_hidden_by_prompt = {
        prompt: e8_hiddens[index]
        for index, prompt in enumerate(e8_population["prompt_order"])
    }
    frozen = frozen_e8_measurement()
    reconstruction = prepare_e8_adapter(
        model, e8_population, frozen, e8_hidden_by_prompt
    )
    return {
        "tokenizer": tokenizer,
        "model": model,
        "model_invariants": invariants,
        "e8_population": e8_population,
        "query_population": query_population,
        "reconstruction": reconstruction,
    }


def run_after_attempt(
    *,
    started: float,
    prepared: dict[str, Any],
    receipt: dict[str, Any],
    execution_sha: str,
    attempt_sha: str,
    native_runtime: dict[str, Any],
    device: torch.device,
) -> None:
    reconstruction = prepared["reconstruction"]
    query_population = prepared["query_population"]
    query_hiddens = final_hidden_batch(
        prepared["model"],
        prepared["tokenizer"],
        query_population["prompt_order"],
        device,
        batch_size=MODEL_FORWARD_BATCH,
    )
    scan = scan_queries(
        reconstruction["adapter"],
        reconstruction["memory"],
        prepared["e8_population"]["cases"],
        reconstruction["slot_by_case"],
        query_population["query_rows"],
        query_hiddens,
    )
    measurement = {
        "population": {
            "query_case_start": QUERY_CASE_START,
            "query_case_stop": QUERY_CASE_STOP,
            "query_case_count": QUERY_CASE_COUNT,
            "query_count": QUERY_COUNT,
            "query_row_sha256": QUERY_ROW_SHA256,
            "query_case_id_sha256": QUERY_CASE_ID_SHA256,
            "query_prompt_sha256": QUERY_PROMPT_SHA256,
            "query_kind_sha256": dict(QUERY_KIND_SHA256),
            "model_forward_batch": MODEL_FORWARD_BATCH,
            "model_forward_batch_count": QUERY_COUNT // MODEL_FORWARD_BATCH,
            "gate_batch": GATE_BATCH,
            "gate_batch_count": QUERY_COUNT // GATE_BATCH,
        },
        "e8_memory_receipt": reconstruction["memory_receipt"],
        "key_whitening": reconstruction["key_whitening"],
        "e8_anchor_scan": reconstruction["anchor_scan"],
        "query_scan": scan,
    }
    scientific = {
        "model_invariants": prepared["model_invariants"],
        "basis_and_directions": reconstruction["basis_and_directions"],
        "measurement": measurement,
    }
    payload = {
        "schema": ADMISSION_SCHEMA,
        "status": "COMPLETE",
        "attempt_sha256": attempt_sha,
        "execution_receipt_sha256": execution_sha,
        "implementation_commit": receipt["implementation_commit"],
        "native_runtime": native_runtime,
        "scientific": scientific,
        "scientific_sha256": scientific_sha256(scientific),
        "runtime_seconds": time.time() - started,
    }
    failures = admission_validity_failures(payload)
    if failures:
        raise RuntimeError("E9 admission is invalid: " + "; ".join(failures))
    atomic_create_json(ADMISSION_PATH, payload)
    summary = scan["summary"]
    print(f"COMPLETE admission: {ADMISSION_PATH}")
    print(
        "wide locality: "
        f"queries={summary['query_count']}, opens={summary['gate_open_count']}, "
        f"max_score={summary['maximum']['selected_score']:.9f}, "
        f"bit_identical={summary['hidden_bit_identical_count']}"
    )


def main() -> None:
    started = time.time()
    assert_outputs_absent([ATTEMPT_PATH, ADMISSION_PATH, RESULT_PATH, VERDICT_PATH])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("registered E9 admission requires exactly one GPU")
    torch.manual_seed(SEED)
    device = torch.device("cuda", 0)
    native_runtime = configure_native_runtime()
    receipt, execution_sha = load_execution_receipt()
    if native_runtime != receipt.get("native_runtime"):
        raise RuntimeError("E9 admission arithmetic differs from registration")
    if environment_record(device) != receipt.get("environment"):
        raise RuntimeError("E9 admission environment differs from registration")
    prepared = prepare_before_attempt(device)
    reconstruction = prepared["reconstruction"]
    if reconstruction["basis_and_directions"] != receipt["basis_and_directions"]:
        raise RuntimeError("E9 pre-attempt basis differs from registration")
    if reconstruction["memory_receipt"] != receipt["e8_memory_receipt"]:
        raise RuntimeError("E9 pre-attempt memory differs from registration")
    if reconstruction["anchor_scan"] != receipt["e8_anchor_scan"]:
        raise RuntimeError("E9 pre-attempt anchor scan differs from registration")
    attempt = {
        "schema": ATTEMPT_SCHEMA,
        "status": "POOL_SPENT",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": receipt["implementation_commit"],
        "execution_receipt_sha256": execution_sha,
        "query_row_sha256": QUERY_ROW_SHA256,
        "query_prompt_sha256": QUERY_PROMPT_SHA256,
        "pre_admission_outputs_absent": True,
    }
    atomic_create_json(ATTEMPT_PATH, attempt)
    attempt_sha: str | None = None
    try:
        attempt_sha = sha256_file(ATTEMPT_PATH)
        run_after_attempt(
            started=started,
            prepared=prepared,
            receipt=receipt,
            execution_sha=execution_sha,
            attempt_sha=attempt_sha,
            native_runtime=native_runtime,
            device=device,
        )
    except Exception as error:
        if attempt_sha is None:
            attempt_sha = sha256_file(ATTEMPT_PATH)
        failure = {
            "schema": ADMISSION_SCHEMA,
            "status": "IMPLEMENTATION_INVALID",
            "attempt_sha256": attempt_sha,
            "execution_receipt_sha256": execution_sha,
            "implementation_commit": receipt["implementation_commit"],
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "runtime_seconds": time.time() - started,
        }
        if not ADMISSION_PATH.exists():
            atomic_create_json(ADMISSION_PATH, failure)
        raise


if __name__ == "__main__":
    main()

"""New-process exact replay for the registered E9 locality scan."""

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

from hlm5.e7_contract import snapshot_path
from hlm5.e7_runtime import (
    assert_model_invariants,
    final_hidden_batch,
    load_pinned_model,
    load_pinned_tokenizer,
)
from hlm5.e7b_runtime import configure_native_runtime, environment_record
from hlm5.e8_contract import selected_population as selected_e8_population
from hlm5.e9_contract import (
    MODEL_FORWARD_BATCH,
    QUERY_CASE_COUNT,
    QUERY_CASE_ID_SHA256,
    QUERY_CASE_START,
    QUERY_CASE_STOP,
    QUERY_COUNT,
    QUERY_KIND_SHA256,
    QUERY_PROMPT_SHA256,
    QUERY_ROW_SHA256,
    RESULT_PATH,
    RESULT_SCHEMA,
    SEED,
    VERDICT_PATH,
    assert_outputs_absent,
    atomic_create_json,
    frozen_e8_measurement,
    load_admission,
    load_attempt,
    load_execution_receipt,
    result_validity_failures,
    scientific_sha256,
    selected_population,
)
from hlm5.e9_runtime import GATE_BATCH, prepare_e8_adapter, scan_queries


def _tensor_hash_projection(measurement: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "native_hidden_sha256",
        "delta_sha256",
        "adapted_hidden_sha256",
    )
    return {
        "anchors": [
            [row.get(field) for field in fields]
            for row in measurement["e8_anchor_scan"]["rows"]
        ],
        "queries": [
            [row.get(field) for field in fields]
            for row in measurement["query_scan"]["rows"]
        ],
    }


def run_replay(
    *,
    started: float,
    admission: dict[str, Any],
    admission_sha: str,
    attempt_sha: str,
    receipt: dict[str, Any],
    native_runtime: dict[str, Any],
    device: torch.device,
) -> None:
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
    reconstruction = prepare_e8_adapter(
        model, e8_population, frozen_e8_measurement(), e8_hidden_by_prompt
    )
    if reconstruction["basis_and_directions"] != receipt["basis_and_directions"]:
        raise RuntimeError("E9 replay basis differs from registration")
    if reconstruction["memory_receipt"] != receipt["e8_memory_receipt"]:
        raise RuntimeError("E9 replay memory differs from registration")
    if reconstruction["anchor_scan"] != receipt["e8_anchor_scan"]:
        raise RuntimeError("E9 replay anchor scan differs from registration")
    query_hiddens = final_hidden_batch(
        model,
        tokenizer,
        query_population["prompt_order"],
        device,
        batch_size=MODEL_FORWARD_BATCH,
    )
    scan = scan_queries(
        reconstruction["adapter"],
        reconstruction["memory"],
        e8_population["cases"],
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
    frozen_scientific = admission["scientific"]
    frozen_measurement = frozen_scientific["measurement"]
    scientific = {
        "model_invariants": invariants,
        "basis_and_directions": reconstruction["basis_and_directions"],
        "admission_scientific_sha256": admission["scientific_sha256"],
        "measurement_exact_match": measurement == frozen_measurement,
        "all_tensor_hashes_match": _tensor_hash_projection(measurement)
        == _tensor_hash_projection(frozen_measurement),
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
    receipt, _execution_sha = load_execution_receipt()
    admission, admission_sha = load_admission()
    _attempt, attempt_sha = load_attempt()
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("registered E9 replay requires exactly one GPU")
        torch.manual_seed(SEED)
        device = torch.device("cuda", 0)
        native_runtime = configure_native_runtime()
        if native_runtime != receipt.get("native_runtime"):
            raise RuntimeError("E9 replay arithmetic differs from registration")
        if environment_record(device) != receipt.get("environment"):
            raise RuntimeError("E9 replay environment differs from registration")
        run_replay(
            started=started,
            admission=admission,
            admission_sha=admission_sha,
            attempt_sha=attempt_sha,
            receipt=receipt,
            native_runtime=native_runtime,
            device=device,
        )
    except Exception as error:
        failure = {
            "schema": RESULT_SCHEMA,
            "status": "IMPLEMENTATION_INVALID",
            "attempt_sha256": attempt_sha,
            "admission_sha256": admission_sha,
            "admission_scientific_sha256": admission["scientific_sha256"],
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "runtime_seconds": time.time() - started,
        }
        if not RESULT_PATH.exists():
            atomic_create_json(RESULT_PATH, failure)
        raise


if __name__ == "__main__":
    main()

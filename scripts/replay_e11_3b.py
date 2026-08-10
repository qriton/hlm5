"""New-process exact replay for HLM5 E11."""

# ruff: noqa: E402
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from hlm5.e7_contract import snapshot_path
from hlm5.e7_runtime import (
    assert_model_invariants,
    final_hidden_batch,
    load_pinned_model,
    load_pinned_tokenizer,
)
from hlm5.e7b_runtime import configure_native_runtime
from hlm5.e10_runtime import MODEL_FORWARD_BATCH
from hlm5.e11_contract import (
    BUNDLE_PATH,
    RESULT_PATH,
    RESULT_SCHEMA,
    SEED,
    VERDICT_PATH,
    assert_outputs_absent,
    atomic_create_json,
    frozen_e10,
    load_admission,
    load_attempt,
    load_execution_receipt,
    result_validity_failures,
    scientific_sha256,
    selected_population,
)
from hlm5.e11_runtime import cross_node_measurement, e11_environment_record, load_bundle


def main() -> None:
    started = time.time()
    assert_outputs_absent([RESULT_PATH, VERDICT_PATH])
    receipt, _receipt_sha = load_execution_receipt()
    admission, admission_sha = load_admission()
    _attempt, attempt_sha = load_attempt()
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("registered E11 replay requires exactly one GPU")
        torch.manual_seed(SEED)
        device = torch.device("cuda", 0)
        native = configure_native_runtime()
        if (
            native != receipt["native_runtime"]
            or e11_environment_record(device) != receipt["environment"]
        ):
            raise RuntimeError("E11 replay runtime differs from registration")
        snapshot = snapshot_path()
        tokenizer = load_pinned_tokenizer(snapshot)
        population = selected_population(tokenizer)
        evidence = frozen_e10()
        bundle = load_bundle(BUNDLE_PATH, evidence["bundle"])
        model = load_pinned_model(snapshot, device)
        invariants = assert_model_invariants(model)
        exact_hidden = final_hidden_batch(
            model,
            tokenizer,
            population["prompt_order"],
            device,
            batch_size=MODEL_FORWARD_BATCH,
        )
        locality_hidden = final_hidden_batch(
            model,
            tokenizer,
            population["query_prompt_order"],
            device,
            batch_size=MODEL_FORWARD_BATCH,
        )
        measurement = cross_node_measurement(
            model,
            population["cases"],
            exact_hidden,
            population["query_rows"],
            locality_hidden,
            bundle,
            evidence["bundle"],
            evidence["selected_indices"],
            device,
        )
        scientific = {
            "model_invariants": invariants,
            "admission_scientific_sha256": admission["scientific_sha256"],
            "measurement_exact_match": measurement
            == admission["scientific"]["measurement"],
            "measurement": measurement,
        }
        payload = {
            "schema": RESULT_SCHEMA,
            "status": "COMPLETE",
            "attempt_sha256": attempt_sha,
            "admission_sha256": admission_sha,
            "admission_scientific_sha256": admission["scientific_sha256"],
            "native_runtime": native,
            "scientific": scientific,
            "scientific_sha256": scientific_sha256(scientific),
            "runtime_seconds": time.time() - started,
        }
        failures = result_validity_failures(payload, admission)
        atomic_create_json(RESULT_PATH, payload)
        print(f"COMPLETE replay: {RESULT_PATH}")
        if failures:
            print("KNOWN VALIDITY FAILURES:")
            [print(f"- {failure}") for failure in failures]
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

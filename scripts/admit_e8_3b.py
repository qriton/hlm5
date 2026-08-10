"""Run the single-use E8 fresh full-path admission."""

# ruff: noqa: E402 -- direct script execution bootstraps the repository root.
from __future__ import annotations

import sys
import time
from pathlib import Path

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
from hlm5.e7c_runtime import raw_and_zca_directions
from hlm5.e8_contract import (
    ADMISSION_PATH,
    ADMISSION_SCHEMA,
    ATTEMPT_PATH,
    ATTEMPT_SCHEMA,
    CASE_SHA256,
    PROMPT_SHA256,
    PROTOCOL_COMMIT,
    PROTOCOL_SHA256,
    RESULT_PATH,
    SEED,
    VERDICT_PATH,
    admission_validity_failures,
    assert_outputs_absent,
    atomic_create_json,
    load_execution_receipt,
    scientific_sha256,
    selected_population,
)
from hlm5.e8_runtime import full_path_measurement


def run_after_attempt(
    *,
    started: float,
    receipt: dict[str, object],
    execution_sha: str,
    attempt_sha: str,
    native_runtime: dict[str, object],
    device: torch.device,
) -> None:
    snapshot = snapshot_path()
    tokenizer = load_pinned_tokenizer(snapshot)
    population = selected_population(tokenizer)
    model = load_pinned_model(snapshot, device)
    invariants = assert_model_invariants(model)
    head_native = model.get_output_embeddings().weight.detach()
    target_ids = torch.tensor(
        [case["target_id"] for case in population["cases"]],
        device=device,
        dtype=torch.long,
    )
    raw, candidate, basis = raw_and_zca_directions(head_native, target_ids)
    if basis != receipt["basis_and_directions"]:
        raise RuntimeError("E8 basis or direction hashes differ from registration")
    hiddens = final_hidden_batch(
        model,
        tokenizer,
        population["prompt_order"],
        device,
        batch_size=16,
    )
    hidden_by_prompt = {
        prompt: hiddens[index]
        for index, prompt in enumerate(population["prompt_order"])
    }
    measurement = full_path_measurement(
        model,
        population["cases"],
        population["prompt_order"],
        hidden_by_prompt,
        population["neutral_prompts"],
        population["whitening_prompts"],
        raw,
        candidate,
    )
    scientific = {
        "model_invariants": invariants,
        "basis_and_directions": basis,
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
        raise RuntimeError("E8 admission is invalid: " + "; ".join(failures))
    atomic_create_json(ADMISSION_PATH, payload)
    summary = measurement["summary"]
    print(f"COMPLETE admission: {ADMISSION_PATH}")
    print(
        "full path: "
        f"eligible={summary['eligible_count']}, raw={summary['raw_admitted']}, "
        f"candidate={summary['candidate_admitted']}, "
        f"candidate_success={summary['candidate_gate']['admitted_exact_target_success']}, "
        f"false_gates={summary['candidate_gate']['false_gate_applications']}, "
        f"rollback={summary['rollback']['bit_identical_count']}/"
        f"{summary['rollback']['prompt_count']}"
    )


def main() -> None:
    started = time.time()
    assert_outputs_absent([ATTEMPT_PATH, ADMISSION_PATH, RESULT_PATH, VERDICT_PATH])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("registered E8 admission requires exactly one GPU")
    torch.manual_seed(SEED)
    device = torch.device("cuda", 0)
    native_runtime = configure_native_runtime()
    receipt, execution_sha = load_execution_receipt()
    if native_runtime != receipt.get("native_runtime"):
        raise RuntimeError("E8 admission arithmetic differs from registration")
    if environment_record(device) != receipt.get("environment"):
        raise RuntimeError("E8 admission environment differs from registration")
    attempt = {
        "schema": ATTEMPT_SCHEMA,
        "status": "POOL_SPENT",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": receipt["implementation_commit"],
        "execution_receipt_sha256": execution_sha,
        "case_sha256": CASE_SHA256,
        "prompt_sha256": PROMPT_SHA256,
        "pre_admission_outputs_absent": True,
    }
    atomic_create_json(ATTEMPT_PATH, attempt)
    attempt_sha = sha256_file(ATTEMPT_PATH)
    try:
        run_after_attempt(
            started=started,
            receipt=receipt,
            execution_sha=execution_sha,
            attempt_sha=attempt_sha,
            native_runtime=native_runtime,
            device=device,
        )
    except Exception as error:
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

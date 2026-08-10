"""New-process exact replay for the registered E8 full-path gate."""

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
from hlm5.e7c_runtime import raw_and_zca_directions
from hlm5.e8_contract import (
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
from hlm5.e8_runtime import full_path_measurement


def _row_hashes(rows: list[dict[str, Any]], *fields: str) -> list[list[Any]]:
    """Return the frozen ordered hash projection for exact replay checks."""
    return [[row.get(field) for field in fields] for row in rows]


def _arm_hashes_match(actual: dict[str, Any], frozen: dict[str, Any]) -> bool:
    direct_fields = ("baseline_logits_sha256", "native_logits_sha256")
    gate_fields = (
        "delta_sha256",
        "baseline_logits_sha256",
        "adapted_logits_sha256",
    )
    return _row_hashes(actual["direct_rows"], *direct_fields) == _row_hashes(
        frozen["direct_rows"], *direct_fields
    ) and _row_hashes(actual["gate_rows"], *gate_fields) == _row_hashes(
        frozen["gate_rows"], *gate_fields
    )


def _rollback_hashes_match(actual: dict[str, Any], frozen: dict[str, Any]) -> bool:
    fields = ("delta_sha256", "baseline_logits_sha256", "adapted_logits_sha256")
    return _row_hashes(actual["rows"], *fields) == _row_hashes(frozen["rows"], *fields)


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
    frozen = admission["scientific"]
    if (
        basis != receipt["basis_and_directions"]
        or basis != frozen["basis_and_directions"]
    ):
        raise RuntimeError("E8 replay basis or direction hashes differ")

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
    frozen_measurement = frozen["measurement"]
    raw_hashes_match = _arm_hashes_match(
        measurement["arms"]["raw"], frozen_measurement["arms"]["raw"]
    )
    candidate_hashes_match = _arm_hashes_match(
        measurement["arms"]["candidate"],
        frozen_measurement["arms"]["candidate"],
    )
    rollback_hashes_match = _rollback_hashes_match(
        measurement["rollback"], frozen_measurement["rollback"]
    )
    scientific = {
        "model_invariants": invariants,
        "basis_and_directions": basis,
        "admission_scientific_sha256": admission["scientific_sha256"],
        "measurement_exact_match": measurement == frozen_measurement,
        "raw_full_logit_hashes_match": raw_hashes_match,
        "candidate_full_logit_hashes_match": candidate_hashes_match,
        "rollback_full_logit_hashes_match": rollback_hashes_match,
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
            raise RuntimeError("registered E8 replay requires exactly one GPU")
        torch.manual_seed(SEED)
        device = torch.device("cuda", 0)
        native_runtime = configure_native_runtime()
        if native_runtime != receipt.get("native_runtime"):
            raise RuntimeError("E8 replay arithmetic differs from registration")
        if environment_record(device) != receipt.get("environment"):
            raise RuntimeError("E8 replay environment differs from registration")
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

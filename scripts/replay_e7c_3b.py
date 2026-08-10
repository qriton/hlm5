"""New-process exact replay for both registered E7c arms."""

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

from hlm5.e7_contract import snapshot_path, stable_json_sha256
from hlm5.e7_runtime import (
    assert_model_invariants,
    final_hidden_batch,
    load_pinned_model,
    load_pinned_tokenizer,
)
from hlm5.e7b_contract import PROMPT, PROMPT_SHA256
from hlm5.e7b_runtime import (
    GEOMETRIC_REACHABLE,
    configure_native_runtime,
    environment_record,
    summarize_rows,
)
from hlm5.e7c_contract import (
    RESULT_PATH,
    RESULT_SCHEMA,
    SEED,
    VERDICT_PATH,
    assert_outputs_absent,
    atomic_create_json,
    dual_summary,
    load_admission,
    load_attempt,
    load_execution_receipt,
    result_validity_failures,
    scientific_sha256,
    target_pools,
)
from hlm5.e7c_runtime import evaluate_arm, raw_and_zca_directions


def token_label(tokenizer: object, token_id: int) -> str:
    return str(tokenizer.convert_ids_to_tokens(int(token_id))).replace("Ġ", " ")


def native_hashes_match(
    actual: list[dict[str, Any]], frozen: list[dict[str, Any]]
) -> bool:
    return all(
        row.get("native_logits_sha256") == prior.get("native_logits_sha256")
        for row, prior in zip(actual, frozen, strict=True)
        if row.get("geometry_class") == GEOMETRIC_REACHABLE
    )


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
    _e7, _e7b, fresh_pool = target_pools(tokenizer)
    model = load_pinned_model(snapshot, device)
    invariants = assert_model_invariants(model)
    head_native = model.get_output_embeddings().weight.detach()
    target_ids = torch.tensor(fresh_pool, device=device, dtype=torch.long)
    raw, candidate, basis = raw_and_zca_directions(head_native, target_ids)
    frozen_scientific = admission["scientific"]
    if (
        basis != receipt["basis_and_directions"]
        or basis != frozen_scientific["basis_and_directions"]
    ):
        raise RuntimeError("E7c replay basis or direction hashes differ")
    hidden = final_hidden_batch(model, tokenizer, [PROMPT], device, batch_size=1)[0]
    head64 = head_native.to(torch.float64)
    raw_rows = evaluate_arm(model, head64, hidden, target_ids, raw)
    candidate_rows = evaluate_arm(model, head64, hidden, target_ids, candidate)
    for rows in (raw_rows, candidate_rows):
        for row in rows:
            row["target_text"] = token_label(tokenizer, row["target_id"])
    arms = {
        "raw": {"rows": raw_rows, "summary": summarize_rows(raw_rows)},
        "candidate": {
            "rows": candidate_rows,
            "summary": summarize_rows(candidate_rows),
        },
    }
    frozen_arms = frozen_scientific["arms"]
    scientific = {
        "prompt": PROMPT,
        "prompt_sha256": PROMPT_SHA256,
        "target_pool_count": len(fresh_pool),
        "target_pool_sha256": stable_json_sha256(fresh_pool),
        "model_invariants": invariants,
        "basis_and_directions": basis,
        "admission_scientific_sha256": admission["scientific_sha256"],
        "raw_all_rows_match": arms["raw"] == frozen_arms["raw"],
        "candidate_all_rows_match": arms["candidate"] == frozen_arms["candidate"],
        "raw_all_native_hashes_match": native_hashes_match(
            raw_rows, frozen_arms["raw"]["rows"]
        ),
        "candidate_all_native_hashes_match": native_hashes_match(
            candidate_rows, frozen_arms["candidate"]["rows"]
        ),
        "arms": arms,
        "summary": dual_summary(raw_rows, candidate_rows),
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
            raise RuntimeError("registered E7c replay requires exactly one GPU")
        torch.manual_seed(SEED)
        device = torch.device("cuda", 0)
        native_runtime = configure_native_runtime()
        if native_runtime != receipt.get("native_runtime"):
            raise RuntimeError("E7c replay arithmetic differs from registration")
        if environment_record(device) != receipt.get("environment"):
            raise RuntimeError("E7c replay environment differs from registration")
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

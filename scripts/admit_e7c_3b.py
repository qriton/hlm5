"""Run the single-use E7c raw-versus-ZCA confirmation admission."""

# ruff: noqa: E402 -- direct script execution bootstraps the repository root.
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

import torch

from hlm5.e7_contract import sha256_file, snapshot_path, stable_json_sha256
from hlm5.e7_runtime import (
    assert_model_invariants,
    final_hidden_batch,
    load_pinned_model,
    load_pinned_tokenizer,
)
from hlm5.e7b_contract import PROMPT, PROMPT_SHA256
from hlm5.e7b_runtime import (
    configure_native_runtime,
    environment_record,
    summarize_rows,
)
from hlm5.e7c_contract import (
    ADMISSION_PATH,
    ADMISSION_SCHEMA,
    ATTEMPT_PATH,
    ATTEMPT_SCHEMA,
    PROTOCOL_COMMIT,
    PROTOCOL_SHA256,
    RESULT_PATH,
    SEED,
    TARGET_POOL_SHA256,
    VERDICT_PATH,
    admission_validity_failures,
    assert_outputs_absent,
    atomic_create_json,
    dual_summary,
    load_execution_receipt,
    scientific_sha256,
    target_pools,
)
from hlm5.e7c_runtime import evaluate_arm, raw_and_zca_directions


def token_label(tokenizer: object, token_id: int) -> str:
    return str(tokenizer.convert_ids_to_tokens(int(token_id))).replace("Ġ", " ")


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
    _e7, _e7b, fresh_pool = target_pools(tokenizer)
    model = load_pinned_model(snapshot, device)
    invariants = assert_model_invariants(model)
    head_native = model.get_output_embeddings().weight.detach()
    target_ids = torch.tensor(fresh_pool, device=device, dtype=torch.long)
    raw, candidate, basis = raw_and_zca_directions(head_native, target_ids)
    if basis != receipt["basis_and_directions"]:
        raise RuntimeError("E7c basis or direction hashes differ from registration")
    hidden = final_hidden_batch(model, tokenizer, [PROMPT], device, batch_size=1)[0]
    head64 = head_native.to(torch.float64)
    raw_rows = evaluate_arm(model, head64, hidden, target_ids, raw)
    candidate_rows = evaluate_arm(model, head64, hidden, target_ids, candidate)
    for rows in (raw_rows, candidate_rows):
        for row in rows:
            row["target_text"] = token_label(tokenizer, row["target_id"])
    raw_arm = {"rows": raw_rows, "summary": summarize_rows(raw_rows)}
    candidate_arm = {
        "rows": candidate_rows,
        "summary": summarize_rows(candidate_rows),
    }
    scientific = {
        "prompt": PROMPT,
        "prompt_sha256": PROMPT_SHA256,
        "target_pool_count": len(fresh_pool),
        "target_pool_sha256": stable_json_sha256(fresh_pool),
        "model_invariants": invariants,
        "basis_and_directions": basis,
        "certificate_boundary_dtypes": {
            "head": "float64",
            "hidden": "float64",
            "raw_directions": "float64",
            "candidate_directions": "float64",
        },
        "arms": {"raw": raw_arm, "candidate": candidate_arm},
        "summary": dual_summary(raw_rows, candidate_rows),
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
        raise RuntimeError("E7c admission is invalid: " + "; ".join(failures))
    atomic_create_json(ADMISSION_PATH, payload)
    summary = scientific["summary"]
    print(f"COMPLETE admission: {ADMISSION_PATH}")
    print(
        "coverage: "
        f"raw={summary['raw']['geometrically_reachable']}/{len(fresh_pool)}, "
        f"candidate={summary['candidate']['geometrically_reachable']}/{len(fresh_pool)}, "
        f"native={summary['candidate']['native_admitted']}/"
        f"{summary['candidate']['geometrically_reachable']}"
    )


def main() -> None:
    started = time.time()
    assert_outputs_absent([ATTEMPT_PATH, ADMISSION_PATH, RESULT_PATH, VERDICT_PATH])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("registered E7c admission requires exactly one GPU")
    torch.manual_seed(SEED)
    device = torch.device("cuda", 0)
    native_runtime = configure_native_runtime()
    receipt, execution_sha = load_execution_receipt()
    if native_runtime != receipt.get("native_runtime"):
        raise RuntimeError("E7c admission arithmetic differs from registration")
    if environment_record(device) != receipt.get("environment"):
        raise RuntimeError("E7c admission environment differs from registration")
    attempt = {
        "schema": ATTEMPT_SCHEMA,
        "status": "POOL_SPENT",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": receipt["implementation_commit"],
        "execution_receipt_sha256": execution_sha,
        "target_pool_sha256": TARGET_POOL_SHA256,
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

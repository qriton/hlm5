"""Run the single-use E7b geometric plus strict-native admission."""

# ruff: noqa: E402 -- direct script execution bootstraps the repository root.
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

import torch

from hlm5.e7_contract import (
    float64_boundary_record,
    sha256_file,
    snapshot_path,
    stable_json_sha256,
)
from hlm5.e7_runtime import (
    assert_model_invariants,
    final_hidden_batch,
    load_pinned_model,
    load_pinned_tokenizer,
)
from hlm5.e7b_contract import (
    ADMISSION_PATH,
    ADMISSION_SCHEMA,
    ATTEMPT_PATH,
    ATTEMPT_SCHEMA,
    PROMPT,
    PROMPT_SHA256,
    PROTOCOL_COMMIT,
    PROTOCOL_SHA256,
    RESULT_PATH,
    SEED,
    TARGET_POOL_SHA256,
    VERDICT_PATH,
    admission_validity_failures,
    assert_outputs_absent,
    atomic_create_json,
    load_execution_receipt,
    scientific_sha256,
    target_pools,
    verify_e7_inputs,
)
from hlm5.e7b_runtime import (
    configure_native_runtime,
    environment_record,
    geometry_rows,
    native_rows,
    summarize_rows,
)


def token_label(tokenizer: object, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    return str(token).replace("Ġ", " ")


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
    _e7_pool, fresh_pool = target_pools(tokenizer)
    model = load_pinned_model(snapshot, device)
    invariants = assert_model_invariants(model)
    hidden = final_hidden_batch(
        model,
        tokenizer,
        [PROMPT],
        device,
        batch_size=1,
    )[0]
    head64 = model.get_output_embeddings().weight.detach().to(torch.float64)
    target_ids = torch.tensor(fresh_pool, device=device, dtype=torch.long)
    geometry, directions, slopes_cpu = geometry_rows(head64, hidden, target_ids)
    rows = native_rows(model, hidden, directions, geometry)
    for row in rows:
        row["target_text"] = token_label(tokenizer, row["target_id"])
    summary = summarize_rows(rows)
    e7_inputs = verify_e7_inputs()
    scientific = {
        "prompt": PROMPT,
        "prompt_sha256": PROMPT_SHA256,
        "target_pool_count": len(fresh_pool),
        "target_pool_sha256": stable_json_sha256(fresh_pool),
        "model_invariants": invariants,
        "certificate_boundary_dtypes": float64_boundary_record(
            head=head64,
            hidden=hidden.to(torch.float64),
            directions=directions,
            slopes=slopes_cpu,
        ),
        "e7_sentinel": e7_inputs["sentinel"],
        "rows": rows,
        "summary": summary,
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
        raise RuntimeError("E7b admission is invalid: " + "; ".join(failures))
    atomic_create_json(ADMISSION_PATH, payload)
    print(f"COMPLETE admission: {ADMISSION_PATH}")
    print(
        "coverage: "
        f"geometry={summary['geometrically_reachable']}/{len(fresh_pool)}, "
        f"native={summary['native_admitted']}/{summary['geometrically_reachable']}"
    )


def main() -> None:
    started = time.time()
    assert_outputs_absent([ATTEMPT_PATH, ADMISSION_PATH, RESULT_PATH, VERDICT_PATH])
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("registered E7b admission requires CUDA")
    torch.manual_seed(SEED)
    device = torch.device("cuda", 0)
    native_runtime = configure_native_runtime()
    receipt, execution_sha = load_execution_receipt()
    if native_runtime != receipt.get("native_runtime"):
        raise RuntimeError("E7b admission arithmetic differs from registration")
    environment = environment_record(device)
    if environment != receipt.get("environment"):
        raise RuntimeError("E7b admission environment differs from registration")

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

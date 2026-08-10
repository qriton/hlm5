"""New-process full-reachable-set replay for the registered E7b admission."""

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

from hlm5.e7_contract import sha256_file, snapshot_path, stable_json_sha256
from hlm5.e7_runtime import (
    assert_model_invariants,
    final_hidden_batch,
    load_pinned_model,
    load_pinned_tokenizer,
)
from hlm5.e7b_contract import (
    ADMISSION_PATH,
    PROMPT,
    PROMPT_SHA256,
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
    target_pools,
)
from hlm5.e7b_runtime import (
    GEOMETRIC_REACHABLE,
    NATIVE_ADMIT,
    configure_native_runtime,
    environment_record,
    geometry_rows,
    native_rows,
    summarize_rows,
)


GEOMETRY_FIELDS = (
    "pool_index",
    "target_id",
    "L",
    "U",
    "hard_blocker",
    "hard_blocker_id",
    "beta",
    "float64_margin",
    "geometry_class",
)
NATIVE_VALUE_FIELDS = (
    "native_target_logit",
    "native_competitor_id",
    "native_competitor_logit",
    "native_margin",
    "native_prediction",
    "native_logits_dtype",
    "native_logits_shape",
    "native_batch_start",
    "native_batch_size",
    "native_batch_offset",
)


def selected(row: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: row.get(field) for field in fields}


def token_label(tokenizer: object, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    return str(token).replace("Ġ", " ")


def run_replay(
    *,
    started: float,
    admission: dict[str, Any],
    admission_sha: str,
    attempt_sha: str,
    native_runtime: dict[str, Any],
    device: torch.device,
) -> None:
    admission_rows = admission["scientific"]["rows"]
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
    recomputed_geometry, directions, _slopes_cpu = geometry_rows(
        head64,
        hidden,
        target_ids,
    )
    geometry_all_match = len(recomputed_geometry) == len(admission_rows) and all(
        selected(recomputed, GEOMETRY_FIELDS) == selected(frozen, GEOMETRY_FIELDS)
        for recomputed, frozen in zip(recomputed_geometry, admission_rows, strict=True)
    )

    # Freeze every reachable beta to the admission receipt before native replay.
    for recomputed, frozen in zip(
        recomputed_geometry,
        admission_rows,
        strict=True,
    ):
        if recomputed["geometry_class"] == GEOMETRIC_REACHABLE:
            recomputed["beta"] = frozen.get("beta")
    replay_rows = native_rows(model, hidden, directions, recomputed_geometry)
    for row in replay_rows:
        row["target_text"] = token_label(tokenizer, row["target_id"])
    replay_summary = summarize_rows(replay_rows)

    native_all_decisions_match = all(
        replay.get("decision") == frozen.get("decision")
        for replay, frozen in zip(replay_rows, admission_rows, strict=True)
    )
    admitted_pairs = [
        (replay, frozen)
        for replay, frozen in zip(replay_rows, admission_rows, strict=True)
        if frozen.get("decision") == NATIVE_ADMIT
    ]
    admitted_native_values_match = all(
        selected(replay, NATIVE_VALUE_FIELDS) == selected(frozen, NATIVE_VALUE_FIELDS)
        for replay, frozen in admitted_pairs
    )
    admitted_native_hashes_match = all(
        replay.get("native_logits_sha256") == frozen.get("native_logits_sha256")
        for replay, frozen in admitted_pairs
    )
    scientific = {
        "prompt": PROMPT,
        "prompt_sha256": PROMPT_SHA256,
        "target_pool_count": len(fresh_pool),
        "target_pool_sha256": stable_json_sha256(fresh_pool),
        "model_invariants": invariants,
        "admission_scientific_sha256": admission["scientific_sha256"],
        "geometry_all_rows_match": geometry_all_match,
        "native_all_decisions_match": native_all_decisions_match,
        "admitted_native_values_match": admitted_native_values_match,
        "admitted_native_hashes_match": admitted_native_hashes_match,
        "rows": replay_rows,
        "summary": replay_summary,
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
    print(f"admission sha256: {sha256_file(ADMISSION_PATH)}")
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
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("registered E7b replay requires CUDA")
        torch.manual_seed(SEED)
        device = torch.device("cuda", 0)
        native_runtime = configure_native_runtime()
        if native_runtime != receipt.get("native_runtime"):
            raise RuntimeError("E7b replay arithmetic differs from registration")
        environment = environment_record(device)
        if environment != receipt.get("environment"):
            raise RuntimeError("E7b replay environment differs from registration")
        run_replay(
            started=started,
            admission=admission,
            admission_sha=admission_sha,
            attempt_sha=attempt_sha,
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

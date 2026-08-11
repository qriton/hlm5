"""Train and evaluate the single registered HLM5 E14 attempt."""

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
from hlm5.e7_runtime import assert_model_invariants, final_hidden_batch, load_pinned_model, load_pinned_tokenizer
from hlm5.e7b_runtime import configure_native_runtime
from hlm5.e14_contract import (
    ADMISSION_PATH,
    ADMISSION_SCHEMA,
    ATTEMPT_PATH,
    ATTEMPT_SCHEMA,
    BUNDLE_PATH,
    EVAL_EXPECTED,
    LOCALITY_EXPECTED,
    PROTOCOL_COMMIT,
    PROTOCOL_SHA256,
    RESULT_PATH,
    TRAIN_EXPECTED,
    VERDICT_PATH,
    assert_outputs_absent,
    atomic_create_bundle,
    atomic_create_json,
    load_execution_receipt,
    scientific_sha256,
    selected_population,
)
from hlm5.e14_runtime import (
    COSINE,
    HLM,
    MODEL_FORWARD_BATCH,
    bundle_manifest,
    bundle_tensors,
    environment_record,
    evaluate_arms,
    key_operator,
    locality_prompt_order,
    model_parameter_sha256,
    train_router,
    training_config,
    whiten,
)


def _hiddens(model: Any, tokenizer: Any, population: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    train_prompts = [row["exact"] for row in population["train"]] + [
        row["paraphrase"] for row in population["train"]
    ]
    eval_prompts = [row["exact"] for row in population["evaluation"]] + [
        row["paraphrase"] for row in population["evaluation"]
    ]
    train = final_hidden_batch(model, tokenizer, train_prompts, device, batch_size=MODEL_FORWARD_BATCH)
    evaluation = final_hidden_batch(model, tokenizer, eval_prompts, device, batch_size=MODEL_FORWARD_BATCH)
    calibration = final_hidden_batch(
        model, tokenizer, population["calibration_prompts"], device, batch_size=MODEL_FORWARD_BATCH
    )
    locality = final_hidden_batch(
        model, tokenizer, locality_prompt_order(population["locality"]), device, batch_size=MODEL_FORWARD_BATCH
    )
    return {
        "train_exact": train[: len(population["train"])],
        "train_paraphrase": train[len(population["train"]) :],
        "eval_exact": evaluation[: len(population["evaluation"])],
        "eval_paraphrase": evaluation[len(population["evaluation"]) :],
        "calibration": calibration,
        "locality": locality,
    }


def run_after_attempt(
    *,
    started: float,
    model: Any,
    tokenizer: Any,
    population: dict[str, Any],
    receipt: dict[str, Any],
    receipt_sha: str,
    attempt_sha: str,
    native_runtime: dict[str, Any],
    device: torch.device,
) -> None:
    trunk_before = model_parameter_sha256(model)
    hidden = _hiddens(model, tokenizer, population, device)
    fit = torch.cat(
        [hidden["train_exact"], hidden["eval_exact"], hidden["calibration"]], dim=0
    )
    key_mean, key_transform, key_record = key_operator(fit)
    train_exact = whiten(hidden["train_exact"], key_mean, key_transform)
    train_paraphrase = whiten(hidden["train_paraphrase"], key_mean, key_transform)
    routers = {}
    training = {}
    for arm in (HLM, COSINE):
        routers[arm], training[arm] = train_router(arm, train_exact, train_paraphrase)
    bundle = bundle_tensors(routers, key_mean, key_transform)
    manifest = bundle_manifest(bundle)
    bundle_sha = atomic_create_bundle(BUNDLE_PATH, bundle)
    evaluation = evaluate_arms(
        model,
        population["evaluation"],
        hidden["eval_exact"],
        hidden["eval_paraphrase"],
        population["locality"],
        hidden["locality"],
        key_mean,
        key_transform,
        routers,
    )
    trunk_after = model_parameter_sha256(model)
    measurement = {
        "population": {
            "train": population["train_record"],
            "evaluation": population["evaluation_record"],
            "locality": population["locality_record"],
        },
        "training_config": training_config(),
        "key_whitening": key_record,
        "training": training,
        "bundle": {"file_sha256": bundle_sha, "tensors": manifest},
        "evaluation": evaluation,
        "trunk_parameter_sha256_before": trunk_before,
        "trunk_parameter_sha256_after": trunk_after,
    }
    scientific = {
        "model_invariants": assert_model_invariants(model),
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
    print(f"COMPLETE E14 admission: {ADMISSION_PATH}")


def main() -> None:
    started = time.time()
    assert_outputs_absent([ATTEMPT_PATH, BUNDLE_PATH, ADMISSION_PATH, RESULT_PATH, VERDICT_PATH])
    receipt, receipt_sha = load_execution_receipt()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("E14 admission requires exactly one GPU")
    device = torch.device("cuda", 0)
    native_runtime = configure_native_runtime()
    if native_runtime != receipt["native_runtime"]:
        raise RuntimeError("E14 admission arithmetic differs from registration")
    if environment_record(device) != receipt["environment"]:
        raise RuntimeError("E14 admission environment differs from registration")
    tokenizer = load_pinned_tokenizer(snapshot_path())
    population = selected_population(tokenizer)
    model = load_pinned_model(snapshot_path(), device)
    attempt = {
        "schema": ATTEMPT_SCHEMA,
        "status": "POOL_SPENT",
        "execution_receipt_sha256": receipt_sha,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": receipt["implementation_commit"],
        "train_row_sha256": TRAIN_EXPECTED["row_sha256"],
        "evaluation_row_sha256": EVAL_EXPECTED["row_sha256"],
        "locality_row_sha256": LOCALITY_EXPECTED["row_sha256"],
        "pre_admission_outputs_absent": True,
        "burn_unix_seconds": time.time(),
    }
    atomic_create_json(ATTEMPT_PATH, attempt)
    attempt_sha = sha256_file(ATTEMPT_PATH)
    try:
        run_after_attempt(
            started=started,
            model=model,
            tokenizer=tokenizer,
            population=population,
            receipt=receipt,
            receipt_sha=receipt_sha,
            attempt_sha=attempt_sha,
            native_runtime=native_runtime,
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

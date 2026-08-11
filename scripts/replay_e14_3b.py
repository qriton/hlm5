"""New-process exact evaluation replay for HLM5 E14."""

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

from hlm5.e7_contract import snapshot_path
from hlm5.e7_runtime import assert_model_invariants, final_hidden_batch, load_pinned_model, load_pinned_tokenizer
from hlm5.e7b_runtime import configure_native_runtime
from hlm5.e14_contract import (
    BUNDLE_PATH,
    RESULT_PATH,
    RESULT_SCHEMA,
    VERDICT_PATH,
    assert_outputs_absent,
    atomic_create_json,
    load_admission,
    load_attempt,
    load_bundle,
    load_execution_receipt,
    scientific_sha256,
    selected_population,
)
from hlm5.e14_runtime import (
    MODEL_FORWARD_BATCH,
    bundle_manifest,
    environment_record,
    evaluate_arms,
    key_operator,
    locality_prompt_order,
    model_parameter_sha256,
    routers_from_bundle,
    training_config,
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
        "eval_exact": evaluation[: len(population["evaluation"])],
        "eval_paraphrase": evaluation[len(population["evaluation"]) :],
        "calibration": calibration,
        "locality": locality,
    }


def main() -> None:
    started = time.time()
    assert_outputs_absent([RESULT_PATH, VERDICT_PATH])
    receipt, _receipt_sha = load_execution_receipt()
    admission, admission_sha = load_admission()
    _attempt, attempt_sha = load_attempt()
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("E14 replay requires exactly one GPU")
        device = torch.device("cuda", 0)
        native_runtime = configure_native_runtime()
        if native_runtime != receipt["native_runtime"]:
            raise RuntimeError("E14 replay arithmetic differs from registration")
        if environment_record(device) != receipt["environment"]:
            raise RuntimeError("E14 replay environment differs from registration")
        population = selected_population(load_pinned_tokenizer(snapshot_path()))
        tokenizer = load_pinned_tokenizer(snapshot_path())
        model = load_pinned_model(snapshot_path(), device)
        trunk_before = model_parameter_sha256(model)
        hidden = _hiddens(model, tokenizer, population, device)
        fit = torch.cat(
            [hidden["train_exact"], hidden["eval_exact"], hidden["calibration"]], dim=0
        )
        recomputed_mean, recomputed_transform, key_record = key_operator(fit)
        frozen = admission["scientific"]["measurement"]
        payload = load_bundle(BUNDLE_PATH, frozen["bundle"]["tensors"])
        if bundle_manifest(payload) != frozen["bundle"]["tensors"]:
            raise RuntimeError("E14 loaded bundle manifest drift")
        key_mean = payload["key_mean"].to(device)
        key_transform = payload["key_transform"].to(device)
        if not torch.equal(key_mean, recomputed_mean) or not torch.equal(
            key_transform, recomputed_transform
        ):
            raise RuntimeError("E14 whitening did not replay exactly")
        routers = routers_from_bundle(payload, device)
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
            "training": frozen["training"],
            "bundle": frozen["bundle"],
            "evaluation": evaluation,
            "trunk_parameter_sha256_before": trunk_before,
            "trunk_parameter_sha256_after": trunk_after,
        }
        bundle_exact = frozen["bundle"]["file_sha256"] == __import__(
            "hlm5.e7_contract", fromlist=["sha256_file"]
        ).sha256_file(BUNDLE_PATH)
        scientific = {
            "admission_scientific_sha256": admission["scientific_sha256"],
            "model_invariants": assert_model_invariants(model),
            "measurement_exact_match": measurement == frozen,
            "bundle_exact_match": bundle_exact,
            "measurement": measurement,
        }
        result = {
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
        atomic_create_json(RESULT_PATH, result)
        print(f"COMPLETE E14 replay: {RESULT_PATH}")
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

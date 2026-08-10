"""Preflight, register, and run the frozen NKT-1 development gate."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer

from .contract import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DEFAULT_CACHE_PATH,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_LICENSE_PATH,
    DEFAULT_TOKENIZER_PATH,
    DEGREE,
    DEPLOYED_TAU_COS,
    MANIFEST_PATH,
    PACKAGE_ROOT,
    PASS_BARS,
    PREFLIGHT_PATH,
    PREREG_PATH,
    REGISTRATION_PATH,
    REPO_ROOT,
    RESULT_PATH,
    ROWS_PATH,
    SHRINKAGE,
    SNAPSHOT_ROOT,
    SUPPORT_PER_INTENT,
    file_hashes,
    git_head,
    read_json,
    sha256_file,
    stable_sha256,
    verify_expected_file,
    write_json,
)
from .geometry import (
    ARM_NAMES,
    apply_arm,
    development_verdict,
    evaluate_arm,
    fit_key_operators,
    paired_intent_bootstrap,
    tensor_sha256,
)
from .prepare_massive import (
    build_populations,
    population_sha256,
    verify_manifest,
)


HIDDEN_BATCH_SIZE = 128
SYNTHETIC_ANCHOR = "NKT one synthetic HLM5 routing anchor."
MODEL_SOURCE_PATHS = (
    SNAPSHOT_ROOT / "hlm5_lm.py",
    SNAPSHOT_ROOT / "hlm5_memory.py",
)


class HarnessInvalid(RuntimeError):
    """Raised when a binding implementation or provenance contract fails."""


def now_local() -> str:
    return dt.datetime.now().astimezone().isoformat()


def scientific_constants() -> dict[str, Any]:
    return {
        "schema": "hlm5-nkt1-scientific-constants-v1",
        "arms": list(ARM_NAMES),
        "candidate": "source_fisher",
        "degree": DEGREE,
        "shrinkage": SHRINKAGE,
        "support_per_intent": SUPPORT_PER_INTENT,
        "hidden_batch_size": HIDDEN_BATCH_SIZE,
        "hidden_dtype": "torch.float32",
        "scientific_device": "cpu",
        "scientific_dtype": "torch.float64",
        "deployed_tau_cos_diagnostic_only": DEPLOYED_TAU_COS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "numpy_quantile_method": "linear",
        "pass_bars": PASS_BARS,
        "no_test_encoding": True,
        "no_target_metric_fit": True,
        "no_recurrence": True,
    }


def environment_record(device: torch.device) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "tokenizers": getattr(sys.modules.get("tokenizers"), "__version__", None),
        "device": str(device),
        "gpu": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
        "cuda": torch.version.cuda,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def configure_determinism() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(20260810)
    np.random.seed(20260810)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260810)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def encode_prompt(tokenizer: Tokenizer, text: str, max_len: int) -> list[int]:
    ids = list(tokenizer.encode(text).ids)
    if ids and ids[0] == 2:
        ids = ids[1:]
    if ids and ids[-1] == 3:
        ids = ids[:-1]
    if not ids:
        raise HarnessInvalid("tokenizer produced an empty utterance")
    if len(ids) > max_len:
        raise HarnessInvalid(f"utterance length {len(ids)} exceeds context {max_len}")
    return ids


def load_model_and_tokenizer(
    checkpoint_path: Path,
    tokenizer_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Tokenizer, dict[str, Any]]:
    verify_expected_file(checkpoint_path, "checkpoint")
    verify_expected_file(tokenizer_path, "tokenizer")
    snapshot = str(SNAPSHOT_ROOT)
    if snapshot not in sys.path:
        sys.path.insert(0, snapshot)
    from hlm5_lm import HLM5LM  # type: ignore

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("architecture") != "HLM5LM":
        raise HarnessInvalid("checkpoint architecture is not HLM5LM")
    cfg = dict(checkpoint["cfg"])
    if cfg != {
        "dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "max_len": 1024,
        "memory_layer": False,
        "memory_size": 256,
    }:
        raise HarnessInvalid(f"checkpoint config changed: {cfg}")
    if int(checkpoint["vocab"]) != 65536:
        raise HarnessInvalid("checkpoint vocabulary changed")
    model = HLM5LM(vocab=int(checkpoint["vocab"]), **cfg)
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise HarnessInvalid(f"strict load mismatch: {incompatible}")
    model.to(device)
    model.eval()
    if model.training:
        raise HarnessInvalid("model did not enter evaluation mode")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return (
        model,
        tokenizer,
        {
            "cfg": cfg,
            "vocab": int(checkpoint["vocab"]),
            "architecture": checkpoint["architecture"],
            "step": int(checkpoint["step"]),
            "val_ppl": float(checkpoint["val_ppl"]),
        },
    )


@torch.inference_mode()
def extract_hiddens(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = HIDDEN_BATCH_SIZE,
) -> tuple[torch.Tensor, dict[str, Any]]:
    encoded = [
        encode_prompt(tokenizer, row["utterance"], int(model.max_len)) for row in rows
    ]
    by_length: dict[int, list[int]] = {}
    for index, ids in enumerate(encoded):
        by_length.setdefault(len(ids), []).append(index)
    output = torch.empty(
        (len(rows), int(model.dim)),
        dtype=torch.float32,
        device="cpu",
    )
    for length in sorted(by_length):
        indices = by_length[length]
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            tokens = torch.tensor(
                [encoded[index] for index in selected],
                dtype=torch.long,
                device=device,
            )
            hidden = (
                model.hidden(tokens)[:, -1]
                .detach()
                .to(device="cpu", dtype=torch.float32)
            )
            output[torch.tensor(selected, dtype=torch.long)] = hidden
    if not bool(torch.isfinite(output).all()):
        raise HarnessInvalid("non-finite hidden state")
    return output, {
        "row_count": len(rows),
        "dimension": int(output.shape[1]),
        "minimum_tokens": min(map(len, encoded)),
        "maximum_tokens": max(map(len, encoded)),
        "length_groups": len(by_length),
        "hidden_sha256": tensor_sha256(output),
    }


def run_unit_tests() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(PACKAGE_ROOT / "tests"),
        "-p",
        "test_*.py",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise HarnessInvalid(
            "targeted tests failed\n" + completed.stdout + "\n" + completed.stderr
        )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_sha256": stable_sha256(completed.stdout),
        "stderr_sha256": stable_sha256(completed.stderr),
        "summary": completed.stderr.strip().splitlines()[-1]
        if completed.stderr.strip()
        else "passed",
    }


def artifact_record(
    data_path: Path,
    license_path: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
) -> dict[str, Any]:
    return {
        "data": {
            "path": str(data_path),
            "sha256": verify_expected_file(data_path, "data"),
        },
        "license": {
            "path": str(license_path),
            "sha256": verify_expected_file(license_path, "license"),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": verify_expected_file(checkpoint_path, "checkpoint"),
        },
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": verify_expected_file(tokenizer_path, "tokenizer"),
        },
        "preregistration": {
            "path": PREREG_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(PREREG_PATH),
        },
        "manifest": {
            "path": MANIFEST_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(MANIFEST_PATH),
        },
        "model_source": {
            path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
            for path in MODEL_SOURCE_PATHS
        },
        "implementation": file_hashes(),
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    configure_determinism()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise HarnessInvalid("registered CUDA device is unavailable")
    data_path = args.data.resolve()
    license_path = args.license.resolve()
    checkpoint_path = args.checkpoint.resolve()
    tokenizer_path = args.tokenizer.resolve()
    manifest = verify_manifest(data_path)
    tests = run_unit_tests()
    model, tokenizer, model_metadata = load_model_and_tokenizer(
        checkpoint_path, tokenizer_path, device
    )
    synthetic_rows = [{"utterance": SYNTHETIC_ANCHOR}]
    first, first_meta = extract_hiddens(model, tokenizer, synthetic_rows, device, 1)
    second, second_meta = extract_hiddens(model, tokenizer, synthetic_rows, device, 1)
    if not torch.equal(first, second):
        raise HarnessInvalid("synthetic anchor is not bit-repeatable")
    token_ids = encode_prompt(tokenizer, SYNTHETIC_ANCHOR, int(model.max_len))
    record = {
        "schema": "hlm5-nkt1-preflight-v1",
        "status": "READY_TO_REGISTER",
        "generated_at_local": now_local(),
        "git_head": git_head(),
        "environment": environment_record(device),
        "artifacts": artifact_record(
            data_path,
            license_path,
            checkpoint_path,
            tokenizer_path,
        ),
        "manifest_scientific_sha256": stable_sha256(manifest),
        "scientific_constants": scientific_constants(),
        "tests": tests,
        "model": model_metadata,
        "synthetic_anchor": {
            "text_sha256": stable_sha256(SYNTHETIC_ANCHOR),
            "token_ids_sha256": stable_sha256(token_ids),
            "hidden_sha256": tensor_sha256(first),
            "repeat_bit_exact": True,
            "first": first_meta,
            "second": second_meta,
        },
        "attestation": {
            "massive_utterances_encoded": False,
            "routing_outcomes_computed": False,
            "test_utterances_encoded": False,
            "only_trunk_input": "fixed synthetic anchor",
        },
    }
    write_json(PREFLIGHT_PATH, record)
    return record


def verify_preflight(record: dict[str, Any]) -> None:
    if record.get("schema") != "hlm5-nkt1-preflight-v1":
        raise HarnessInvalid("preflight schema mismatch")
    if record.get("status") != "READY_TO_REGISTER":
        raise HarnessInvalid("preflight is not ready")
    if record.get("git_head") != git_head():
        raise HarnessInvalid("preflight Git head is stale")
    if record.get("scientific_constants") != scientific_constants():
        raise HarnessInvalid("scientific constants changed after preflight")
    if record["artifacts"]["implementation"] != file_hashes():
        raise HarnessInvalid("implementation changed after preflight")
    if record["artifacts"]["preregistration"]["sha256"] != sha256_file(PREREG_PATH):
        raise HarnessInvalid("preregistration changed after preflight")
    if record["artifacts"]["manifest"]["sha256"] != sha256_file(MANIFEST_PATH):
        raise HarnessInvalid("manifest changed after preflight")
    if not record["synthetic_anchor"]["repeat_bit_exact"]:
        raise HarnessInvalid("synthetic repeatability did not pass")
    if any(
        record["attestation"].get(key)
        for key in (
            "massive_utterances_encoded",
            "routing_outcomes_computed",
            "test_utterances_encoded",
        )
    ):
        raise HarnessInvalid("preflight attestation reports outcome access")


def register_development() -> dict[str, Any]:
    if not PREFLIGHT_PATH.is_file():
        raise HarnessInvalid("preflight file is missing")
    preflight_record = read_json(PREFLIGHT_PATH)
    verify_preflight(preflight_record)
    protocol = scientific_constants()
    record = {
        "schema": "hlm5-nkt1-development-registration-v1",
        "status": "REGISTERED_BEFORE_OUTCOMES",
        "generated_at_local": now_local(),
        "git_head": git_head(),
        "preflight": {
            "path": PREFLIGHT_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(PREFLIGHT_PATH),
        },
        "artifacts": preflight_record["artifacts"],
        "environment": preflight_record["environment"],
        "manifest_scientific_sha256": preflight_record["manifest_scientific_sha256"],
        "scientific_constants": protocol,
        "protocol_sha256": stable_sha256(protocol),
        "timing_attestation": {
            "massive_utterances_encoded_before_registration": False,
            "routing_outcomes_computed_before_registration": False,
            "test_utterances_encoded": False,
        },
    }
    write_json(REGISTRATION_PATH, record)
    return record


def verify_registration(
    registration: dict[str, Any],
    data_path: Path,
    license_path: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    if registration.get("schema") != "hlm5-nkt1-development-registration-v1":
        raise HarnessInvalid("registration schema mismatch")
    if registration.get("status") != "REGISTERED_BEFORE_OUTCOMES":
        raise HarnessInvalid("registration status mismatch")
    if registration.get("git_head") != git_head():
        raise HarnessInvalid("registration Git head is stale")
    if registration.get("scientific_constants") != scientific_constants():
        raise HarnessInvalid("registered scientific constants changed")
    if registration.get("protocol_sha256") != stable_sha256(scientific_constants()):
        raise HarnessInvalid("registered protocol hash changed")
    if registration["preflight"]["sha256"] != sha256_file(PREFLIGHT_PATH):
        raise HarnessInvalid("registered preflight bytes changed")
    preflight_record = read_json(PREFLIGHT_PATH)
    verify_preflight(preflight_record)
    current_artifacts = artifact_record(
        data_path,
        license_path,
        checkpoint_path,
        tokenizer_path,
    )
    if registration["artifacts"] != current_artifacts:
        raise HarnessInvalid("registered artifact hashes changed")
    if registration["environment"] != environment_record(device):
        raise HarnessInvalid("registered execution environment changed")
    if any(registration["timing_attestation"].values()):
        raise HarnessInvalid("registration timing attestation is invalid")
    manifest = verify_manifest(data_path)
    if stable_sha256(manifest) != registration["manifest_scientific_sha256"]:
        raise HarnessInvalid("manifest scientific hash changed")
    return manifest


def hidden_population_rows(populations: dict[str, Any]) -> list[dict[str, Any]]:
    combined = (
        list(populations["source_train"])
        + list(populations["support_rows"])
        + list(populations["development"])
    )
    by_index: dict[int, dict[str, Any]] = {}
    for row in combined:
        if row["partition"] == "test":
            raise HarnessInvalid("test row reached the hidden population")
        existing = by_index.get(row["source_index"])
        if existing is not None and existing != row:
            raise HarnessInvalid("source index collision in hidden population")
        by_index[row["source_index"]] = row
    return [by_index[index] for index in sorted(by_index)]


def hidden_cache_metadata(
    rows: list[dict[str, Any]],
    populations: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "hlm5-nkt1-development-hidden-cache-v1",
        "checkpoint_sha256": verify_expected_file(
            DEFAULT_CHECKPOINT_PATH, "checkpoint"
        ),
        "tokenizer_sha256": verify_expected_file(DEFAULT_TOKENIZER_PATH, "tokenizer"),
        "row_population_sha256": population_sha256(rows),
        "source_train_sha256": population_sha256(populations["source_train"]),
        "support_sha256": population_sha256(populations["support_rows"]),
        "development_sha256": population_sha256(populations["development"]),
        "source_indices": [row["source_index"] for row in rows],
        "test_utterances_encoded": False,
    }


def load_or_create_hiddens(
    cache_path: Path,
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    populations: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[int, int], dict[str, Any]]:
    rows = hidden_population_rows(populations)
    expected_metadata = hidden_cache_metadata(rows, populations)
    if cache_path.is_file():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache.get("metadata") != expected_metadata:
            raise HarnessInvalid("hidden cache metadata mismatch")
        hiddens = cache.get("hiddens")
        if not isinstance(hiddens, torch.Tensor):
            raise HarnessInvalid("hidden cache tensor is missing")
        hiddens = hiddens.to(device="cpu", dtype=torch.float32).contiguous()
        observed_sha = tensor_sha256(hiddens)
        if observed_sha != cache.get("hidden_sha256"):
            raise HarnessInvalid("hidden cache tensor hash mismatch")
        extraction = cache["extraction"]
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        hiddens, extraction = extract_hiddens(
            model,
            tokenizer,
            rows,
            device,
            HIDDEN_BATCH_SIZE,
        )
        cache = {
            "metadata": expected_metadata,
            "extraction": extraction,
            "hidden_sha256": tensor_sha256(hiddens),
            "hiddens": hiddens,
        }
        torch.save(cache, cache_path)
    if tuple(hiddens.shape) != (len(rows), 768):
        raise HarnessInvalid(f"hidden cache shape changed: {tuple(hiddens.shape)}")
    index = {
        source_index: position
        for position, source_index in enumerate(expected_metadata["source_indices"])
    }
    cache_record = {
        "path": str(cache_path),
        "sha256": sha256_file(cache_path),
        "hidden_sha256": tensor_sha256(hiddens),
        "metadata": expected_metadata,
        "extraction": extraction,
    }
    return hiddens, index, cache_record


def select_hiddens(
    hiddens: torch.Tensor,
    index: dict[int, int],
    rows: list[dict[str, Any]],
) -> torch.Tensor:
    positions = []
    for row in rows:
        if row["source_index"] not in index:
            raise HarnessInvalid("row missing from hidden cache")
        positions.append(index[row["source_index"]])
    return hiddens[torch.tensor(positions, dtype=torch.long)].to(dtype=torch.float64)


def changed_transitions(
    candidate_rows: list[dict[str, Any]],
    comparator_rows: list[dict[str, Any]],
) -> dict[str, int]:
    if len(candidate_rows) != len(comparator_rows):
        raise HarnessInvalid("transition row counts differ")
    counts = {
        "comparator_wrong_candidate_right": 0,
        "comparator_right_candidate_wrong": 0,
        "both_right": 0,
        "both_wrong": 0,
        "prediction_changed": 0,
    }
    for candidate, comparator in zip(candidate_rows, comparator_rows):
        candidate_right = bool(candidate["correct"])
        comparator_right = bool(comparator["correct"])
        if candidate_right and comparator_right:
            counts["both_right"] += 1
        elif candidate_right:
            counts["comparator_wrong_candidate_right"] += 1
        elif comparator_right:
            counts["comparator_right_candidate_wrong"] += 1
        else:
            counts["both_wrong"] += 1
        if candidate["predicted_intent"] != comparator["predicted_intent"]:
            counts["prediction_changed"] += 1
    return counts


def snapshot_reader_parity(
    query_keys: torch.Tensor,
    support_keys: torch.Tensor,
) -> dict[str, Any]:
    snapshot = str(SNAPSHOT_ROOT)
    if snapshot not in sys.path:
        sys.path.insert(0, snapshot)
    from hlm5_memory import EditableHLM5Memory  # type: ignore

    memory = EditableHLM5Memory(
        dim=int(support_keys.shape[1]),
        memory_size=int(support_keys.shape[0]),
        degree=DEGREE,
        learnable_memory=False,
        read_mode="hard_top1",
    )
    with torch.no_grad():
        memory.keys.copy_(support_keys.to(dtype=torch.float32))
        memory.alphas.fill_(1.0)
        memory.active.fill_(True)
        snapshot_scores = memory.score(query_keys.to(dtype=torch.float32))
    registered_scores = torch.relu(query_keys @ support_keys.T).pow(DEGREE)
    snapshot_slots = torch.argmax(snapshot_scores, dim=1)
    registered_slots = torch.argmax(registered_scores, dim=1)
    snapshot_abstentions = torch.max(snapshot_scores, dim=1).values == 0.0
    registered_abstentions = torch.max(registered_scores, dim=1).values == 0.0
    mismatch = (snapshot_slots != registered_slots) | (
        snapshot_abstentions != registered_abstentions
    )
    prediction_match = not bool(mismatch.any())
    return {
        "prediction_match": prediction_match,
        "prediction_or_abstention_mismatch_count": int(torch.sum(mismatch).item()),
        "score_max_abs_error": float(
            torch.max(
                torch.abs(snapshot_scores.to(dtype=torch.float64) - registered_scores)
            ).item()
        ),
        "snapshot_scores_sha256": tensor_sha256(snapshot_scores),
    }


def run_development(args: argparse.Namespace) -> dict[str, Any]:
    configure_determinism()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise HarnessInvalid("development is registered for local CUDA")
    data_path = args.data.resolve()
    license_path = args.license.resolve()
    checkpoint_path = args.checkpoint.resolve()
    tokenizer_path = args.tokenizer.resolve()
    cache_path = args.cache.resolve()
    if checkpoint_path != DEFAULT_CHECKPOINT_PATH.resolve():
        raise HarnessInvalid("checkpoint path is not registered")
    if tokenizer_path != DEFAULT_TOKENIZER_PATH.resolve():
        raise HarnessInvalid("tokenizer path is not registered")
    if cache_path != DEFAULT_CACHE_PATH.resolve():
        raise HarnessInvalid("cache path is not registered")
    if not REGISTRATION_PATH.is_file():
        raise HarnessInvalid("development registration is missing")
    registration = read_json(REGISTRATION_PATH)
    manifest = verify_registration(
        registration,
        data_path,
        license_path,
        checkpoint_path,
        tokenizer_path,
        device,
    )
    populations = build_populations(data_path)
    model, tokenizer, model_metadata = load_model_and_tokenizer(
        checkpoint_path, tokenizer_path, device
    )
    hiddens, hidden_index, cache_record = load_or_create_hiddens(
        cache_path,
        model,
        tokenizer,
        populations,
        device,
    )

    source_rows = populations["source_train"]
    support_rows = populations["support_rows"]
    query_rows = populations["target_development"]
    off_support_rows = populations["source_development"]
    source_values = select_hiddens(hiddens, hidden_index, source_rows)
    support_values = select_hiddens(hiddens, hidden_index, support_rows)
    query_values = select_hiddens(hiddens, hidden_index, query_rows)
    off_support_values = select_hiddens(hiddens, hidden_index, off_support_rows)
    source_labels = [row["intent"] for row in source_rows]
    support_labels = [row["intent"] for row in support_rows]
    query_labels = [row["intent"] for row in query_rows]
    shuffled_labels = populations["shuffled_labels"]

    operators = fit_key_operators(source_values, source_labels, shuffled_labels)
    identity_errors = [
        operators.metadata[arm]["inverse_identity_max_abs_error"]
        for arm in ("blind_zca", "shuffled_within", "source_fisher")
    ]
    if max(identity_errors) > 1e-8:
        raise HarnessInvalid("inverse-root identity check exceeded 1e-8")

    arm_metrics: dict[str, dict[str, Any]] = {}
    arm_rows: dict[str, list[dict[str, Any]]] = {}
    key_hashes: dict[str, dict[str, str]] = {}
    snapshot_parity: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        support_keys = apply_arm(support_values, operators, arm)
        query_keys = apply_arm(query_values, operators, arm)
        off_support_keys = apply_arm(off_support_values, operators, arm)
        metrics, rows = evaluate_arm(
            query_keys,
            support_keys,
            query_labels,
            support_labels,
            off_support_keys,
        )
        arm_metrics[arm] = metrics
        arm_rows[arm] = rows
        snapshot_parity[arm] = snapshot_reader_parity(query_keys, support_keys)
        if not snapshot_parity[arm]["prediction_match"]:
            raise HarnessInvalid(f"checked-in float32 reader changed {arm} predictions")
        key_hashes[arm] = {
            "support": tensor_sha256(support_keys),
            "query": tensor_sha256(query_keys),
            "off_support": tensor_sha256(off_support_keys),
        }

    comparisons = {
        comparator: paired_intent_bootstrap(
            arm_metrics["source_fisher"]["per_intent_accuracy"],
            arm_metrics[comparator]["per_intent_accuracy"],
        )
        for comparator in ("raw", "centered", "blind_zca", "shuffled_within")
    }
    verdict, bars = development_verdict(arm_metrics, comparisons)
    transitions = {
        comparator: changed_transitions(arm_rows["source_fisher"], arm_rows[comparator])
        for comparator in ("raw", "centered", "blind_zca", "shuffled_within")
    }

    row_records: list[dict[str, Any]] = []
    for index, source_row in enumerate(query_rows):
        record = {
            "id": source_row["id"],
            "source_index": source_row["source_index"],
            "scenario": source_row["scenario"],
            "true_intent": source_row["intent"],
            "arms": {arm: arm_rows[arm][index] for arm in ARM_NAMES},
        }
        row_records.append(record)

    validity = {
        "registration_verified": True,
        "manifest_verified": True,
        "test_utterances_not_encoded": True,
        "model_strict_load": True,
        "all_finite": True,
        "source_fit_has_no_target_domain_rows": all(
            row["scenario"] not in set(populations["target_domains"])
            for row in source_rows
        ),
        "support_slots_exact": len(support_rows) == 69,
        "target_development_rows_exact": len(query_rows) == 494,
        "target_development_intents_exact": len(set(query_labels)) == 23,
        "inverse_root_identity": max(identity_errors) <= 1e-8,
        "shuffle_multiset_preserved": sorted(source_labels) == sorted(shuffled_labels),
        "degree5_cosine_and_numpy_match": all(
            metrics["independent_max_abs_error"] <= 1e-10
            for metrics in arm_metrics.values()
        ),
        "snapshot_float32_reader_predictions_match": all(
            record["prediction_match"] for record in snapshot_parity.values()
        ),
        "threshold_diagnostic_only": True,
    }
    if not all(value is True for value in validity.values()):
        raise HarnessInvalid(f"binding validity failure: {validity}")

    scientific = {
        "population": {
            "manifest_scientific_sha256": stable_sha256(manifest),
            "source_train_rows": len(source_rows),
            "source_intents": len(set(source_labels)),
            "support_rows": len(support_rows),
            "support_intents": len(set(support_labels)),
            "target_development_rows": len(query_rows),
            "target_development_intents": len(set(query_labels)),
            "off_support_rows": len(off_support_rows),
            "target_domains": populations["target_domains"],
        },
        "operators": operators.metadata,
        "key_hashes": key_hashes,
        "snapshot_float32_reader_parity": snapshot_parity,
        "arms": arm_metrics,
        "comparisons": comparisons,
        "transitions": transitions,
        "bars": bars,
        "verdict": verdict,
    }
    result = {
        "schema": "hlm5-nkt1-development-result-v1",
        "status": "VALID",
        "verdict": verdict,
        "generated_at_local": now_local(),
        "git_head": git_head(),
        "command": [sys.executable, *sys.argv],
        "environment": environment_record(device),
        "registration": {
            "path": REGISTRATION_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(REGISTRATION_PATH),
            "protocol_sha256": registration["protocol_sha256"],
        },
        "artifacts": registration["artifacts"],
        "model": model_metadata,
        "hidden_cache": cache_record,
        "validity": validity,
        "scientific": scientific,
        "scientific_sha256": stable_sha256(scientific),
        "claim_scope": (
            "Cross-domain three-shot routing in the frozen HLM5-136M final-hidden "
            "key geometry only; cache-equivalent reader, no recurrence, no test, "
            "no scaling or product claim."
        ),
    }
    write_json(RESULT_PATH, result)
    ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ROWS_PATH.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in row_records
        ),
        encoding="utf-8",
        newline="\n",
    )
    return result


def common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--license", type=Path, default=DEFAULT_LICENSE_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--device", default="cuda")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    common_paths(preflight_parser)
    subparsers.add_parser("register")
    development_parser = subparsers.add_parser("development")
    common_paths(development_parser)
    development_parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "preflight":
        result = preflight(args)
        print(f"status={result['status']}")
        print(f"preflight_sha256={sha256_file(PREFLIGHT_PATH)}")
        print("massive_utterances_encoded=false")
    elif args.mode == "register":
        result = register_development()
        print(f"status={result['status']}")
        print(f"registration_sha256={sha256_file(REGISTRATION_PATH)}")
    elif args.mode == "development":
        result = run_development(args)
        print(f"status={result['status']}")
        print(f"verdict={result['verdict']}")
        print(f"scientific_sha256={result['scientific_sha256']}")
        for arm in ARM_NAMES:
            metrics = result["scientific"]["arms"][arm]
            print(
                f"{arm}: macro={metrics['macro_accuracy']:.9f} "
                f"row={metrics['row_accuracy']:.9f} "
                f"auroc={metrics['supported_off_support_auroc']:.9f}"
            )
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()

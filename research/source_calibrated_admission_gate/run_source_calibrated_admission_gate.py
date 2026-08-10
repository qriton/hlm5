"""Preflight, register, and run the frozen SCA-1 development assay."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.natural_key_transfer_gate.run_natural_key_transfer_gate import (
    configure_determinism,
    extract_hiddens,
    load_model_and_tokenizer,
    snapshot_reader_parity,
)

from .contract import (
    ARM_NAMES,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CALIBRATION_NEGATIVE_QUANTILE,
    CALIBRATION_QUANTILE_METHOD,
    CANDIDATE_ARM,
    DEFAULT_CACHE_PATH,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DATA_ROOT,
    DEFAULT_TOKENIZER_PATH,
    DEGREE,
    DEPENDENCY_PATHS,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_RANDOM_BASIS_SHA256,
    GROUP_INTENT_COUNT,
    IMPLEMENTATION_PATHS,
    MANIFEST_PATH,
    PACKAGE_ROOT,
    PASS_BARS,
    PREFLIGHT_PATH,
    PREREG_PATH,
    RANDOM_BASIS_SEED,
    REGISTRATION_PATH,
    REPO_ROOT,
    RESULT_PATH,
    ROWS_PATH,
    ROWS_PER_INTENT,
    SHRINKAGE,
    SOURCE_DOMAINS,
    SOURCE_FIT_PER_DOMAIN,
    SUPPORT_PER_INTENT,
    TARGET_DOMAINS,
    git_head,
    read_json,
    relative_hashes,
    sha256_bytes,
    sha256_file,
    stable_sha256,
    verify_data_files,
    verify_model_file,
    write_json,
)
from .geometry import (
    apply_arm,
    calibration_threshold,
    development_verdict,
    evaluate_arm,
    fit_key_operators,
    paired_intent_bootstrap,
    registered_random_basis,
    tensor_sha256,
)
from .prepare_topv2 import build_populations, population_sha256, verify_manifest


HIDDEN_BATCH_SIZE = 128
MODEL_DETERMINISM_SEED = 20260810
SYNTHETIC_ANCHOR = "SCA one synthetic HLM5 admission anchor."
HIDDEN_POPULATION_NAMES = (
    "source_fit",
    "target_support",
    "target_development",
    "calibration_support",
    "calibration_positive",
    "calibration_negative",
    "off_support_audit",
)


class HarnessInvalid(RuntimeError):
    """Raised when a binding implementation or provenance contract fails."""


def now_local() -> str:
    return dt.datetime.now().astimezone().isoformat()


def scientific_constants() -> dict[str, Any]:
    return {
        "schema": "hlm5-sca1-scientific-constants-v1",
        "arms": list(ARM_NAMES),
        "candidate": CANDIDATE_ARM,
        "degree": DEGREE,
        "shrinkage": SHRINKAGE,
        "source_domains": list(SOURCE_DOMAINS),
        "target_domains": list(TARGET_DOMAINS),
        "support_per_intent": SUPPORT_PER_INTENT,
        "rows_per_intent": ROWS_PER_INTENT,
        "source_fit_per_domain": SOURCE_FIT_PER_DOMAIN,
        "group_intent_count": GROUP_INTENT_COUNT,
        "hidden_population_names": list(HIDDEN_POPULATION_NAMES),
        "hidden_batch_size": HIDDEN_BATCH_SIZE,
        "hidden_dtype": "torch.float32",
        "model_determinism_seed": MODEL_DETERMINISM_SEED,
        "scientific_device": "cpu",
        "scientific_dtype": "torch.float64",
        "random_basis_seed": RANDOM_BASIS_SEED,
        "random_basis_sha256": EXPECTED_RANDOM_BASIS_SHA256,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "calibration_negative_quantile": CALIBRATION_NEGATIVE_QUANTILE,
        "calibration_quantile_method": CALIBRATION_QUANTILE_METHOD,
        "admission_comparison": "strict_greater_than",
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
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda": torch.version.cuda,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_tf32": (
            torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None
        ),
        "cudnn_tf32": (
            torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else None
        ),
    }


def registered_execution_contract() -> dict[str, Any]:
    return {
        "mode": "development",
        "data_root": str(DEFAULT_DATA_ROOT.resolve()),
        "checkpoint": str(DEFAULT_CHECKPOINT_PATH.resolve()),
        "tokenizer": str(DEFAULT_TOKENIZER_PATH.resolve()),
        "cache": str(DEFAULT_CACHE_PATH.resolve()),
        "device": "cuda",
    }


def _require_registered_paths(
    data_root: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
    cache_path: Path | None = None,
) -> None:
    pairs = (
        ("data root", data_root, DEFAULT_DATA_ROOT),
        ("checkpoint", checkpoint_path, DEFAULT_CHECKPOINT_PATH),
        ("tokenizer", tokenizer_path, DEFAULT_TOKENIZER_PATH),
    )
    for name, observed, expected in pairs:
        if observed.resolve() != expected.resolve():
            raise HarnessInvalid(f"{name} path is not registered")
    if cache_path is not None and cache_path.resolve() != DEFAULT_CACHE_PATH.resolve():
        raise HarnessInvalid("hidden-cache path is not registered")


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
        "summary": (
            completed.stderr.strip().splitlines()[-1]
            if completed.stderr.strip()
            else "passed"
        ),
    }


def _verified_manifest(data_root: Path) -> dict[str, Any]:
    if sha256_file(MANIFEST_PATH) != EXPECTED_MANIFEST_SHA256["file"]:
        raise HarnessInvalid("registered TOPv2 manifest file hash changed")
    manifest = verify_manifest(data_root)
    if manifest.get("scientific_sha256") != EXPECTED_MANIFEST_SHA256["scientific"]:
        raise HarnessInvalid("registered TOPv2 manifest scientific hash changed")
    return manifest


def artifact_record(
    data_root: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
) -> dict[str, Any]:
    return {
        "data": {
            "root": str(data_root),
            "files": verify_data_files(data_root),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": verify_model_file(checkpoint_path, "checkpoint"),
        },
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": verify_model_file(tokenizer_path, "tokenizer"),
        },
        "preregistration": {
            "path": PREREG_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(PREREG_PATH),
        },
        "manifest": {
            "path": MANIFEST_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(MANIFEST_PATH),
            "scientific_sha256": EXPECTED_MANIFEST_SHA256["scientific"],
        },
        "implementation": relative_hashes(IMPLEMENTATION_PATHS),
        "dependencies": relative_hashes(DEPENDENCY_PATHS),
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    configure_determinism()
    if args.device != "cuda":
        raise HarnessInvalid("preflight device must be exactly 'cuda'")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise HarnessInvalid("preflight is registered for local CUDA")
    data_root = args.data_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    tokenizer_path = args.tokenizer.resolve()
    _require_registered_paths(data_root, checkpoint_path, tokenizer_path)
    if any(path.exists() for path in (REGISTRATION_PATH, RESULT_PATH, ROWS_PATH)):
        raise HarnessInvalid("stale registration or result artifact exists")
    if DEFAULT_CACHE_PATH.exists():
        raise HarnessInvalid("registered development cache already exists")
    manifest = _verified_manifest(data_root)
    tests = run_unit_tests()
    _, random_basis = registered_random_basis(768)
    if random_basis["basis_sha256"] != EXPECTED_RANDOM_BASIS_SHA256:
        raise HarnessInvalid("registered random-basis hash changed")
    model, tokenizer, model_metadata = load_model_and_tokenizer(
        checkpoint_path, tokenizer_path, device
    )
    synthetic_rows = [{"utterance": SYNTHETIC_ANCHOR}]
    first, first_meta = extract_hiddens(model, tokenizer, synthetic_rows, device, 1)
    second, second_meta = extract_hiddens(model, tokenizer, synthetic_rows, device, 1)
    if not torch.equal(first, second):
        raise HarnessInvalid("synthetic anchor is not bit-repeatable")
    record = {
        "schema": "hlm5-sca1-preflight-v1",
        "status": "READY_TO_REGISTER",
        "generated_at_local": now_local(),
        "git_head": git_head(),
        "command": [sys.executable, *sys.argv],
        "environment": environment_record(device),
        "artifacts": artifact_record(data_root, checkpoint_path, tokenizer_path),
        "manifest_scientific_sha256": manifest["scientific_sha256"],
        "scientific_constants": scientific_constants(),
        "execution_contract": registered_execution_contract(),
        "execution_contract_sha256": stable_sha256(registered_execution_contract()),
        "random_basis": random_basis,
        "tests": tests,
        "model": model_metadata,
        "synthetic_anchor": {
            "text_sha256": stable_sha256(SYNTHETIC_ANCHOR),
            "hidden_sha256": tensor_sha256(first),
            "repeat_bit_exact": True,
            "first": first_meta,
            "second": second_meta,
        },
        "attestation": {
            "topv2_utterances_encoded": False,
            "routing_outcomes_computed": False,
            "test_utterances_encoded": False,
            "development_cache_existed": False,
            "only_trunk_input": "fixed synthetic anchor",
        },
    }
    write_json(PREFLIGHT_PATH, record)
    return record


def verify_preflight(record: dict[str, Any]) -> None:
    if record.get("schema") != "hlm5-sca1-preflight-v1":
        raise HarnessInvalid("preflight schema mismatch")
    if record.get("status") != "READY_TO_REGISTER":
        raise HarnessInvalid("preflight is not ready")
    if record.get("git_head") != git_head():
        raise HarnessInvalid("preflight Git head is stale")
    if record.get("scientific_constants") != scientific_constants():
        raise HarnessInvalid("scientific constants changed after preflight")
    if record.get("execution_contract") != registered_execution_contract():
        raise HarnessInvalid("execution contract changed after preflight")
    if record.get("execution_contract_sha256") != stable_sha256(
        registered_execution_contract()
    ):
        raise HarnessInvalid("execution contract hash changed after preflight")
    current_artifacts = artifact_record(
        DEFAULT_DATA_ROOT.resolve(),
        DEFAULT_CHECKPOINT_PATH.resolve(),
        DEFAULT_TOKENIZER_PATH.resolve(),
    )
    if record.get("artifacts") != current_artifacts:
        raise HarnessInvalid("artifacts changed after preflight")
    manifest = _verified_manifest(DEFAULT_DATA_ROOT.resolve())
    if record.get("manifest_scientific_sha256") != manifest["scientific_sha256"]:
        raise HarnessInvalid("manifest changed after preflight")
    if not record.get("synthetic_anchor", {}).get("repeat_bit_exact"):
        raise HarnessInvalid("synthetic repeatability did not pass")
    _, random_basis = registered_random_basis(768)
    if record.get("random_basis") != random_basis:
        raise HarnessInvalid("registered random basis changed after preflight")
    if any(
        record.get("attestation", {}).get(key)
        for key in (
            "topv2_utterances_encoded",
            "routing_outcomes_computed",
            "test_utterances_encoded",
            "development_cache_existed",
        )
    ):
        raise HarnessInvalid("preflight attestation reports outcome access")


def register_development() -> dict[str, Any]:
    if not PREFLIGHT_PATH.is_file():
        raise HarnessInvalid("preflight file is missing")
    preflight_record = read_json(PREFLIGHT_PATH)
    verify_preflight(preflight_record)
    if any(path.exists() for path in (RESULT_PATH, ROWS_PATH, DEFAULT_CACHE_PATH)):
        raise HarnessInvalid("outcome or cache artifact exists before registration")
    protocol = scientific_constants()
    record = {
        "schema": "hlm5-sca1-development-registration-v1",
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
        "execution_contract": registered_execution_contract(),
        "execution_contract_sha256": stable_sha256(registered_execution_contract()),
        "random_basis": preflight_record["random_basis"],
        "timing_attestation": {
            "topv2_utterances_encoded_before_registration": False,
            "routing_outcomes_computed_before_registration": False,
            "test_utterances_encoded": False,
            "development_cache_existed": False,
        },
    }
    write_json(REGISTRATION_PATH, record)
    return record


def verify_registration(
    registration: dict[str, Any],
    data_root: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
    cache_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    _require_registered_paths(
        data_root, checkpoint_path, tokenizer_path, cache_path=cache_path
    )
    if registration.get("schema") != "hlm5-sca1-development-registration-v1":
        raise HarnessInvalid("registration schema mismatch")
    if registration.get("status") != "REGISTERED_BEFORE_OUTCOMES":
        raise HarnessInvalid("registration status mismatch")
    if registration.get("git_head") != git_head():
        raise HarnessInvalid("registration Git head is stale")
    if registration.get("scientific_constants") != scientific_constants():
        raise HarnessInvalid("registered scientific constants changed")
    if registration.get("protocol_sha256") != stable_sha256(scientific_constants()):
        raise HarnessInvalid("registered protocol hash changed")
    if registration.get("execution_contract") != registered_execution_contract():
        raise HarnessInvalid("registered execution contract changed")
    if registration.get("execution_contract_sha256") != stable_sha256(
        registered_execution_contract()
    ):
        raise HarnessInvalid("registered execution contract hash changed")
    _, random_basis = registered_random_basis(768)
    if registration.get("random_basis") != random_basis:
        raise HarnessInvalid("registered random basis changed")
    if registration.get("preflight", {}).get("sha256") != sha256_file(PREFLIGHT_PATH):
        raise HarnessInvalid("registered preflight bytes changed")
    preflight_record = read_json(PREFLIGHT_PATH)
    verify_preflight(preflight_record)
    current_artifacts = artifact_record(data_root, checkpoint_path, tokenizer_path)
    if registration.get("artifacts") != current_artifacts:
        raise HarnessInvalid("registered artifact hashes changed")
    if registration.get("environment") != environment_record(device):
        raise HarnessInvalid("registered execution environment changed")
    if any(registration.get("timing_attestation", {}).values()):
        raise HarnessInvalid("registration timing attestation is invalid")
    manifest = _verified_manifest(data_root)
    if manifest["scientific_sha256"] != registration.get("manifest_scientific_sha256"):
        raise HarnessInvalid("registered manifest scientific hash changed")
    return manifest


def hidden_population_rows(
    populations: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for name in HIDDEN_POPULATION_NAMES:
        for row in populations[name]:
            if row["partition"] == "test":
                raise HarnessInvalid("test row reached the hidden population")
            existing = by_id.get(row["row_id"])
            if existing is not None and existing != row:
                raise HarnessInvalid("row-ID collision in hidden population")
            by_id[row["row_id"]] = row
    return [by_id[row_id] for row_id in sorted(by_id)]


def hidden_cache_metadata(
    rows: list[dict[str, Any]], populations: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    return {
        "schema": "hlm5-sca1-development-hidden-cache-v1",
        "checkpoint_sha256": verify_model_file(DEFAULT_CHECKPOINT_PATH, "checkpoint"),
        "tokenizer_sha256": verify_model_file(DEFAULT_TOKENIZER_PATH, "tokenizer"),
        "row_population_sha256": population_sha256(rows),
        "population_sha256": {
            name: population_sha256(populations[name])
            for name in HIDDEN_POPULATION_NAMES
        },
        "row_ids": [row["row_id"] for row in rows],
        "test_utterances_encoded": False,
    }


def load_or_create_hiddens(
    cache_path: Path,
    model: torch.nn.Module,
    tokenizer: Any,
    populations: dict[str, list[dict[str, Any]]],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, int], dict[str, Any]]:
    rows = hidden_population_rows(populations)
    expected_metadata = hidden_cache_metadata(rows, populations)
    created = not cache_path.is_file()
    if created:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        hiddens, extraction = extract_hiddens(
            model, tokenizer, rows, device, HIDDEN_BATCH_SIZE
        )
        cache = {
            "metadata": expected_metadata,
            "extraction": extraction,
            "hidden_sha256": tensor_sha256(hiddens),
            "hiddens": hiddens,
        }
        torch.save(cache, cache_path)
    else:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache.get("metadata") != expected_metadata:
            raise HarnessInvalid("hidden-cache metadata mismatch")
        hiddens = cache.get("hiddens")
        if not isinstance(hiddens, torch.Tensor):
            raise HarnessInvalid("hidden-cache tensor is missing")
        hiddens = hiddens.to(device="cpu", dtype=torch.float32).contiguous()
        extraction = cache.get("extraction")
        if not isinstance(extraction, dict):
            raise HarnessInvalid("hidden-cache extraction record is missing")
    if tuple(hiddens.shape) != (len(rows), 768):
        raise HarnessInvalid(f"hidden-cache shape changed: {tuple(hiddens.shape)}")
    if not bool(torch.isfinite(hiddens).all()):
        raise HarnessInvalid("hidden cache contains non-finite values")
    if tensor_sha256(hiddens) != cache.get("hidden_sha256"):
        raise HarnessInvalid("hidden-cache tensor hash mismatch")
    index = {
        row_id: position for position, row_id in enumerate(expected_metadata["row_ids"])
    }
    record = {
        "path": str(cache_path),
        "sha256": sha256_file(cache_path),
        "created_this_invocation": created,
        "hidden_sha256": tensor_sha256(hiddens),
        "metadata": expected_metadata,
        "extraction": extraction,
    }
    return hiddens, index, record


def select_hiddens(
    hiddens: torch.Tensor,
    index: dict[str, int],
    rows: list[dict[str, Any]],
) -> torch.Tensor:
    positions: list[int] = []
    for row in rows:
        if row["row_id"] not in index:
            raise HarnessInvalid("population row is missing from hidden cache")
        positions.append(index[row["row_id"]])
    selected = hiddens[torch.tensor(positions, dtype=torch.long)].to(
        device="cpu", dtype=torch.float64
    )
    if not bool(torch.isfinite(selected).all()):
        raise HarnessInvalid("selected hidden state is non-finite")
    return selected


def _population_validity(
    populations: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, bool]:
    source_fit = populations["source_fit"]
    source_counts = collections.Counter(row["domain"] for row in source_fit)
    target_labels = [row["intent"] for row in populations["target_development"]]
    target_support_labels = [row["intent"] for row in populations["target_support"]]
    calibration_support_labels = [
        row["intent"] for row in populations["calibration_support"]
    ]
    groups = metadata["source_groups"]
    group_sets = [
        set(groups[name])
        for name in (
            "calibration_supported",
            "calibration_negative",
            "off_support_audit",
        )
    ]
    hidden_ids = {
        row["row_id"] for name in HIDDEN_POPULATION_NAMES for row in populations[name]
    }
    test_ids = {row["row_id"] for row in populations["test_target"]}
    return {
        "manifest_population_hashes_match": all(
            manifest["populations"][name]["sha256"]
            == population_sha256(populations[name])
            and manifest["populations"][name]["count"] == len(populations[name])
            for name in populations
        ),
        "hidden_population_names_exact": manifest["development_hidden_population_names"]
        == list(HIDDEN_POPULATION_NAMES),
        "source_fit_rows_exact": len(source_fit)
        == len(SOURCE_DOMAINS) * SOURCE_FIT_PER_DOMAIN,
        "source_fit_domains_balanced": all(
            source_counts[domain] == SOURCE_FIT_PER_DOMAIN for domain in SOURCE_DOMAINS
        ),
        "source_fit_train_only": all(row["partition"] == "train" for row in source_fit),
        "source_fit_excludes_target_domains": all(
            row["domain"] not in TARGET_DOMAINS for row in source_fit
        ),
        "target_support_slots_exact": len(populations["target_support"])
        == GROUP_INTENT_COUNT * SUPPORT_PER_INTENT,
        "calibration_support_slots_exact": len(populations["calibration_support"])
        == GROUP_INTENT_COUNT * SUPPORT_PER_INTENT,
        "target_development_rows_exact": len(populations["target_development"])
        == GROUP_INTENT_COUNT * ROWS_PER_INTENT,
        "calibration_positive_rows_exact": len(populations["calibration_positive"])
        == GROUP_INTENT_COUNT * ROWS_PER_INTENT,
        "calibration_negative_rows_exact": len(populations["calibration_negative"])
        == GROUP_INTENT_COUNT * ROWS_PER_INTENT,
        "off_support_audit_rows_exact": len(populations["off_support_audit"])
        == GROUP_INTENT_COUNT * ROWS_PER_INTENT,
        "target_intents_exact": len(set(target_labels)) == GROUP_INTENT_COUNT,
        "three_target_supports_per_intent": all(
            count == SUPPORT_PER_INTENT
            for count in collections.Counter(target_support_labels).values()
        )
        and len(set(target_support_labels)) == GROUP_INTENT_COUNT,
        "three_calibration_supports_per_intent": all(
            count == SUPPORT_PER_INTENT
            for count in collections.Counter(calibration_support_labels).values()
        )
        and len(set(calibration_support_labels)) == GROUP_INTENT_COUNT,
        "twenty_target_rows_per_intent": all(
            count == ROWS_PER_INTENT
            for count in collections.Counter(target_labels).values()
        ),
        "source_calibration_groups_disjoint": all(
            group_sets[left].isdisjoint(group_sets[right])
            for left in range(len(group_sets))
            for right in range(left + 1, len(group_sets))
        ),
        "test_rows_not_in_hidden_population": not bool(hidden_ids & test_ids),
        "hidden_partitions_exclude_test": all(
            row["partition"] != "test"
            for name in HIDDEN_POPULATION_NAMES
            for row in populations[name]
        ),
    }


def _transition_counts(
    candidate: dict[str, Any], comparator: dict[str, Any]
) -> dict[str, int]:
    if len(candidate["predicted"]) != len(comparator["predicted"]):
        raise HarnessInvalid("transition row counts differ")
    result = {
        "comparator_wrong_candidate_right_admitted": 0,
        "comparator_right_admitted_candidate_wrong": 0,
        "both_correct_admitted": 0,
        "neither_correct_admitted": 0,
        "prediction_changed": 0,
        "admission_changed": 0,
    }
    for index in range(len(candidate["predicted"])):
        candidate_right = bool(candidate["correct_admitted"][index])
        comparator_right = bool(comparator["correct_admitted"][index])
        if candidate_right and comparator_right:
            result["both_correct_admitted"] += 1
        elif candidate_right:
            result["comparator_wrong_candidate_right_admitted"] += 1
        elif comparator_right:
            result["comparator_right_admitted_candidate_wrong"] += 1
        else:
            result["neither_correct_admitted"] += 1
        if candidate["predicted"][index] != comparator["predicted"][index]:
            result["prediction_changed"] += 1
        if bool(candidate["admitted"][index]) != bool(comparator["admitted"][index]):
            result["admission_changed"] += 1
    return result


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _row_records(
    target_rows: list[dict[str, Any]],
    arm_metrics: dict[str, dict[str, Any]],
    arm_arrays: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(target_rows):
        records.append(
            {
                "row_id": row["row_id"],
                "utterance_sha256": row["utterance_sha256"],
                "domain": row["domain"],
                "partition": row["partition"],
                "source_line": row["source_line"],
                "true_intent": row["intent"],
                "arms": {
                    arm: {
                        "predicted_intent": arm_arrays[arm]["target"]["predicted"][
                            index
                        ],
                        "maximum_score": float(
                            arm_arrays[arm]["target"]["maximum_scores"][index].item()
                        ),
                        "maximum_cosine": float(
                            arm_arrays[arm]["target"]["maximum_cosines"][index].item()
                        ),
                        "threshold": arm_metrics[arm]["threshold"],
                        "admitted": bool(
                            arm_arrays[arm]["target"]["admitted"][index].item()
                        ),
                        "correct": bool(
                            arm_arrays[arm]["target"]["correct"][index].item()
                        ),
                        "correct_admitted": bool(
                            arm_arrays[arm]["target"]["correct_admitted"][index].item()
                        ),
                    }
                    for arm in ARM_NAMES
                },
            }
        )
    return records


def _compute_development(
    hiddens: torch.Tensor,
    hidden_index: dict[str, int],
    populations: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bool]]:
    values = {
        name: select_hiddens(hiddens, hidden_index, populations[name])
        for name in HIDDEN_POPULATION_NAMES
    }
    operators = fit_key_operators(values["source_fit"])
    labels = {
        name: [row["intent"] for row in populations[name]]
        for name in HIDDEN_POPULATION_NAMES
    }

    arm_metrics: dict[str, dict[str, Any]] = {}
    arm_arrays: dict[str, dict[str, Any]] = {}
    key_hashes: dict[str, dict[str, str]] = {}
    reader_parity: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ARM_NAMES:
        metrics, arrays = evaluate_arm(
            operators,
            arm,
            values["calibration_support"],
            labels["calibration_support"],
            values["calibration_positive"],
            labels["calibration_positive"],
            values["calibration_negative"],
            values["target_support"],
            labels["target_support"],
            values["target_development"],
            labels["target_development"],
            values["off_support_audit"],
        )
        arm_metrics[arm] = metrics
        arm_arrays[arm] = arrays

        transformed = {
            name: apply_arm(values[name], operators, arm)
            for name in (
                "calibration_support",
                "calibration_positive",
                "calibration_negative",
                "target_support",
                "target_development",
                "off_support_audit",
            )
        }
        key_hashes[arm] = {
            name: tensor_sha256(matrix) for name, matrix in transformed.items()
        }
        reader_parity[arm] = {
            "calibration_positive": snapshot_reader_parity(
                transformed["calibration_positive"],
                transformed["calibration_support"],
            ),
            "calibration_negative": snapshot_reader_parity(
                transformed["calibration_negative"],
                transformed["calibration_support"],
            ),
            "target_development": snapshot_reader_parity(
                transformed["target_development"], transformed["target_support"]
            ),
            "off_support_audit": snapshot_reader_parity(
                transformed["off_support_audit"], transformed["target_support"]
            ),
        }

    comparisons = {
        comparator: paired_intent_bootstrap(
            arm_metrics[CANDIDATE_ARM], arm_metrics[comparator]
        )
        for comparator in ("raw", "centered", "diagonal_std", "random_basis_zca")
    }
    verdict, bars = development_verdict(arm_metrics, comparisons)
    transitions = {
        comparator: _transition_counts(
            arm_arrays[CANDIDATE_ARM]["target"], arm_arrays[comparator]["target"]
        )
        for comparator in ("raw", "centered", "diagonal_std", "random_basis_zca")
    }
    population_validity = _population_validity(populations, metadata, manifest)
    validity = {
        **population_validity,
        "model_hidden_dimension_exact": hiddens.shape[1] == 768,
        "all_hiddens_finite": bool(torch.isfinite(hiddens).all()),
        "blind_inverse_root_identity": operators.metadata["blind_zca"][
            "inverse_identity_max_abs_error"
        ]
        <= 1e-10,
        "matched_spectrum_absolute": operators.metadata[
            "matched_spectrum_max_abs_error"
        ]
        <= 1e-10,
        "matched_spectrum_relative": operators.metadata[
            "matched_spectrum_max_rel_error"
        ]
        <= 1e-10,
        "blind_and_random_bases_differ": operators.metadata["basis_hashes_differ"],
        "random_basis_orthogonal": operators.metadata[
            "random_basis_orthogonality_max_abs_error"
        ]
        <= 1e-10,
        "calibration_negative_rate_controlled": all(
            metrics["calibration_negative"]["false_admission_rate"] <= 0.05
            for metrics in arm_metrics.values()
        ),
        "threshold_recomputed_exactly": all(
            metrics["threshold"]
            == calibration_threshold(
                arm_arrays[arm]["calibration_negative"]["maximum_scores"]
            )
            for arm, metrics in arm_metrics.items()
        ),
        "strict_admission_recomputed_exactly": all(
            torch.equal(
                arrays[population]["admitted"],
                arrays[population]["maximum_scores"] > metrics["threshold"],
            )
            for arm, metrics in arm_metrics.items()
            for arrays in (arm_arrays[arm],)
            for population in (
                "calibration_negative",
                "calibration_positive",
                "target",
                "off_support_audit",
            )
        ),
        "degree5_cosine_and_numpy_match": all(
            section["independent_max_abs_error"] <= 1e-10
            for metrics in arm_metrics.values()
            for section in (
                metrics["calibration_negative"],
                metrics["calibration_positive"],
                metrics["target"],
                metrics["off_support_audit"],
            )
        ),
        "snapshot_float32_reader_predictions_match": all(
            record["prediction_match"]
            for arm_record in reader_parity.values()
            for record in arm_record.values()
        ),
        "test_utterances_not_encoded": True,
    }
    if not all(value is True for value in validity.values()):
        raise HarnessInvalid(f"binding validity failure: {validity}")

    scientific = {
        "population": {
            "manifest_scientific_sha256": manifest["scientific_sha256"],
            "counts": {name: len(rows) for name, rows in populations.items()},
            "hashes": {
                name: population_sha256(rows) for name, rows in populations.items()
            },
            "target_intents": metadata["target_intents"],
            "source_groups": metadata["source_groups"],
            "test_utterances_tokenized_or_encoded": False,
        },
        "operators": operators.metadata,
        "key_hashes": key_hashes,
        "snapshot_float32_reader_parity": reader_parity,
        "arms": arm_metrics,
        "comparisons": comparisons,
        "transitions": transitions,
        "bars": bars,
        "verdict": verdict,
    }
    if not _all_finite(scientific):
        raise HarnessInvalid("scientific object contains non-finite values")
    rows = _row_records(populations["target_development"], arm_metrics, arm_arrays)
    return scientific, rows, validity


def _rows_text(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    )


def run_development(args: argparse.Namespace) -> dict[str, Any]:
    configure_determinism()
    if args.device != "cuda":
        raise HarnessInvalid("development device must be exactly 'cuda'")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise HarnessInvalid("development is registered for local CUDA")
    data_root = args.data_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    tokenizer_path = args.tokenizer.resolve()
    cache_path = args.cache.resolve()
    _require_registered_paths(
        data_root, checkpoint_path, tokenizer_path, cache_path=cache_path
    )
    if not REGISTRATION_PATH.is_file():
        raise HarnessInvalid("development registration is missing")
    registration = read_json(REGISTRATION_PATH)
    manifest = verify_registration(
        registration,
        data_root,
        checkpoint_path,
        tokenizer_path,
        cache_path,
        device,
    )
    if cache_path.exists():
        if not RESULT_PATH.is_file():
            raise HarnessInvalid(
                "unprovenanced hidden cache exists before first result"
            )
        previous_result = read_json(RESULT_PATH)
        if (
            previous_result.get("schema") != "hlm5-sca1-development-result-v1"
            or previous_result.get("git_head") != git_head()
            or previous_result.get("registration", {}).get("sha256")
            != sha256_file(REGISTRATION_PATH)
            or previous_result.get("hidden_cache", {}).get("sha256")
            != sha256_file(cache_path)
        ):
            raise HarnessInvalid(
                "existing hidden cache is not pinned by the prior result"
            )
    built = build_populations(data_root)
    populations = built["rows"]
    model, tokenizer, model_metadata = load_model_and_tokenizer(
        checkpoint_path, tokenizer_path, device
    )
    hiddens, hidden_index, cache_record = load_or_create_hiddens(
        cache_path, model, tokenizer, populations, device
    )
    scientific, row_records, validity = _compute_development(
        hiddens, hidden_index, populations, built["metadata"], manifest
    )
    if (
        scientific["operators"]["random_basis_sha256"]
        != registration["random_basis"]["basis_sha256"]
    ):
        raise HarnessInvalid("development random basis differs from registration")
    validity["registered_random_basis_exact"] = True

    replay_hiddens, replay_index, replay_cache_record = load_or_create_hiddens(
        cache_path, model, tokenizer, populations, device
    )
    replay_scientific, replay_rows, replay_validity = _compute_development(
        replay_hiddens,
        replay_index,
        populations,
        built["metadata"],
        manifest,
    )
    replay_validity["registered_random_basis_exact"] = (
        replay_scientific["operators"]["random_basis_sha256"]
        == registration["random_basis"]["basis_sha256"]
    )
    replay_exact = (
        stable_sha256(scientific) == stable_sha256(replay_scientific)
        and _rows_text(row_records) == _rows_text(replay_rows)
        and validity == replay_validity
        and cache_record["hidden_sha256"] == replay_cache_record["hidden_sha256"]
    )
    if not replay_exact:
        raise HarnessInvalid("hidden-cache replay changed science or rows")
    validity = {
        "registration_verified": True,
        "manifest_verified": True,
        "model_strict_load_and_eval": not model.training,
        **validity,
        "hidden_cache_replay_exact": True,
    }

    result = {
        "schema": "hlm5-sca1-development-result-v1",
        "status": "VALID",
        "verdict": scientific["verdict"],
        "generated_at_local": now_local(),
        "git_head": git_head(),
        "command": [sys.executable, *sys.argv],
        "environment": environment_record(device),
        "registration": {
            "path": REGISTRATION_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(REGISTRATION_PATH),
            "protocol_sha256": registration["protocol_sha256"],
            "execution_contract_sha256": registration["execution_contract_sha256"],
        },
        "artifacts": registration["artifacts"],
        "model": model_metadata,
        "hidden_cache": cache_record,
        "validity": validity,
        "scientific": scientific,
        "scientific_sha256": stable_sha256(scientific),
        "rows_object_sha256": stable_sha256(row_records),
        "rows_jsonl_sha256": sha256_bytes(_rows_text(row_records).encode("utf-8")),
        "claim_scope": (
            "Balanced three-shot TOPv2 target-domain routing in frozen HLM5-136M "
            "final-hidden geometry only, with source-only calibration and a "
            "cache-equivalent static reader; no recurrence, test, scaling, or "
            "product claim."
        ),
    }
    if not all(value is True for value in validity.values()):
        raise HarnessInvalid(f"binding validity failure: {validity}")
    ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ROWS_PATH.write_text(_rows_text(row_records), encoding="utf-8", newline="\n")
    if sha256_file(ROWS_PATH) != result["rows_jsonl_sha256"]:
        raise HarnessInvalid("written row artifact hash mismatch")
    write_json(RESULT_PATH, result)
    return result


def common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
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
        print("topv2_utterances_encoded=false")
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
                f"{arm}: correct_admission="
                f"{metrics['target']['macro_correct_admission_rate']:.9f} "
                f"coverage={metrics['target']['macro_coverage']:.9f} "
                f"selective_accuracy={metrics['target']['selective_accuracy']:.9f} "
                f"off_support_far="
                f"{metrics['off_support_audit']['false_admission_rate']:.9f}"
            )
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("status=INCONCLUSIVE", file=sys.stderr)
        print("reason=interrupted before a valid scientific result", file=sys.stderr)
        raise
    except Exception as error:
        print("status=HARNESS_INVALID", file=sys.stderr)
        print(f"reason={type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2) from error

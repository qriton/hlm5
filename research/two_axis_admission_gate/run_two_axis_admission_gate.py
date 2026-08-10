"""Preflight, register, run, and replay the frozen SCA-2 development assay."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from research.natural_key_transfer_gate.run_natural_key_transfer_gate import (
    configure_determinism,
    extract_hiddens,
    load_model_and_tokenizer,
    snapshot_reader_parity,
)

from .contract import (
    ABSOLUTE_NEGATIVE_QUANTILE,
    ARM_NAMES,
    AUDIT_BARS,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CANDIDATE_ARM,
    DATASET_GIT_COMMIT,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DATA_ROOT,
    DEFAULT_SOURCE_CACHE_PATH,
    DEFAULT_TARGET_CACHE_PATH,
    DEFAULT_TOKENIZER_PATH,
    DEGREE,
    DEPENDENCY_PATHS,
    DEVELOPMENT_FOLD,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PREREG_SHA256,
    EXPECTED_RANDOM_BASIS_SHA256,
    GROUP_INTENT_COUNT,
    GROUP_NAMES,
    IMPLEMENTATION_PATHS,
    MANIFEST_PATH,
    MARGIN_ERROR_QUANTILE,
    MINIMUM_MARGIN_CALIBRATION_ERRORS,
    PACKAGE_ROOT,
    PASS_BARS,
    PREFLIGHT_PATH,
    PREREG_COMMIT,
    PREREG_PATH,
    QUANTILE_METHOD,
    RANDOM_BASIS_SEED,
    REGISTRATION_PATH,
    REPLAY_PATH,
    REPO_ROOT,
    RESULT_PATH,
    ROWS_PATH,
    ROWS_PER_INTENT,
    SHRINKAGE,
    SUPPORT_PER_INTENT,
    TEST_FOLD,
    TRAIN_FOLDS,
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
    all_finite,
    apply_arm,
    calibrate_arm,
    development_verdict,
    evaluate_stage,
    fit_key_operators,
    paired_intent_selective_bootstrap,
    registered_random_basis,
    source_audit_verdict,
    tensor_sha256,
)
from .prepare_hwu64 import build_populations, population_sha256, verify_manifest


HIDDEN_BATCH_SIZE = 128
MODEL_DETERMINISM_SEED = 20260810
SYNTHETIC_ANCHOR = "SCA two synthetic HLM5 admission anchor."
SOURCE_HIDDEN_POPULATION_NAMES = (
    "source_fit",
    "calibration_support",
    "calibration_positive",
    "calibration_negative",
    "audit_support",
    "audit_positive",
    "audit_negative",
)
TARGET_HIDDEN_POPULATION_NAMES = (
    "target_support",
    "target_development",
)


class HarnessInvalid(RuntimeError):
    """Raised when a binding implementation or provenance contract fails."""


def now_local() -> str:
    return dt.datetime.now().astimezone().isoformat()


def prereg_commit_is_ancestor() -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PREREG_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    return completed.returncode == 0


def git_worktree_is_clean() -> bool:
    output = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
    )
    return not output.strip()


def scientific_constants() -> dict[str, Any]:
    return {
        "schema": "hlm5-sca2-scientific-constants-v1",
        "arms": list(ARM_NAMES),
        "candidate": CANDIDATE_ARM,
        "degree": DEGREE,
        "shrinkage": SHRINKAGE,
        "development_fold": DEVELOPMENT_FOLD,
        "test_fold": TEST_FOLD,
        "train_folds": list(TRAIN_FOLDS),
        "support_per_intent": SUPPORT_PER_INTENT,
        "rows_per_intent": ROWS_PER_INTENT,
        "group_intent_count": GROUP_INTENT_COUNT,
        "group_names": list(GROUP_NAMES),
        "source_hidden_population_names": list(SOURCE_HIDDEN_POPULATION_NAMES),
        "target_hidden_population_names": list(TARGET_HIDDEN_POPULATION_NAMES),
        "hidden_batch_size": HIDDEN_BATCH_SIZE,
        "hidden_dtype": "torch.float32",
        "model_determinism_seed": MODEL_DETERMINISM_SEED,
        "scientific_device": "cpu",
        "scientific_dtype": "torch.float64",
        "random_basis_seed": RANDOM_BASIS_SEED,
        "random_basis_sha256": EXPECTED_RANDOM_BASIS_SHA256,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "absolute_negative_quantile": ABSOLUTE_NEGATIVE_QUANTILE,
        "margin_error_quantile": MARGIN_ERROR_QUANTILE,
        "quantile_method": QUANTILE_METHOD,
        "minimum_margin_calibration_errors": MINIMUM_MARGIN_CALIBRATION_ERRORS,
        "admission_comparison": "strict_greater_than",
        "class_score_reducer": "maximum_over_three_support_slots",
        "matched_proximity_tie_break": "row_id_ascending",
        "audit_bars": AUDIT_BARS,
        "pass_bars": PASS_BARS,
        "source_audit_precedes_target_encoding": True,
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
        "tokenizers": importlib.metadata.version("tokenizers"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
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
        "source_cache": str(DEFAULT_SOURCE_CACHE_PATH.resolve()),
        "target_cache": str(DEFAULT_TARGET_CACHE_PATH.resolve()),
        "device": "cuda",
    }


def _require_registered_paths(
    data_root: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
    source_cache_path: Path | None = None,
    target_cache_path: Path | None = None,
) -> None:
    pairs = (
        ("data root", data_root, DEFAULT_DATA_ROOT),
        ("checkpoint", checkpoint_path, DEFAULT_CHECKPOINT_PATH),
        ("tokenizer", tokenizer_path, DEFAULT_TOKENIZER_PATH),
    )
    for name, observed, expected in pairs:
        if observed.resolve() != expected.resolve():
            raise HarnessInvalid(f"{name} path is not registered")
    if (
        source_cache_path is not None
        and source_cache_path.resolve() != DEFAULT_SOURCE_CACHE_PATH.resolve()
    ):
        raise HarnessInvalid("source hidden-cache path is not registered")
    if (
        target_cache_path is not None
        and target_cache_path.resolve() != DEFAULT_TARGET_CACHE_PATH.resolve()
    ):
        raise HarnessInvalid("target hidden-cache path is not registered")


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
        raise HarnessInvalid("registered HWU64 manifest file hash changed")
    manifest = verify_manifest(data_root)
    if manifest.get("scientific_sha256") != EXPECTED_MANIFEST_SHA256["scientific"]:
        raise HarnessInvalid("registered HWU64 manifest scientific hash changed")
    return manifest


def artifact_record(
    data_root: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
) -> dict[str, Any]:
    if sha256_file(PREREG_PATH) != EXPECTED_PREREG_SHA256:
        raise HarnessInvalid("preregistration bytes changed after the prereg commit")
    if not prereg_commit_is_ancestor():
        raise HarnessInvalid("registered preregistration commit is not an ancestor")
    return {
        "data": {
            "root": str(data_root),
            "git_commit": DATASET_GIT_COMMIT,
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
            "committed_before_hwu64_encoding": PREREG_COMMIT,
            "commit_is_ancestor": True,
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
    if not git_worktree_is_clean():
        raise HarnessInvalid("preflight requires a clean Git worktree")
    forbidden = (
        PREFLIGHT_PATH,
        REGISTRATION_PATH,
        RESULT_PATH,
        ROWS_PATH,
        REPLAY_PATH,
        DEFAULT_SOURCE_CACHE_PATH,
        DEFAULT_TARGET_CACHE_PATH,
    )
    if any(path.exists() for path in forbidden):
        raise HarnessInvalid("stale preflight, registration, result, or cache exists")
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
        "schema": "hlm5-sca2-preflight-v1",
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
            "hwu64_utterances_encoded": False,
            "routing_outcomes_computed": False,
            "target_utterances_encoded": False,
            "test_utterances_encoded": False,
            "source_cache_existed": False,
            "target_cache_existed": False,
            "only_trunk_input": "fixed synthetic anchor",
        },
    }
    write_json(PREFLIGHT_PATH, record)
    return record


def verify_preflight(record: dict[str, Any]) -> None:
    if record.get("schema") != "hlm5-sca2-preflight-v1":
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
    attestation = record.get("attestation", {})
    if any(
        attestation.get(key)
        for key in (
            "hwu64_utterances_encoded",
            "routing_outcomes_computed",
            "target_utterances_encoded",
            "test_utterances_encoded",
            "source_cache_existed",
            "target_cache_existed",
        )
    ):
        raise HarnessInvalid("preflight attestation reports outcome access")


def register_development() -> dict[str, Any]:
    if not PREFLIGHT_PATH.is_file():
        raise HarnessInvalid("preflight file is missing")
    preflight_record = read_json(PREFLIGHT_PATH)
    verify_preflight(preflight_record)
    if any(
        path.exists()
        for path in (
            RESULT_PATH,
            ROWS_PATH,
            REPLAY_PATH,
            DEFAULT_SOURCE_CACHE_PATH,
            DEFAULT_TARGET_CACHE_PATH,
        )
    ):
        raise HarnessInvalid("outcome or cache artifact exists before registration")
    protocol = scientific_constants()
    record = {
        "schema": "hlm5-sca2-development-registration-v1",
        "status": "REGISTERED_BEFORE_OUTCOMES",
        "generated_at_local": now_local(),
        "git_head": git_head(),
        "preflight": {
            "path": PREFLIGHT_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(PREFLIGHT_PATH),
        },
        "artifacts": preflight_record["artifacts"],
        "environment": preflight_record["environment"],
        "manifest_scientific_sha256": preflight_record[
            "manifest_scientific_sha256"
        ],
        "scientific_constants": protocol,
        "protocol_sha256": stable_sha256(protocol),
        "execution_contract": registered_execution_contract(),
        "execution_contract_sha256": stable_sha256(registered_execution_contract()),
        "random_basis": preflight_record["random_basis"],
        "timing_attestation": {
            "hwu64_utterances_encoded_before_registration": False,
            "routing_outcomes_computed_before_registration": False,
            "target_utterances_encoded": False,
            "test_utterances_encoded": False,
            "source_cache_existed": False,
            "target_cache_existed": False,
        },
    }
    write_json(REGISTRATION_PATH, record)
    return record


def verify_registration(
    registration: dict[str, Any],
    data_root: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
    source_cache_path: Path,
    target_cache_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    _require_registered_paths(
        data_root,
        checkpoint_path,
        tokenizer_path,
        source_cache_path,
        target_cache_path,
    )
    if registration.get("schema") != "hlm5-sca2-development-registration-v1":
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
    if manifest["scientific_sha256"] != registration.get(
        "manifest_scientific_sha256"
    ):
        raise HarnessInvalid("registered manifest scientific hash changed")
    return manifest


def hidden_population_rows(
    populations: dict[str, list[dict[str, Any]]], names: Iterable[str]
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for name in names:
        for row in populations[name]:
            if row["fold"] == TEST_FOLD:
                raise HarnessInvalid("test-fold row reached a hidden population")
            existing = by_id.get(row["row_id"])
            if existing is not None and existing != row:
                raise HarnessInvalid("row-ID collision in hidden population")
            by_id[row["row_id"]] = row
    return [by_id[row_id] for row_id in sorted(by_id)]


def hidden_cache_metadata(
    stage: str,
    rows: list[dict[str, Any]],
    populations: dict[str, list[dict[str, Any]]],
    names: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema": "hlm5-sca2-development-hidden-cache-v1",
        "stage": stage,
        "checkpoint_sha256": verify_model_file(DEFAULT_CHECKPOINT_PATH, "checkpoint"),
        "tokenizer_sha256": verify_model_file(DEFAULT_TOKENIZER_PATH, "tokenizer"),
        "row_population_sha256": population_sha256(rows),
        "population_sha256": {
            name: population_sha256(populations[name]) for name in names
        },
        "population_names": list(names),
        "row_ids": [row["row_id"] for row in rows],
        "folds": sorted({row["fold"] for row in rows}),
        "test_utterances_encoded": False,
    }


def create_hidden_cache(
    stage: str,
    cache_path: Path,
    model: torch.nn.Module,
    tokenizer: Any,
    populations: dict[str, list[dict[str, Any]]],
    names: tuple[str, ...],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, int], dict[str, Any]]:
    if cache_path.exists():
        raise HarnessInvalid(f"{stage} cache existed before first encoding")
    rows = hidden_population_rows(populations, names)
    metadata = hidden_cache_metadata(stage, rows, populations, names)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    hiddens, extraction = extract_hiddens(
        model, tokenizer, rows, device, HIDDEN_BATCH_SIZE
    )
    hiddens = hiddens.to(device="cpu", dtype=torch.float32).contiguous()
    if tuple(hiddens.shape) != (len(rows), 768):
        raise HarnessInvalid(f"{stage} hidden shape changed: {tuple(hiddens.shape)}")
    if not bool(torch.isfinite(hiddens).all()):
        raise HarnessInvalid(f"{stage} hidden cache contains non-finite values")
    cache = {
        "metadata": metadata,
        "extraction": extraction,
        "hidden_sha256": tensor_sha256(hiddens),
        "hiddens": hiddens,
    }
    torch.save(cache, cache_path)
    index = {row_id: position for position, row_id in enumerate(metadata["row_ids"])}
    return hiddens, index, {
        "path": str(cache_path),
        "sha256": sha256_file(cache_path),
        "created_this_invocation": True,
        "hidden_sha256": cache["hidden_sha256"],
        "metadata": metadata,
        "extraction": extraction,
    }


def load_hidden_cache(
    stage: str,
    cache_path: Path,
    populations: dict[str, list[dict[str, Any]]],
    names: tuple[str, ...],
) -> tuple[torch.Tensor, dict[str, int], dict[str, Any]]:
    if not cache_path.is_file():
        raise HarnessInvalid(f"{stage} replay cache is missing")
    rows = hidden_population_rows(populations, names)
    expected = hidden_cache_metadata(stage, rows, populations, names)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cache.get("metadata") != expected:
        raise HarnessInvalid(f"{stage} cache metadata mismatch")
    hiddens = cache.get("hiddens")
    if not isinstance(hiddens, torch.Tensor):
        raise HarnessInvalid(f"{stage} cache tensor is missing")
    hiddens = hiddens.to(device="cpu", dtype=torch.float32).contiguous()
    if tuple(hiddens.shape) != (len(rows), 768):
        raise HarnessInvalid(f"{stage} replay hidden shape changed")
    if not bool(torch.isfinite(hiddens).all()):
        raise HarnessInvalid(f"{stage} replay cache contains non-finite values")
    if tensor_sha256(hiddens) != cache.get("hidden_sha256"):
        raise HarnessInvalid(f"{stage} replay tensor hash mismatch")
    index = {row_id: position for position, row_id in enumerate(expected["row_ids"])}
    return hiddens, index, {
        "path": str(cache_path),
        "sha256": sha256_file(cache_path),
        "created_this_invocation": False,
        "hidden_sha256": cache["hidden_sha256"],
        "metadata": expected,
        "extraction": cache.get("extraction"),
    }


def select_hiddens(
    hiddens: torch.Tensor,
    index: dict[str, int],
    rows: list[dict[str, Any]],
) -> torch.Tensor:
    try:
        positions = [index[row["row_id"]] for row in rows]
    except KeyError as error:
        raise HarnessInvalid("population row is missing from hidden cache") from error
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
    groups = metadata["groups"]
    group_sets = [set(groups[name]) for name in GROUP_NAMES]
    target = set(groups["target_supported"])
    development_names = SOURCE_HIDDEN_POPULATION_NAMES + TARGET_HIDDEN_POPULATION_NAMES
    test_ids = {
        row["row_id"]
        for name in ("target_test", "off_support_test")
        for row in populations[name]
    }
    development_ids = {
        row["row_id"] for name in development_names for row in populations[name]
    }
    return {
        "manifest_population_hashes_match": all(
            manifest["populations"][name]["sha256"]
            == population_sha256(populations[name])
            and manifest["populations"][name]["count"] == len(populations[name])
            for name in populations
        ),
        "manifest_hidden_names_exact": manifest["development_hidden_population_names"]
        == list(development_names),
        "all_64_intents_present": len(metadata["all_intents"]) == 64,
        "eligible_intent_count_exact": len(metadata["eligible_intents"]) == 61,
        "five_groups_exact": all(len(group) == GROUP_INTENT_COUNT for group in group_sets),
        "five_groups_disjoint": sum(len(group) for group in group_sets)
        == len(set().union(*group_sets)),
        "source_fit_rows_exact": len(populations["source_fit"]) == 7047,
        "source_fit_excludes_target": all(
            row["intent"] not in target for row in populations["source_fit"]
        ),
        "source_fit_train_folds_only": all(
            row["fold"] in TRAIN_FOLDS for row in populations["source_fit"]
        ),
        "support_counts_exact": all(
            len(populations[name]) == GROUP_INTENT_COUNT * SUPPORT_PER_INTENT
            for name in ("calibration_support", "audit_support", "target_support")
        ),
        "evaluation_counts_exact": all(
            len(populations[name]) == GROUP_INTENT_COUNT * ROWS_PER_INTENT
            for name in (
                "calibration_positive",
                "calibration_negative",
                "audit_positive",
                "audit_negative",
                "target_development",
                "target_test",
                "off_support_test",
            )
        ),
        "development_fold_exact": all(
            row["fold"] == DEVELOPMENT_FOLD
            for name in (
                "calibration_positive",
                "calibration_negative",
                "audit_positive",
                "audit_negative",
                "target_development",
            )
            for row in populations[name]
        ),
        "test_fold_exact": all(
            row["fold"] == TEST_FOLD
            for name in ("target_test", "off_support_test")
            for row in populations[name]
        ),
        "development_test_ids_disjoint": not bool(development_ids & test_ids),
        "test_unencoded_manifest_attestation": not manifest[
            "test_utterances_tokenized_or_encoded"
        ],
    }


def _values_and_labels(
    hiddens: torch.Tensor,
    index: dict[str, int],
    populations: dict[str, list[dict[str, Any]]],
    names: Iterable[str],
) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    values = {
        name: select_hiddens(hiddens, index, populations[name]) for name in names
    }
    labels = {
        name: [row["intent"] for row in populations[name]] for name in names
    }
    return values, labels


def _snapshot_parity(
    values: dict[str, torch.Tensor],
    operators: Any,
    arm: str,
    queries: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    transformed = {name: apply_arm(value, operators, arm) for name, value in values.items()}
    return {
        query_name: snapshot_reader_parity(
            transformed[query_name], transformed[support_name]
        )
        for query_name, support_name in queries
    }


def compute_source_science(
    hiddens: torch.Tensor,
    index: dict[str, int],
    populations: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, torch.Tensor]]:
    values, labels = _values_and_labels(
        hiddens, index, populations, SOURCE_HIDDEN_POPULATION_NAMES
    )
    operators = fit_key_operators(values["source_fit"])
    calibrations: dict[str, dict[str, Any]] = {}
    calibration_arrays: dict[str, dict[str, Any]] = {}
    parity: dict[str, Any] = {}
    for arm in ARM_NAMES:
        metrics, arrays = calibrate_arm(
            operators,
            arm,
            values["calibration_support"],
            labels["calibration_support"],
            values["calibration_positive"],
            labels["calibration_positive"],
            values["calibration_negative"],
        )
        calibrations[arm] = metrics
        calibration_arrays[arm] = arrays
        parity[arm] = _snapshot_parity(
            values,
            operators,
            arm,
            (
                ("calibration_positive", "calibration_support"),
                ("calibration_negative", "calibration_support"),
            ),
        )

    sufficient = all(
        metrics["thresholds"]["sufficient_margin_errors"]
        for metrics in calibrations.values()
    )
    source: dict[str, Any] = {
        "operators": operators.metadata,
        "calibration": calibrations,
        "calibration_all_arms_have_eight_errors": sufficient,
        "snapshot_float32_reader_parity": parity,
        "source_audit": None,
        "source_audit_bars": None,
        "source_audit_comparisons": None,
        "source_audit_bootstrap": None,
        "source_audit_verdict": (
            "NOT_RUN_INSUFFICIENT_CALIBRATION_ERRORS" if not sufficient else None
        ),
    }
    arrays: dict[str, Any] = {"calibration": calibration_arrays, "source_audit": {}}
    if not sufficient:
        return source, arrays, operators, values

    audit_arms: dict[str, dict[str, Any]] = {}
    audit_arrays: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        metrics, arm_arrays = evaluate_stage(
            operators,
            arm,
            calibrations[arm]["thresholds"],
            values["audit_support"],
            labels["audit_support"],
            values["audit_positive"],
            labels["audit_positive"],
            [row["row_id"] for row in populations["audit_positive"]],
            values["audit_negative"],
        )
        audit_arms[arm] = metrics
        audit_arrays[arm] = arm_arrays
        parity[arm].update(
            _snapshot_parity(
                values,
                operators,
                arm,
                (
                    ("audit_positive", "audit_support"),
                    ("audit_negative", "audit_support"),
                ),
            )
        )
    verdict, bars, comparisons = source_audit_verdict(audit_arms)
    bootstrap = {
        "dual_minus_absolute_only": paired_intent_selective_bootstrap(
            audit_arms[CANDIDATE_ARM], "dual", "absolute_only"
        ),
        "dual_minus_matched_proximity": paired_intent_selective_bootstrap(
            audit_arms[CANDIDATE_ARM], "dual", "matched_proximity_top_k"
        ),
    }
    source.update(
        {
            "source_audit": audit_arms,
            "source_audit_bars": bars,
            "source_audit_comparisons": comparisons,
            "source_audit_bootstrap": bootstrap,
            "source_audit_verdict": verdict,
        }
    )
    arrays["source_audit"] = audit_arrays
    return source, arrays, operators, values


def compute_target_science(
    hiddens: torch.Tensor,
    index: dict[str, int],
    populations: dict[str, list[dict[str, Any]]],
    source_values: dict[str, torch.Tensor],
    source_science: dict[str, Any],
    operators: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_values, target_labels = _values_and_labels(
        hiddens, index, populations, TARGET_HIDDEN_POPULATION_NAMES
    )
    target_values["audit_negative"] = source_values["audit_negative"]
    target_labels["audit_negative"] = [
        row["intent"] for row in populations["audit_negative"]
    ]
    arms: dict[str, dict[str, Any]] = {}
    arrays: dict[str, dict[str, Any]] = {}
    parity: dict[str, Any] = {}
    for arm in ARM_NAMES:
        metrics, arm_arrays = evaluate_stage(
            operators,
            arm,
            source_science["calibration"][arm]["thresholds"],
            target_values["target_support"],
            target_labels["target_support"],
            target_values["target_development"],
            target_labels["target_development"],
            [row["row_id"] for row in populations["target_development"]],
            target_values["audit_negative"],
        )
        arms[arm] = metrics
        arrays[arm] = arm_arrays
        parity[arm] = _snapshot_parity(
            target_values,
            operators,
            arm,
            (
                ("target_development", "target_support"),
                ("audit_negative", "target_support"),
            ),
        )
    verdict, bars, comparisons = development_verdict(arms)
    bootstrap = {
        "dual_minus_absolute_only": paired_intent_selective_bootstrap(
            arms[CANDIDATE_ARM], "dual", "absolute_only"
        ),
        "dual_minus_matched_proximity": paired_intent_selective_bootstrap(
            arms[CANDIDATE_ARM], "dual", "matched_proximity_top_k"
        ),
    }
    return (
        {
            "arms": arms,
            "bars": bars,
            "comparisons": comparisons,
            "bootstrap": bootstrap,
            "snapshot_float32_reader_parity": parity,
            "verdict": verdict,
        },
        arrays,
    )


def _row_records_for_supported(
    stage: str,
    population: list[dict[str, Any]],
    arm_metrics: dict[str, dict[str, Any]],
    arm_arrays: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(population):
        arms: dict[str, Any] = {}
        for arm in ARM_NAMES:
            metrics = arm_metrics[arm]
            arrays = arm_arrays[arm]["supported"]
            arms[arm] = {
                "predicted_intent": arrays["predicted"][row_index],
                "maximum_score": float(arrays["maximum_scores"][row_index].item()),
                "class_margin": float(arrays["class_margins"][row_index].item()),
                "tau_absolute": metrics["thresholds"]["tau_absolute"],
                "tau_margin": metrics["thresholds"]["tau_margin"],
                "correct": bool(arrays["correct"][row_index].item()),
                "admission": {
                    name: bool(mask[row_index].item())
                    for name, mask in arrays["masks"].items()
                },
            }
        records.append(
            {
                "stage": stage,
                "kind": "supported",
                "row_id": row["row_id"],
                "fold": row["fold"],
                "source_index": row["source_index"],
                "utterance_sha256": row["utterance_sha256"],
                "true_intent": row["intent"],
                "arms": arms,
            }
        )
    return records


def _row_records_for_negative(
    stage: str,
    population: list[dict[str, Any]],
    arm_metrics: dict[str, dict[str, Any]],
    arm_arrays: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(population):
        arms: dict[str, Any] = {}
        for arm in ARM_NAMES:
            metrics = arm_metrics[arm]
            arrays = arm_arrays[arm]["off_support"]
            arms[arm] = {
                "predicted_intent": arrays["predicted"][row_index],
                "maximum_score": float(arrays["maximum_scores"][row_index].item()),
                "class_margin": float(arrays["class_margins"][row_index].item()),
                "tau_absolute": metrics["thresholds"]["tau_absolute"],
                "tau_margin": metrics["thresholds"]["tau_margin"],
                "admission": {
                    name: bool(mask[row_index].item())
                    for name, mask in arrays["masks"].items()
                },
            }
        records.append(
            {
                "stage": stage,
                "kind": "off_support",
                "row_id": row["row_id"],
                "fold": row["fold"],
                "source_index": row["source_index"],
                "utterance_sha256": row["utterance_sha256"],
                "source_intent": row["intent"],
                "arms": arms,
            }
        )
    return records


def _calibration_row_records(
    populations: dict[str, list[dict[str, Any]]],
    science: dict[str, Any],
    arrays: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for kind, population_name, route_name in (
        ("supported", "calibration_positive", "positive_route"),
        ("off_support", "calibration_negative", "negative_route"),
    ):
        for row_index, row in enumerate(populations[population_name]):
            arms: dict[str, Any] = {}
            for arm in ARM_NAMES:
                thresholds = science["calibration"][arm]["thresholds"]
                route = arrays["calibration"][arm][route_name]
                maximum_score = float(route["maximum_scores"][row_index].item())
                class_margin = float(route["class_margins"][row_index].item())
                absolute = maximum_score > thresholds["tau_absolute"]
                tau_margin = thresholds["tau_margin"]
                dual = (
                    None
                    if tau_margin is None
                    else absolute and class_margin > tau_margin
                )
                arms[arm] = {
                    "predicted_intent": route["predicted"][row_index],
                    "maximum_score": maximum_score,
                    "class_margin": class_margin,
                    "tau_absolute": thresholds["tau_absolute"],
                    "tau_margin": tau_margin,
                    "admission": {
                        "absolute_only": absolute,
                        "dual": dual,
                    },
                }
                if kind == "supported":
                    arms[arm]["correct"] = (
                        route["predicted"][row_index] == row["intent"]
                    )
            record = {
                "stage": population_name,
                "kind": kind,
                "row_id": row["row_id"],
                "fold": row["fold"],
                "source_index": row["source_index"],
                "utterance_sha256": row["utterance_sha256"],
                "arms": arms,
            }
            if kind == "supported":
                record["true_intent"] = row["intent"]
            else:
                record["source_intent"] = row["intent"]
            records.append(record)
    return records


def build_row_records(
    populations: dict[str, list[dict[str, Any]]],
    source_science: dict[str, Any],
    source_arrays: dict[str, Any],
    target_science: dict[str, Any] | None,
    target_arrays: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    records = _calibration_row_records(populations, source_science, source_arrays)
    if source_science["source_audit"] is not None:
        records.extend(
            _row_records_for_supported(
                "source_audit_positive",
                populations["audit_positive"],
                source_science["source_audit"],
                source_arrays["source_audit"],
            )
        )
        records.extend(
            _row_records_for_negative(
                "source_audit_negative",
                populations["audit_negative"],
                source_science["source_audit"],
                source_arrays["source_audit"],
            )
        )
    if target_science is not None and target_arrays is not None:
        records.extend(
            _row_records_for_supported(
                "target_development",
                populations["target_development"],
                target_science["arms"],
                target_arrays,
            )
        )
        records.extend(
            _row_records_for_negative(
                "target_off_support",
                populations["audit_negative"],
                target_science["arms"],
                target_arrays,
            )
        )
    return records


def _rows_text(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    )


def _science_validity(
    source_science: dict[str, Any],
    target_science: dict[str, Any] | None,
) -> dict[str, bool]:
    calibrations = source_science["calibration"]
    calibration_rates_valid = all(
        metrics["thresholds"]["calibration_negative_absolute_rate"] <= 0.05
        and (
            not metrics["thresholds"]["sufficient_margin_errors"]
            or metrics["thresholds"]["calibration_error_margin_survival_rate"]
            <= 0.10
        )
        for metrics in calibrations.values()
    )
    parity_records = [
        record
        for arm in source_science["snapshot_float32_reader_parity"].values()
        for record in arm.values()
    ]
    if target_science is not None:
        parity_records.extend(
            record
            for arm in target_science["snapshot_float32_reader_parity"].values()
            for record in arm.values()
        )
    independent_errors = [
        error
        for metrics in calibrations.values()
        for error in (
            metrics["positive_independent_max_abs_error"],
            metrics["negative_independent_max_abs_error"],
        )
    ]
    matched_counts = True
    stages = []
    if source_science["source_audit"] is not None:
        stages.append(source_science["source_audit"])
    if target_science is not None:
        stages.append(target_science["arms"])
    for stage in stages:
        for arm in stage.values():
            independent_errors.extend(
                (
                    arm["supported"]["independent_max_abs_error"],
                    arm["off_support"]["independent_max_abs_error"],
                )
            )
            gates = arm["supported"]["gates"]
            matched_counts &= (
                gates["dual"]["admitted_count"]
                == gates["matched_proximity_top_k"]["admitted_count"]
            )
    operators = source_science["operators"]
    return {
        "blind_inverse_root_identity": operators["blind_zca"][
            "inverse_identity_max_abs_error"
        ]
        <= 1e-10,
        "matched_spectrum_absolute": operators["matched_spectrum_max_abs_error"]
        <= 1e-10,
        "matched_spectrum_relative": operators["matched_spectrum_max_rel_error"]
        <= 1e-10,
        "blind_and_random_bases_differ": operators["basis_hashes_differ"],
        "random_basis_orthogonal": operators[
            "random_basis_orthogonality_max_abs_error"
        ]
        <= 1e-10,
        "random_basis_hash_registered": operators["random_basis_sha256"]
        == EXPECTED_RANDOM_BASIS_SHA256,
        "calibration_construction_rates_valid": calibration_rates_valid,
        "degree5_cosine_numpy_class_reader_match": all(
            error <= 1e-10 for error in independent_errors
        ),
        "snapshot_float32_reader_predictions_match": all(
            record["prediction_match"] for record in parity_records
        ),
        "matched_proximity_counts_exact": bool(matched_counts),
        "scientific_values_finite": all_finite(source_science)
        and (target_science is None or all_finite(target_science)),
    }


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
    source_cache_path = args.source_cache.resolve()
    target_cache_path = args.target_cache.resolve()
    _require_registered_paths(
        data_root,
        checkpoint_path,
        tokenizer_path,
        source_cache_path,
        target_cache_path,
    )
    if not REGISTRATION_PATH.is_file():
        raise HarnessInvalid("development registration is missing")
    if any(path.exists() for path in (RESULT_PATH, ROWS_PATH, REPLAY_PATH)):
        raise HarnessInvalid("canonical result already exists; use replay")
    if source_cache_path.exists() or target_cache_path.exists():
        raise HarnessInvalid("unprovenanced cache exists before first result")
    registration = read_json(REGISTRATION_PATH)
    manifest = verify_registration(
        registration,
        data_root,
        checkpoint_path,
        tokenizer_path,
        source_cache_path,
        target_cache_path,
        device,
    )
    built = build_populations(data_root)
    populations = built["rows"]
    population_validity = _population_validity(
        populations, built["metadata"], manifest
    )
    if not all(population_validity.values()):
        raise HarnessInvalid(f"population validity failed: {population_validity}")
    model, tokenizer, model_metadata = load_model_and_tokenizer(
        checkpoint_path, tokenizer_path, device
    )
    source_hiddens, source_index, source_cache_record = create_hidden_cache(
        "source_audit",
        source_cache_path,
        model,
        tokenizer,
        populations,
        SOURCE_HIDDEN_POPULATION_NAMES,
        device,
    )
    source_science, source_arrays, operators, source_values = compute_source_science(
        source_hiddens, source_index, populations
    )

    target_science: dict[str, Any] | None = None
    target_arrays: dict[str, Any] | None = None
    target_cache_record: dict[str, Any] | None = None
    if not source_science["calibration_all_arms_have_eight_errors"]:
        verdict = "STOP_BEFORE_TARGET"
    elif source_science["source_audit_verdict"] != "PASS_TO_TARGET":
        verdict = "STOP_BEFORE_TARGET"
    else:
        if target_cache_path.exists():
            raise HarnessInvalid("target cache existed before source audit passed")
        target_hiddens, target_index, target_cache_record = create_hidden_cache(
            "target_development",
            target_cache_path,
            model,
            tokenizer,
            populations,
            TARGET_HIDDEN_POPULATION_NAMES,
            device,
        )
        target_science, target_arrays = compute_target_science(
            target_hiddens,
            target_index,
            populations,
            source_values,
            source_science,
            operators,
        )
        verdict = target_science["verdict"]

    if verdict == "STOP_BEFORE_TARGET" and target_cache_path.exists():
        raise HarnessInvalid("target cache exists despite failed source audit")
    science_validity = _science_validity(source_science, target_science)
    validity = {
        "registration_verified": True,
        "manifest_verified": True,
        "prereg_committed_before_hwu64_encoding": prereg_commit_is_ancestor(),
        "model_strict_load_and_eval": not model.training,
        **population_validity,
        "source_hidden_dimension_exact": source_hiddens.shape[1] == 768,
        "source_hiddens_finite": bool(torch.isfinite(source_hiddens).all()),
        "source_audit_preceded_target_encoding": True,
        "failed_source_audit_left_target_unencoded": (
            verdict != "STOP_BEFORE_TARGET" or target_cache_record is None
        ),
        "test_utterances_not_encoded": True,
        **science_validity,
    }
    if target_cache_record is not None:
        target_folds = set(target_cache_record["metadata"]["folds"])
        validity["target_hidden_stage_and_folds_exact"] = (
            target_cache_record["metadata"]["stage"] == "target_development"
            and DEVELOPMENT_FOLD in target_folds
            and TEST_FOLD not in target_folds
            and target_folds.issubset(set(TRAIN_FOLDS) | {DEVELOPMENT_FOLD})
        )
    if not all(value is True for value in validity.values()):
        raise HarnessInvalid(f"binding validity failure: {validity}")

    scientific = {
        "population": {
            "manifest_scientific_sha256": manifest["scientific_sha256"],
            "counts": {name: len(rows) for name, rows in populations.items()},
            "hashes": {
                name: population_sha256(rows) for name, rows in populations.items()
            },
            "groups": built["metadata"]["groups"],
            "test_utterances_tokenized_or_encoded": False,
        },
        "source": source_science,
        "target": target_science,
        "verdict": verdict,
    }
    if not all_finite(scientific):
        raise HarnessInvalid("scientific object contains non-finite values")
    row_records = build_row_records(
        populations,
        source_science,
        source_arrays,
        target_science,
        target_arrays,
    )
    rows_text = _rows_text(row_records)
    result = {
        "schema": "hlm5-sca2-development-result-v1",
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
            "execution_contract_sha256": registration[
                "execution_contract_sha256"
            ],
        },
        "artifacts": registration["artifacts"],
        "model": model_metadata,
        "hidden_caches": {
            "source": source_cache_record,
            "target": target_cache_record,
        },
        "validity": validity,
        "scientific": scientific,
        "scientific_sha256": stable_sha256(scientific),
        "rows_object_sha256": stable_sha256(row_records),
        "rows_jsonl_sha256": sha256_bytes(rows_text.encode("utf-8")),
        "claim_scope": (
            "Balanced three-shot HWU64 selective routing in frozen HLM5-136M "
            "final-hidden geometry only, with source-only two-axis calibration "
            "and a cache-equivalent static reader; no recurrence, test, scaling, "
            "or product claim."
        ),
    }
    ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ROWS_PATH.write_text(rows_text, encoding="utf-8", newline="\n")
    if sha256_file(ROWS_PATH) != result["rows_jsonl_sha256"]:
        raise HarnessInvalid("written row artifact hash mismatch")
    write_json(RESULT_PATH, result)
    return result


def replay_development(args: argparse.Namespace) -> dict[str, Any]:
    configure_determinism()
    if args.device != "cuda":
        raise HarnessInvalid("replay device must be exactly 'cuda'")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise HarnessInvalid("replay is registered for local CUDA")
    data_root = args.data_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    tokenizer_path = args.tokenizer.resolve()
    source_cache_path = args.source_cache.resolve()
    target_cache_path = args.target_cache.resolve()
    if not RESULT_PATH.is_file() or not ROWS_PATH.is_file():
        raise HarnessInvalid("canonical result and rows are required for replay")
    if REPLAY_PATH.exists():
        raise HarnessInvalid("replay receipt already exists")
    registration = read_json(REGISTRATION_PATH)
    manifest = verify_registration(
        registration,
        data_root,
        checkpoint_path,
        tokenizer_path,
        source_cache_path,
        target_cache_path,
        device,
    )
    canonical = read_json(RESULT_PATH)
    if canonical.get("schema") != "hlm5-sca2-development-result-v1":
        raise HarnessInvalid("canonical result schema mismatch")
    if canonical.get("registration", {}).get("sha256") != sha256_file(
        REGISTRATION_PATH
    ):
        raise HarnessInvalid("canonical result does not bind registration")
    if canonical.get("hidden_caches", {}).get("source", {}).get("sha256") != sha256_file(
        source_cache_path
    ):
        raise HarnessInvalid("canonical result does not bind source cache")
    built = build_populations(data_root)
    populations = built["rows"]
    source_hiddens, source_index, source_cache_record = load_hidden_cache(
        "source_audit",
        source_cache_path,
        populations,
        SOURCE_HIDDEN_POPULATION_NAMES,
    )
    source_science, source_arrays, operators, source_values = compute_source_science(
        source_hiddens, source_index, populations
    )
    target_science: dict[str, Any] | None = None
    target_arrays: dict[str, Any] | None = None
    target_cache_record: dict[str, Any] | None = None
    if source_science["source_audit_verdict"] == "PASS_TO_TARGET":
        if canonical.get("hidden_caches", {}).get("target", {}).get(
            "sha256"
        ) != sha256_file(target_cache_path):
            raise HarnessInvalid("canonical result does not bind target cache")
        target_hiddens, target_index, target_cache_record = load_hidden_cache(
            "target_development",
            target_cache_path,
            populations,
            TARGET_HIDDEN_POPULATION_NAMES,
        )
        target_science, target_arrays = compute_target_science(
            target_hiddens,
            target_index,
            populations,
            source_values,
            source_science,
            operators,
        )
        verdict = target_science["verdict"]
    else:
        if target_cache_path.exists():
            raise HarnessInvalid("failed source audit unexpectedly has target cache")
        verdict = "STOP_BEFORE_TARGET"
    scientific = {
        "population": {
            "manifest_scientific_sha256": manifest["scientific_sha256"],
            "counts": {name: len(rows) for name, rows in populations.items()},
            "hashes": {
                name: population_sha256(rows) for name, rows in populations.items()
            },
            "groups": built["metadata"]["groups"],
            "test_utterances_tokenized_or_encoded": False,
        },
        "source": source_science,
        "target": target_science,
        "verdict": verdict,
    }
    row_records = build_row_records(
        populations,
        source_science,
        source_arrays,
        target_science,
        target_arrays,
    )
    rows_text = _rows_text(row_records)
    replay_checks = {
        "scientific_sha256_exact": stable_sha256(scientific)
        == canonical["scientific_sha256"],
        "rows_object_sha256_exact": stable_sha256(row_records)
        == canonical["rows_object_sha256"],
        "rows_jsonl_sha256_exact": sha256_bytes(rows_text.encode("utf-8"))
        == canonical["rows_jsonl_sha256"]
        == sha256_file(ROWS_PATH),
        "source_cache_hash_exact": source_cache_record["sha256"]
        == canonical["hidden_caches"]["source"]["sha256"],
        "target_cache_hash_exact": (
            target_cache_record is None
            and canonical["hidden_caches"]["target"] is None
        )
        or (
            target_cache_record is not None
            and canonical["hidden_caches"]["target"] is not None
            and target_cache_record["sha256"]
            == canonical["hidden_caches"]["target"]["sha256"]
        ),
        "verdict_exact": verdict == canonical["verdict"],
    }
    if not all(replay_checks.values()):
        raise HarnessInvalid(f"replay mismatch: {replay_checks}")
    receipt = {
        "schema": "hlm5-sca2-development-replay-v1",
        "status": "EXACT_REPLAY",
        "generated_at_local": now_local(),
        "git_head": git_head(),
        "command": [sys.executable, *sys.argv],
        "canonical_result": {
            "path": RESULT_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(RESULT_PATH),
        },
        "canonical_rows": {
            "path": ROWS_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(ROWS_PATH),
        },
        "checks": replay_checks,
    }
    write_json(REPLAY_PATH, receipt)
    return receipt


def common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--device", default="cuda")


def execution_paths(parser: argparse.ArgumentParser) -> None:
    common_paths(parser)
    parser.add_argument(
        "--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE_PATH
    )
    parser.add_argument(
        "--target-cache", type=Path, default=DEFAULT_TARGET_CACHE_PATH
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    common_paths(preflight_parser)
    subparsers.add_parser("register")
    development_parser = subparsers.add_parser("development")
    execution_paths(development_parser)
    replay_parser = subparsers.add_parser("replay")
    execution_paths(replay_parser)
    return parser.parse_args()


def _print_result(result: dict[str, Any]) -> None:
    print(f"status={result['status']}")
    if "verdict" in result:
        print(f"verdict={result['verdict']}")
    if "scientific_sha256" in result:
        print(f"scientific_sha256={result['scientific_sha256']}")


def main() -> None:
    args = parse_args()
    if args.mode == "preflight":
        result = preflight(args)
        _print_result(result)
        print(f"preflight_sha256={sha256_file(PREFLIGHT_PATH)}")
        print("hwu64_utterances_encoded=false")
    elif args.mode == "register":
        result = register_development()
        _print_result(result)
        print(f"registration_sha256={sha256_file(REGISTRATION_PATH)}")
    elif args.mode == "development":
        result = run_development(args)
        _print_result(result)
        source = result["scientific"]["source"]
        print(f"source_audit={source['source_audit_verdict']}")
        if result["scientific"]["target"] is not None:
            candidate = result["scientific"]["target"]["arms"][CANDIDATE_ARM]
            dual = candidate["supported"]["gates"]["dual"]
            print(
                "blind_zca_dual: "
                f"correct_admission={dual['macro_correct_admission_rate']:.9f} "
                f"coverage={dual['macro_coverage']:.9f} "
                f"selective_accuracy={dual['selective_accuracy']:.9f}"
            )
        print("test_utterances_encoded=false")
    elif args.mode == "replay":
        result = replay_development(args)
        _print_result(result)
        print(f"replay_sha256={sha256_file(REPLAY_PATH)}")
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()

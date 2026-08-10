"""Preflight, register, and execute the Banking77 task-aligned energy gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from contract import (
    DEVELOPMENT_MANIFEST_PATH,
    DEV_REGISTERED_COMMAND,
    DEV_RESULT_PATH,
    DEV_ROWS_PATH,
    DEV_VERDICT_PATH,
    HASHED_IMPLEMENTATION_PATHS,
    MODEL_ID,
    MODEL_REVISION,
    PREFLIGHT_PATH,
    REGISTRATION_PATH,
    ROOT,
    SCIENTIFIC_CONSTANTS,
    SOURCE_COMMIT,
    TEST_MANIFEST_PATH,
    TEST_REGISTERED_COMMAND,
    TEST_REGISTRATION_PATH,
    TEST_RESULT_PATH,
    TEST_ROWS_PATH,
    TEST_VERDICT_PATH,
)
from energy import (
    AlignedModel,
    aligned_scores,
    centroid_control_scores,
    constrained_settle,
    decision_boundary_radius,
    fit_aligned_model,
    fit_centroid_controls,
    independent_matched_settle,
    transform,
    unit,
)
from metrics import (
    classify_development_verdict,
    intent_accuracies,
    macro_accuracy,
    maximum_intent_prediction_share,
    paired_intent_bootstrap,
    row_accuracy,
    select_highest_macro,
    select_shrinkage,
    transition_counts,
)
from prepare_banking77 import normalize_text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
                + "\n"
            )


def implementation_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in HASHED_IMPLEMENTATION_PATHS:
        if not path.is_file():
            raise FileNotFoundError(f"required registered file is missing: {path}")
        hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return dict(sorted(hashes.items()))


def environment_info() -> dict[str, Any]:
    import sentence_transformers
    import transformers

    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "transformers": transformers.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def git_info() -> dict[str, Any]:
    top_level = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    ).resolve()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    allowed = {
        path.resolve().relative_to(top_level).as_posix()
        for path in (
            PREFLIGHT_PATH,
            REGISTRATION_PATH,
            DEV_ROWS_PATH,
            DEV_RESULT_PATH,
            DEV_VERDICT_PATH,
            TEST_REGISTRATION_PATH,
            TEST_ROWS_PATH,
            TEST_RESULT_PATH,
            TEST_VERDICT_PATH,
        )
    }
    unexpected = [
        line
        for line in status
        if len(line) < 4 or line[3:].replace("\\", "/") not in allowed
    ]
    return {"head": head, "branch": branch, "unexpected_status": unexpected}


def model_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    ).resolve()


def model_file_hashes(snapshot: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(snapshot.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file():
            hashes[path.relative_to(snapshot).as_posix()] = sha256_file(path.resolve())
    if not hashes:
        raise RuntimeError("model snapshot contains no files")
    return hashes


def load_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        device="cpu",
        local_files_only=True,
    )


def encode_texts(encoder, texts: list[str]) -> torch.Tensor:
    embeddings = encoder.encode(
        texts,
        batch_size=int(SCIENTIFIC_CONSTANTS["encoder_batch_size"]),
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return unit(embeddings.detach().cpu().to(torch.float64))


def tensor_digest(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(header + contiguous.numpy().tobytes(order="C")).hexdigest()


def all_finite(value: Any) -> bool:
    """Reject NaN/Inf anywhere in a nested scientific record."""

    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    return False


def manifest_errors(development: dict[str, Any], test: dict[str, Any]) -> list[str]:
    constants = SCIENTIFIC_CONSTANTS
    errors: list[str] = []
    if development.get("phase") != "development" or test.get("phase") != "test":
        errors.append("manifest phase mismatch")
    if development.get("experiment_id") != constants["experiment_id"]:
        errors.append("development experiment ID mismatch")
    if test.get("experiment_id") != constants["experiment_id"]:
        errors.append("test experiment ID mismatch")
    if development.get("source") != test.get("source"):
        errors.append("manifest source records differ")
    if development.get("source", {}).get("commit") != SOURCE_COMMIT:
        errors.append("source commit mismatch")
    categories = development.get("categories", [])
    if (
        categories != test.get("categories")
        or len(categories) != constants["intent_count"]
    ):
        errors.append("category order/count mismatch")
    if len(set(categories)) != len(categories):
        errors.append("categories are not unique")
    expected_development_counts = {
        "official_train": constants["official_train_count"],
        "deduplicated_train": constants["deduplicated_train_count"],
        "fit": constants["fit_count"],
        "calibration": constants["calibration_count"],
        "development": constants["development_count"],
        "train_duplicate_exclusions": constants["normalized_train_duplicate_groups"],
    }
    if development.get("counts") != expected_development_counts:
        errors.append("development manifest counts mismatch")
    expected_test_counts = {
        "official": constants["official_test_count"],
        "primary": constants["primary_test_count"],
        "normalized_train_overlap_exclusions": constants[
            "normalized_train_test_overlap_count"
        ],
    }
    if test.get("counts") != expected_test_counts:
        errors.append("test manifest counts mismatch")

    role_ids: dict[str, set[str]] = {}
    role_normalized: dict[str, set[str]] = {}
    for role in ("fit", "calibration", "development"):
        rows = development.get(role, [])
        if len(rows) != int(expected_development_counts[role]):
            errors.append(f"{role} row count mismatch")
            continue
        role_ids[role] = {str(row.get("row_id")) for row in rows}
        role_normalized[role] = {normalize_text(str(row.get("text"))) for row in rows}
        if len(role_ids[role]) != len(rows) or len(role_normalized[role]) != len(rows):
            errors.append(f"{role} contains duplicate rows/text")
        for row in rows:
            text_hash = hashlib.sha256(
                normalize_text(str(row.get("text"))).encode("utf-8")
            ).hexdigest()
            if row.get("normalized_text_sha256") != text_hash:
                errors.append(f"{role} normalized text hash mismatch")
                break
            target = row.get("target_index")
            if not isinstance(target, int) or not 0 <= target < int(
                constants["intent_count"]
            ):
                errors.append(f"{role} target out of range")
                break
            if categories[target] != row.get("intent"):
                errors.append(f"{role} target/intent mapping mismatch")
                break
        counts = np.bincount(
            [int(row["target_index"]) for row in rows],
            minlength=int(constants["intent_count"]),
        )
        if role in ("calibration", "development"):
            if not bool(np.all(counts == int(constants[f"{role}_per_intent"]))):
                errors.append(f"{role} is not exactly balanced")
        elif int(counts.min()) != 15:
            errors.append(
                "fit does not span all intents with registered minimum support"
            )
    if len(role_ids) == 3:
        if any(
            role_ids[left] & role_ids[right]
            for index, left in enumerate(role_ids)
            for right in list(role_ids)[index + 1 :]
        ):
            errors.append("development roles overlap by row ID")
        if any(
            role_normalized[left] & role_normalized[right]
            for index, left in enumerate(role_normalized)
            for right in list(role_normalized)[index + 1 :]
        ):
            errors.append("development roles overlap by normalized text")

    official_rows = test.get("official", [])
    primary_rows = test.get("primary", [])
    excluded_rows = test.get("normalized_train_overlap_exclusions", [])
    if len(official_rows) != int(constants["official_test_count"]):
        errors.append("official test row count mismatch")
    if len(primary_rows) != int(constants["primary_test_count"]):
        errors.append("primary test row count mismatch")
    if len(excluded_rows) != int(constants["normalized_train_test_overlap_count"]):
        errors.append("test exclusion count mismatch")
    official_ids = {str(row.get("row_id")) for row in official_rows}
    primary_ids = {str(row.get("row_id")) for row in primary_rows}
    excluded_ids = {str(row.get("row_id")) for row in excluded_rows}
    if primary_ids & excluded_ids or primary_ids | excluded_ids != official_ids:
        errors.append("primary/excluded test partition is not exact")
    if len(role_normalized) == 3:
        normalized_training = set().union(*role_normalized.values())
        normalized_primary = {
            normalize_text(str(row.get("text"))) for row in primary_rows
        }
        normalized_excluded = {
            normalize_text(str(row.get("text"))) for row in excluded_rows
        }
        if normalized_training & normalized_primary:
            errors.append("primary test has normalized training overlap")
        if len(normalized_training & normalized_excluded) != int(
            constants["normalized_train_test_overlap_count"]
        ):
            errors.append("test exclusions do not exactly realize overlap rule")
    for row in official_rows:
        target = row.get("target_index")
        if not isinstance(target, int) or not 0 <= target < int(
            constants["intent_count"]
        ):
            errors.append("official test target out of range")
            break
        if categories[target] != row.get("intent"):
            errors.append("official test target/intent mapping mismatch")
            break
    return errors


def run_test_scripts() -> list[dict[str, Any]]:
    tests = (
        ROOT / "tests" / "test_prepare_banking77.py",
        ROOT / "tests" / "test_energy.py",
        ROOT / "tests" / "test_registration.py",
    )
    records: list[dict[str, Any]] = []
    for path in tests:
        completed = subprocess.run(
            [sys.executable, "-B", str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        records.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "returncode": completed.returncode,
                "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
            }
        )
        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)
            raise RuntimeError(f"test failed: {path}")
    return records


def synthetic_preflight() -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(20_260_810)
    class_count = 4
    features = []
    targets = []
    for class_index in range(class_count):
        center = torch.zeros(7, dtype=torch.float64)
        center[class_index] = 1.0
        noise = 0.05 * torch.randn((12, 7), generator=generator, dtype=torch.float64)
        features.append(unit(center + noise))
        targets.extend([class_index] * 12)
    fit_features = torch.cat(features, dim=0)
    fit_targets = torch.tensor(targets, dtype=torch.int64)
    model = fit_aligned_model(fit_features, fit_targets, class_count, 0.25)
    states = transform(fit_features[:8], model)
    radius = decision_boundary_radius(
        transform(fit_features, model), fit_targets, model.centroids
    )
    kwargs = dynamics_kwargs(int(SCIENTIFIC_CONSTANTS["steps"]))
    candidate, trace = constrained_settle(states, model.centroids, radius, **kwargs)
    matched = independent_matched_settle(states, model.centroids, radius, **kwargs)
    maximum_error = float(torch.max(torch.abs(candidate - matched)))
    maximum_energy_increase = float(
        torch.max(trace.energies[:, 1:] - trace.energies[:, :-1])
    )
    maximum_cap_error = float(
        torch.max(torch.linalg.vector_norm(candidate - states, dim=1) - radius)
    )
    maximum_armijo = float(torch.max(trace.armijo_residuals))
    if maximum_error > float(SCIENTIFIC_CONSTANTS["matched_state_tolerance"]):
        raise RuntimeError("synthetic matched optimizer disagrees")
    if maximum_energy_increase > float(SCIENTIFIC_CONSTANTS["energy_tolerance"]):
        raise RuntimeError("synthetic energy descent failed")
    if maximum_cap_error > float(SCIENTIFIC_CONSTANTS["cap_tolerance"]):
        raise RuntimeError("synthetic cap bound failed")
    if maximum_armijo > float(SCIENTIFIC_CONSTANTS["energy_tolerance"]):
        raise RuntimeError("synthetic Armijo bound failed")
    return {
        "fit_shape": list(fit_features.shape),
        "projection_shape": list(model.projection.shape),
        "trust_radius": radius,
        "matched_max_abs_error": maximum_error,
        "maximum_energy_increase": maximum_energy_increase,
        "maximum_cap_error": maximum_cap_error,
        "maximum_armijo_residual": maximum_armijo,
    }


def run_preflight() -> None:
    if REGISTRATION_PATH.exists() or TEST_REGISTRATION_PATH.exists():
        raise RuntimeError("a registration exists; preflight is frozen")
    development = load_json(DEVELOPMENT_MANIFEST_PATH)
    test = load_json(TEST_MANIFEST_PATH)
    errors = manifest_errors(development, test)
    if errors:
        raise RuntimeError("; ".join(errors))
    repository = git_info()
    if repository["unexpected_status"]:
        raise RuntimeError(
            f"unexpected worktree changes: {repository['unexpected_status']}"
        )
    tests = run_test_scripts()
    snapshot_hashes = model_file_hashes(model_snapshot())
    encoder = load_encoder()
    synthetic_embeddings = encode_texts(
        encoder, ["synthetic banking intent alpha", "synthetic banking intent beta"]
    )
    if tuple(synthetic_embeddings.shape) != (2, 384):
        raise RuntimeError("unexpected encoder output shape")
    receipt = {
        "schema_version": 1,
        "experiment_id": SCIENTIFIC_CONSTANTS["experiment_id"],
        "status": "SYNTHETIC_PREFLIGHT_ONLY",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "real_banking77_embeddings_computed": False,
        "test_embeddings_computed": False,
        "development_manifest_sha256": sha256_file(DEVELOPMENT_MANIFEST_PATH),
        "test_manifest_sha256": sha256_file(TEST_MANIFEST_PATH),
        "implementation_sha256": implementation_hashes(),
        "git": repository,
        "environment": environment_info(),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "snapshot_file_sha256": snapshot_hashes,
            "snapshot_digest": canonical_json_sha256(snapshot_hashes),
            "synthetic_embedding_digest": tensor_digest(synthetic_embeddings),
        },
        "synthetic_math": synthetic_preflight(),
        "tests": tests,
    }
    write_json(PREFLIGHT_PATH, receipt)
    print(f"Synthetic-only preflight PASS: {PREFLIGHT_PATH}")


def create_development_registration() -> None:
    if REGISTRATION_PATH.exists():
        raise RuntimeError("development registration already exists")
    preflight = load_json(PREFLIGHT_PATH)
    repository = git_info()
    current_hashes = implementation_hashes()
    current_model_hashes = model_file_hashes(model_snapshot())
    if preflight.get("real_banking77_embeddings_computed") is not False:
        raise RuntimeError("preflight real-data access flag is not false")
    if preflight.get("test_embeddings_computed") is not False:
        raise RuntimeError("preflight test-access flag is not false")
    if preflight.get("implementation_sha256") != current_hashes:
        raise RuntimeError("implementation changed after preflight")
    if preflight.get("environment") != environment_info():
        raise RuntimeError("environment changed after preflight")
    if preflight.get("git", {}).get("head") != repository["head"]:
        raise RuntimeError("Git HEAD changed after preflight")
    if repository["unexpected_status"]:
        raise RuntimeError(
            f"unexpected worktree changes: {repository['unexpected_status']}"
        )
    if preflight.get("model", {}).get("snapshot_file_sha256") != current_model_hashes:
        raise RuntimeError("model snapshot changed after preflight")
    registration = {
        "schema_version": 1,
        "experiment_id": SCIENTIFIC_CONSTANTS["experiment_id"],
        "phase": "development",
        "status": "PRE_OUTCOME_INTERNAL_REGISTRATION",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "registered_command": DEV_REGISTERED_COMMAND,
        "scientific_constants": SCIENTIFIC_CONSTANTS,
        "implementation_sha256": current_hashes,
        "development_manifest_sha256": sha256_file(DEVELOPMENT_MANIFEST_PATH),
        "test_manifest_sha256": sha256_file(TEST_MANIFEST_PATH),
        "preflight_sha256": sha256_file(PREFLIGHT_PATH),
        "environment": environment_info(),
        "git": repository,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "snapshot_file_sha256": current_model_hashes,
            "snapshot_digest": canonical_json_sha256(current_model_hashes),
        },
    }
    registration["scientific_digest"] = canonical_json_sha256(registration)
    write_json(REGISTRATION_PATH, registration)
    print(f"Registered before Banking77 encoder outcomes: {REGISTRATION_PATH}")
    print(f"Scientific digest: {registration['scientific_digest']}")


def registration_digest_matches(registration: dict[str, Any]) -> bool:
    payload = dict(registration)
    recorded = payload.pop("scientific_digest", None)
    return canonical_json_sha256(payload) == recorded


def development_registration_errors(registration: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    repository = git_info()
    if not registration_digest_matches(registration):
        errors.append("registration scientific digest mismatch")
    if registration.get("registered_command") != DEV_REGISTERED_COMMAND:
        errors.append("registered development command mismatch")
    if registration.get("scientific_constants") != SCIENTIFIC_CONSTANTS:
        errors.append("scientific constants mismatch")
    if registration.get("implementation_sha256") != implementation_hashes():
        errors.append("implementation hashes mismatch")
    if registration.get("development_manifest_sha256") != sha256_file(
        DEVELOPMENT_MANIFEST_PATH
    ):
        errors.append("development manifest hash mismatch")
    if registration.get("test_manifest_sha256") != sha256_file(TEST_MANIFEST_PATH):
        errors.append("sealed test manifest hash mismatch")
    if registration.get("preflight_sha256") != sha256_file(PREFLIGHT_PATH):
        errors.append("preflight hash mismatch")
    if registration.get("environment") != environment_info():
        errors.append("environment mismatch")
    if registration.get("git", {}).get("head") != repository["head"]:
        errors.append("Git HEAD mismatch")
    if repository["unexpected_status"]:
        errors.append(f"unexpected worktree changes: {repository['unexpected_status']}")
    current_model_hashes = model_file_hashes(model_snapshot())
    if (
        registration.get("model", {}).get("snapshot_file_sha256")
        != current_model_hashes
    ):
        errors.append("model snapshot mismatch")
    return errors


def dynamics_kwargs(steps: int) -> dict[str, int | float]:
    return {
        "temperature": float(SCIENTIFIC_CONSTANTS["energy_temperature"]),
        "steps": steps,
        "initial_step_radius_fraction": float(
            SCIENTIFIC_CONSTANTS["initial_step_radius_fraction"]
        ),
        "armijo_c": float(SCIENTIFIC_CONSTANTS["armijo_c"]),
        "max_halvings": int(SCIENTIFIC_CONSTANTS["max_halvings"]),
        "stationary_tolerance": float(SCIENTIFIC_CONSTANTS["stationary_tolerance"]),
        "energy_tolerance": float(SCIENTIFIC_CONSTANTS["energy_tolerance"]),
        "cap_tolerance": float(SCIENTIFIC_CONSTANTS["cap_tolerance"]),
    }


def targets_from_rows(rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([int(row["target_index"]) for row in rows], dtype=torch.int64)


def intents_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["intent"]) for row in rows]


def arm_metrics(
    predictions: torch.Tensor, targets: torch.Tensor, intents: list[str]
) -> dict[str, Any]:
    prediction_list = predictions.tolist()
    target_list = targets.tolist()
    per_intent = intent_accuracies(prediction_list, target_list, intents)
    return {
        "macro_accuracy": macro_accuracy(per_intent),
        "row_accuracy": row_accuracy(prediction_list, target_list),
        "maximum_intent_share": maximum_intent_prediction_share(prediction_list),
        "per_intent_accuracy": per_intent,
    }


def model_digests(model: AlignedModel) -> dict[str, str]:
    return {
        "global_mean": tensor_digest(model.global_mean),
        "projection": tensor_digest(model.projection),
        "centroids": tensor_digest(model.centroids),
        "within_eigenvalues": tensor_digest(model.within_eigenvalues),
        "discriminant_eigenvalues": tensor_digest(model.discriminant_eigenvalues),
    }


def model_diagnostics(model: AlignedModel) -> dict[str, Any]:
    return {
        "alpha": model.alpha,
        "projection_shape": list(model.projection.shape),
        "within_eigenvalues": model.within_eigenvalues.tolist(),
        "discriminant_eigenvalues": model.discriminant_eigenvalues.tolist(),
        "centroid_norms": torch.linalg.vector_norm(model.centroids, dim=1).tolist(),
    }


def select_models(
    fit_features: torch.Tensor,
    fit_targets: torch.Tensor,
    calibration_features: torch.Tensor,
    calibration_targets: torch.Tensor,
    calibration_intents: list[str],
    class_count: int,
) -> tuple[AlignedModel, Any, dict[str, Any]]:
    controls = fit_centroid_controls(fit_features, fit_targets, class_count)
    calibration_control_scores = centroid_control_scores(calibration_features, controls)
    control_macros: dict[str, float] = {}
    for name, scores in calibration_control_scores.items():
        predictions = torch.argmax(scores, dim=1)
        control_macros[name] = arm_metrics(
            predictions, calibration_targets, calibration_intents
        )["macro_accuracy"]
    selected_control = select_highest_macro(control_macros)

    shrinkage_macros: dict[float, float] = {}
    models: dict[float, AlignedModel] = {}
    for raw_alpha in SCIENTIFIC_CONSTANTS["shrinkage_grid"]:
        alpha = float(raw_alpha)
        model = fit_aligned_model(fit_features, fit_targets, class_count, alpha)
        models[alpha] = model
        states = transform(calibration_features, model)
        predictions = torch.argmax(aligned_scores(states, model.centroids), dim=1)
        shrinkage_macros[alpha] = arm_metrics(
            predictions, calibration_targets, calibration_intents
        )["macro_accuracy"]
    selected_alpha = select_shrinkage(shrinkage_macros)
    selection = {
        "control_macros": control_macros,
        "selected_unaligned_control": selected_control,
        "shrinkage_macros": {
            str(alpha): value for alpha, value in shrinkage_macros.items()
        },
        "selected_alpha": selected_alpha,
        "selection_partition": "calibration",
    }
    return models[selected_alpha], controls, selection


def run_dynamics_batches(
    states: torch.Tensor, model: AlignedModel, radius: float
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], list[dict[str, Any]]]:
    batch_size = int(SCIENTIFIC_CONSTANTS["dynamics_batch_size"])
    one_parts: list[torch.Tensor] = []
    candidate_parts: list[torch.Tensor] = []
    row_diagnostics: list[dict[str, Any]] = []
    maximum_matched_error = 0.0
    maximum_energy_increase = -float("inf")
    maximum_armijo_residual = -float("inf")
    maximum_cap_error = -float("inf")
    maximum_identity_error = 0.0
    matched_prediction_mismatches = 0
    total_stationary = 0
    final_cap_rows = 0
    for start in range(0, states.shape[0], batch_size):
        initial = states[start : start + batch_size]
        identity, _ = constrained_settle(
            initial, model.centroids, radius, **dynamics_kwargs(0)
        )
        maximum_identity_error = max(
            maximum_identity_error, float(torch.max(torch.abs(identity - initial)))
        )
        one, _ = constrained_settle(
            initial,
            model.centroids,
            radius,
            **dynamics_kwargs(int(SCIENTIFIC_CONSTANTS["one_step_steps"])),
        )
        candidate, trace = constrained_settle(
            initial,
            model.centroids,
            radius,
            **dynamics_kwargs(int(SCIENTIFIC_CONSTANTS["steps"])),
        )
        matched = independent_matched_settle(
            initial,
            model.centroids,
            radius,
            **dynamics_kwargs(int(SCIENTIFIC_CONSTANTS["steps"])),
        )
        maximum_matched_error = max(
            maximum_matched_error, float(torch.max(torch.abs(candidate - matched)))
        )
        candidate_prediction = torch.argmax(
            aligned_scores(candidate, model.centroids), dim=1
        )
        matched_prediction = torch.argmax(
            aligned_scores(matched, model.centroids), dim=1
        )
        matched_prediction_mismatches += int(
            torch.sum(candidate_prediction != matched_prediction).item()
        )
        energy_change = trace.energies[:, 1:] - trace.energies[:, :-1]
        maximum_energy_increase = max(
            maximum_energy_increase, float(torch.max(energy_change))
        )
        maximum_armijo_residual = max(
            maximum_armijo_residual, float(torch.max(trace.armijo_residuals))
        )
        cap_error = torch.linalg.vector_norm(candidate - initial, dim=1) - radius
        maximum_cap_error = max(maximum_cap_error, float(torch.max(cap_error)))
        final_cap_rows += int(
            torch.sum(
                torch.linalg.vector_norm(candidate - initial, dim=1)
                >= radius - float(SCIENTIFIC_CONSTANTS["cap_tolerance"])
            ).item()
        )
        total_stationary += int(torch.sum(trace.constrained_stationary).item())
        one_parts.append(one)
        candidate_parts.append(candidate)
        for row_index in range(initial.shape[0]):
            row_diagnostics.append(
                {
                    "initial_energy": float(trace.energies[row_index, 0]),
                    "final_energy": float(trace.energies[row_index, -1]),
                    "energy_trajectory": trace.energies[row_index].tolist(),
                    "step_lengths": trace.step_lengths[row_index].tolist(),
                    "gradient_norms": trace.gradient_norms[row_index].tolist(),
                    "directional_terms": trace.directional_terms[row_index].tolist(),
                    "armijo_residuals": trace.armijo_residuals[row_index].tolist(),
                    "halvings": trace.halvings[row_index].tolist(),
                    "constrained_stationary_steps": int(
                        torch.sum(trace.constrained_stationary[row_index]).item()
                    ),
                }
            )
    diagnostics = {
        "identity_max_abs_error": maximum_identity_error,
        "matched_max_abs_error": maximum_matched_error,
        "matched_prediction_mismatches": matched_prediction_mismatches,
        "maximum_energy_increase": maximum_energy_increase,
        "maximum_armijo_residual": maximum_armijo_residual,
        "maximum_cap_error": maximum_cap_error,
        "total_constrained_stationary_steps": total_stationary,
        "final_cap_fraction": final_cap_rows / float(states.shape[0]),
    }
    return (
        torch.cat(one_parts),
        torch.cat(candidate_parts),
        diagnostics,
        row_diagnostics,
    )


def evaluate_population(
    rows: list[dict[str, Any]],
    features: torch.Tensor,
    model: AlignedModel,
    controls: Any,
    selected_unaligned: str,
    radius: float,
    shuffled_model: AlignedModel | None,
    *,
    run_dynamics: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    targets = targets_from_rows(rows)
    intents = intents_from_rows(rows)
    control_scores = centroid_control_scores(features, controls)
    control_predictions = {
        name: torch.argmax(scores, dim=1) for name, scores in control_scores.items()
    }
    aligned_states = transform(features, model)
    aligned_prediction = torch.argmax(
        aligned_scores(aligned_states, model.centroids), dim=1
    )
    arms: dict[str, Any] = {
        name: arm_metrics(prediction, targets, intents)
        for name, prediction in control_predictions.items()
    }
    arms["aligned_static"] = arm_metrics(aligned_prediction, targets, intents)
    predictions: dict[str, torch.Tensor] = {
        **control_predictions,
        "aligned_static": aligned_prediction,
    }
    dynamics_diagnostics: dict[str, Any] | None = None
    row_dynamics: list[dict[str, Any]] = [{} for _ in rows]
    if run_dynamics:
        one_states, candidate_states, dynamics_diagnostics, row_dynamics = (
            run_dynamics_batches(aligned_states, model, radius)
        )
        one_prediction = torch.argmax(
            aligned_scores(one_states, model.centroids), dim=1
        )
        candidate_prediction = torch.argmax(
            aligned_scores(candidate_states, model.centroids), dim=1
        )
        predictions["one_step"] = one_prediction
        predictions["candidate"] = candidate_prediction
        arms["one_step"] = arm_metrics(one_prediction, targets, intents)
        arms["candidate"] = arm_metrics(candidate_prediction, targets, intents)
    if shuffled_model is not None:
        shuffled_states = transform(features, shuffled_model)
        shuffled_prediction = torch.argmax(
            aligned_scores(shuffled_states, shuffled_model.centroids), dim=1
        )
        predictions["shuffled_label_static"] = shuffled_prediction
        arms["shuffled_label_static"] = arm_metrics(
            shuffled_prediction, targets, intents
        )

    raw_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        record: dict[str, Any] = {
            "row_id": row["row_id"],
            "source_index": row["source_index"],
            "intent": row["intent"],
            "target_index": int(targets[index]),
            "predictions": {
                name: int(prediction[index]) for name, prediction in predictions.items()
            },
        }
        if run_dynamics:
            record["dynamics"] = row_dynamics[index]
        raw_rows.append(record)

    comparisons: dict[str, Any] = {}
    aligned_per_intent = arms["aligned_static"]["per_intent_accuracy"]
    baseline_per_intent = arms[selected_unaligned]["per_intent_accuracy"]
    comparisons["alignment_minus_unaligned"] = paired_intent_bootstrap(
        aligned_per_intent,
        baseline_per_intent,
        resamples=int(SCIENTIFIC_CONSTANTS["bootstrap_resamples"]),
        seed=int(SCIENTIFIC_CONSTANTS["seed"]),
    )
    if run_dynamics:
        comparisons["candidate_minus_aligned_static"] = paired_intent_bootstrap(
            arms["candidate"]["per_intent_accuracy"],
            aligned_per_intent,
            resamples=int(SCIENTIFIC_CONSTANTS["bootstrap_resamples"]),
            seed=int(SCIENTIFIC_CONSTANTS["seed"]),
        )
        comparisons["candidate_minus_one_step"] = paired_intent_bootstrap(
            arms["candidate"]["per_intent_accuracy"],
            arms["one_step"]["per_intent_accuracy"],
            resamples=int(SCIENTIFIC_CONSTANTS["bootstrap_resamples"]),
            seed=int(SCIENTIFIC_CONSTANTS["seed"]),
        )
        comparisons["candidate_vs_static_transitions"] = transition_counts(
            predictions["candidate"].tolist(),
            aligned_prediction.tolist(),
            targets.tolist(),
        )
        comparisons["candidate_vs_one_step_transitions"] = transition_counts(
            predictions["candidate"].tolist(),
            predictions["one_step"].tolist(),
            targets.tolist(),
        )
    return {
        "arms": arms,
        "comparisons": comparisons,
        "dynamics_diagnostics": dynamics_diagnostics,
    }, raw_rows


def shuffled_targets(targets: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(
        int(SCIENTIFIC_CONSTANTS["seed"])
    )
    permutation = torch.randperm(targets.shape[0], generator=generator)
    return targets[permutation]


def build_validity(
    evaluation: dict[str, Any],
    model: AlignedModel,
    radius: float,
    registration_ok: bool,
) -> dict[str, bool]:
    diagnostics = evaluation["dynamics_diagnostics"]
    arms = evaluation["arms"]
    finite_values = [
        radius,
        *[float(value) for value in model.within_eigenvalues.tolist()],
        *[float(value) for value in model.discriminant_eigenvalues.tolist()],
    ]
    return {
        "registration_and_hashes": registration_ok,
        "population": True,
        "calibration_selection": True,
        "finite_and_positive_covariance": bool(
            np.isfinite(finite_values).all()
            and all_finite(evaluation)
            and float(model.within_eigenvalues.min()) > 0.0
        ),
        "energy_and_armijo": bool(
            diagnostics["maximum_energy_increase"]
            <= float(SCIENTIFIC_CONSTANTS["energy_tolerance"])
            and diagnostics["maximum_armijo_residual"]
            <= float(SCIENTIFIC_CONSTANTS["energy_tolerance"])
        ),
        "cap": bool(
            diagnostics["maximum_cap_error"]
            <= float(SCIENTIFIC_CONSTANTS["cap_tolerance"])
        ),
        "identity": diagnostics["identity_max_abs_error"] == 0.0,
        "independent_matched_optimizer": bool(
            diagnostics["matched_max_abs_error"]
            <= float(SCIENTIFIC_CONSTANTS["matched_state_tolerance"])
            and diagnostics["matched_prediction_mismatches"] == 0
        ),
        "shuffled_label_control": bool(
            arms["shuffled_label_static"]["macro_accuracy"]
            < float(SCIENTIFIC_CONSTANTS["shuffled_label_macro_max"])
        ),
        "trust_radius": bool(np.isfinite(radius) and radius > 0.0),
        "test_sealed": True,
    }


def scientific_bars(
    evaluation: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, bool]]:
    comparisons = evaluation["comparisons"]
    arms = evaluation["arms"]
    alignment_comparison = comparisons["alignment_minus_unaligned"]
    alignment = {
        "macro_gain": alignment_comparison["point"]
        >= float(SCIENTIFIC_CONSTANTS["alignment_gain_bar"]),
        "interval_lower": alignment_comparison["low"]
        > float(SCIENTIFIC_CONSTANTS["alignment_interval_lower_bar"]),
        "maximum_intent_share": arms["aligned_static"]["maximum_intent_share"]
        <= float(SCIENTIFIC_CONSTANTS["maximum_intent_share_bar"]),
    }
    static_comparison = comparisons["candidate_minus_aligned_static"]
    one_comparison = comparisons["candidate_minus_one_step"]
    dynamics = {
        "static_macro_gain": static_comparison["point"]
        >= float(SCIENTIFIC_CONSTANTS["dynamics_static_gain_bar"]),
        "static_interval_lower": static_comparison["low"]
        > float(SCIENTIFIC_CONSTANTS["dynamics_static_interval_lower_bar"]),
        "one_step_macro_gain": one_comparison["point"]
        >= float(SCIENTIFIC_CONSTANTS["dynamics_one_step_gain_bar"]),
        "one_step_interval_lower": one_comparison["low"]
        > float(SCIENTIFIC_CONSTANTS["dynamics_one_step_interval_lower_bar"]),
        "maximum_intent_share": arms["candidate"]["maximum_intent_share"]
        <= float(SCIENTIFIC_CONSTANTS["maximum_intent_share_bar"]),
    }
    return alignment, dynamics


def render_development_verdict(result: dict[str, Any]) -> str:
    evaluation = result["evaluation"]
    arms = evaluation["arms"]
    comparisons = evaluation["comparisons"]
    selection = result["selection"]
    return f"""# B77-E1 Development Verdict — Task-Aligned Energy Decomposition

**Date:** 2026-08-10

**Formal verdict:** `{result["verdict"]}`

**Validity:** {"PASS" if all(result["validity"].values()) else "FAIL"}

## Result

The calibration-selected shrinkage was `{selection["selected_alpha"]}` and the
selected unaligned control was `{selection["selected_unaligned_control"]}`.

| Arm | Development macro accuracy |
|---|---:|
| Selected unaligned | {arms[selection["selected_unaligned_control"]]["macro_accuracy"]:.12f} |
| Aligned static | {arms["aligned_static"]["macro_accuracy"]:.12f} |
| One step | {arms["one_step"]["macro_accuracy"]:.12f} |
| Eight-step candidate | {arms["candidate"]["macro_accuracy"]:.12f} |
| Shuffled-label static | {arms["shuffled_label_static"]["macro_accuracy"]:.12f} |

Alignment gain: `{comparisons["alignment_minus_unaligned"]["point"]}` with 95%
intent-cluster interval `[{comparisons["alignment_minus_unaligned"]["low"]},
{comparisons["alignment_minus_unaligned"]["high"]}]`.

Candidate minus aligned static: `{comparisons["candidate_minus_aligned_static"]["point"]}`
with interval `[{comparisons["candidate_minus_aligned_static"]["low"]},
{comparisons["candidate_minus_aligned_static"]["high"]}]`.

Candidate minus one step: `{comparisons["candidate_minus_one_step"]["point"]}`
with interval `[{comparisons["candidate_minus_one_step"]["low"]},
{comparisons["candidate_minus_one_step"]["high"]}]`.

## Gates

```json
{json.dumps({"validity": result["validity"], "alignment_bars": result["alignment_bars"], "dynamics_bars": result["dynamics_bars"]}, indent=2, sort_keys=True)}
```

The official Banking77 test split was not encoded during this development run.
Only a claim named by the formal verdict may be registered for test. This
result does not establish an HLM-specific advantage, explainability,
compliance, deployment readiness, or a reason to scale.
"""


def execute_development() -> None:
    registration = load_json(REGISTRATION_PATH)
    errors = development_registration_errors(registration)
    development = load_json(DEVELOPMENT_MANIFEST_PATH)
    test = load_json(TEST_MANIFEST_PATH)
    errors += manifest_errors(development, test)
    if errors:
        raise RuntimeError("HARNESS_INVALID: " + "; ".join(errors))

    encoder = load_encoder()
    fit_rows = list(development["fit"])
    calibration_rows = list(development["calibration"])
    # The development encoder call is intentionally below completed selection.
    fit_features = encode_texts(encoder, [str(row["text"]) for row in fit_rows])
    calibration_features = encode_texts(
        encoder, [str(row["text"]) for row in calibration_rows]
    )
    fit_targets = targets_from_rows(fit_rows)
    calibration_targets = targets_from_rows(calibration_rows)
    selected_model, controls, selection = select_models(
        fit_features,
        fit_targets,
        calibration_features,
        calibration_targets,
        intents_from_rows(calibration_rows),
        int(SCIENTIFIC_CONSTANTS["intent_count"]),
    )
    selected_alpha = float(selection["selected_alpha"])
    shuffled_model = fit_aligned_model(
        fit_features,
        shuffled_targets(fit_targets),
        int(SCIENTIFIC_CONSTANTS["intent_count"]),
        selected_alpha,
    )
    fit_states = transform(fit_features, selected_model)
    radius = decision_boundary_radius(fit_states, fit_targets, selected_model.centroids)

    development_rows = list(development["development"])
    development_features = encode_texts(
        encoder, [str(row["text"]) for row in development_rows]
    )
    evaluation, raw_rows = evaluate_population(
        development_rows,
        development_features,
        selected_model,
        controls,
        str(selection["selected_unaligned_control"]),
        radius,
        shuffled_model,
        run_dynamics=True,
    )
    validity = build_validity(evaluation, selected_model, radius, True)
    alignment_bars, dynamics_bars = scientific_bars(evaluation)
    verdict = classify_development_verdict(validity, alignment_bars, dynamics_bars)
    write_jsonl(DEV_ROWS_PATH, raw_rows)
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": SCIENTIFIC_CONSTANTS["experiment_id"],
        "phase": "development",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "verdict": verdict,
        "test_embeddings_computed": False,
        "registration_sha256": sha256_file(REGISTRATION_PATH),
        "registration_scientific_digest": registration["scientific_digest"],
        "development_manifest_sha256": sha256_file(DEVELOPMENT_MANIFEST_PATH),
        "test_manifest_sha256": sha256_file(TEST_MANIFEST_PATH),
        "implementation_sha256": implementation_hashes(),
        "model_snapshot_digest": registration["model"]["snapshot_digest"],
        "feature_digests": {
            "fit": tensor_digest(fit_features),
            "calibration": tensor_digest(calibration_features),
            "development": tensor_digest(development_features),
        },
        "fitted_model_digests": model_digests(selected_model),
        "fitted_model_diagnostics": model_diagnostics(selected_model),
        "selection": selection,
        "trust_radius": radius,
        "evaluation": evaluation,
        "validity": validity,
        "alignment_bars": alignment_bars,
        "dynamics_bars": dynamics_bars,
        "raw_rows_sha256": sha256_file(DEV_ROWS_PATH),
    }
    result["result_digest"] = canonical_json_sha256(result)
    write_json(DEV_RESULT_PATH, result)
    DEV_VERDICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEV_VERDICT_PATH.write_text(
        render_development_verdict(result), encoding="utf-8", newline="\n"
    )
    print(f"Development verdict: {verdict}")
    print(f"Result: {DEV_RESULT_PATH}")


def result_digest_matches(result: dict[str, Any]) -> bool:
    payload = dict(result)
    recorded = payload.pop("result_digest", None)
    return canonical_json_sha256(payload) == recorded


def create_test_registration() -> None:
    if TEST_REGISTRATION_PATH.exists():
        raise RuntimeError("test registration already exists")
    result = load_json(DEV_RESULT_PATH)
    if not result_digest_matches(result):
        raise RuntimeError("development result digest mismatch")
    if result.get("raw_rows_sha256") != sha256_file(DEV_ROWS_PATH):
        raise RuntimeError("development raw rows hash mismatch")
    verdict = result.get("verdict")
    if verdict == "PASS_BOTH_TO_TEST":
        claim_scope = "alignment_and_dynamics"
    elif verdict == "PASS_ALIGNMENT_TO_TEST":
        claim_scope = "alignment_only"
    else:
        raise RuntimeError(f"development verdict {verdict!r} does not authorize test")
    if not all(result.get("validity", {}).values()):
        raise RuntimeError("development validity did not pass")
    registration = load_json(REGISTRATION_PATH)
    errors = development_registration_errors(registration)
    if errors:
        raise RuntimeError(
            "development registration no longer verifies: " + "; ".join(errors)
        )
    receipt = {
        "schema_version": 1,
        "experiment_id": SCIENTIFIC_CONSTANTS["experiment_id"],
        "phase": "test",
        "status": "PRE_TEST_INTERNAL_REGISTRATION",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "registered_command": TEST_REGISTERED_COMMAND,
        "claim_scope": claim_scope,
        "development_result_sha256": sha256_file(DEV_RESULT_PATH),
        "development_result_digest": result["result_digest"],
        "development_raw_rows_sha256": sha256_file(DEV_ROWS_PATH),
        "selected_alpha": result["selection"]["selected_alpha"],
        "selected_unaligned_control": result["selection"]["selected_unaligned_control"],
        "trust_radius": result["trust_radius"],
        "implementation_sha256": implementation_hashes(),
        "test_manifest_sha256": sha256_file(TEST_MANIFEST_PATH),
        "environment": environment_info(),
        "git": git_info(),
        "model": registration["model"],
    }
    if receipt["git"]["unexpected_status"]:
        raise RuntimeError(
            f"unexpected worktree changes: {receipt['git']['unexpected_status']}"
        )
    receipt["scientific_digest"] = canonical_json_sha256(receipt)
    write_json(TEST_REGISTRATION_PATH, receipt)
    print(f"Registered test scope {claim_scope}: {TEST_REGISTRATION_PATH}")


def test_registration_errors(receipt: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    payload = dict(receipt)
    recorded = payload.pop("scientific_digest", None)
    if canonical_json_sha256(payload) != recorded:
        errors.append("test registration digest mismatch")
    if receipt.get("registered_command") != TEST_REGISTERED_COMMAND:
        errors.append("test command mismatch")
    if receipt.get("implementation_sha256") != implementation_hashes():
        errors.append("implementation changed before test")
    if receipt.get("test_manifest_sha256") != sha256_file(TEST_MANIFEST_PATH):
        errors.append("test manifest changed")
    if receipt.get("development_result_sha256") != sha256_file(DEV_RESULT_PATH):
        errors.append("development result changed")
    if receipt.get("development_raw_rows_sha256") != sha256_file(DEV_ROWS_PATH):
        errors.append("development raw rows changed")
    if receipt.get("environment") != environment_info():
        errors.append("environment changed")
    repository = git_info()
    if receipt.get("git", {}).get("head") != repository["head"]:
        errors.append("Git HEAD changed")
    if repository["unexpected_status"]:
        errors.append(f"unexpected worktree changes: {repository['unexpected_status']}")
    if receipt.get("model", {}).get("snapshot_file_sha256") != model_file_hashes(
        model_snapshot()
    ):
        errors.append("model snapshot changed")
    return errors


def execute_test() -> None:
    receipt = load_json(TEST_REGISTRATION_PATH)
    errors = test_registration_errors(receipt)
    if errors:
        raise RuntimeError("HARNESS_INVALID: " + "; ".join(errors))
    development = load_json(DEVELOPMENT_MANIFEST_PATH)
    test = load_json(TEST_MANIFEST_PATH)
    manifest_problems = manifest_errors(development, test)
    if manifest_problems:
        raise RuntimeError("HARNESS_INVALID: " + "; ".join(manifest_problems))
    encoder = load_encoder()
    fit_rows = list(development["fit"])
    calibration_rows = list(development["calibration"])
    fit_features = encode_texts(encoder, [str(row["text"]) for row in fit_rows])
    calibration_features = encode_texts(
        encoder, [str(row["text"]) for row in calibration_rows]
    )
    fit_targets = targets_from_rows(fit_rows)
    calibration_targets = targets_from_rows(calibration_rows)
    model, controls, selection = select_models(
        fit_features,
        fit_targets,
        calibration_features,
        calibration_targets,
        intents_from_rows(calibration_rows),
        int(SCIENTIFIC_CONSTANTS["intent_count"]),
    )
    if float(selection["selected_alpha"]) != float(receipt["selected_alpha"]):
        raise RuntimeError("HARNESS_INVALID: selected alpha did not reproduce")
    if selection["selected_unaligned_control"] != receipt["selected_unaligned_control"]:
        raise RuntimeError("HARNESS_INVALID: selected control did not reproduce")
    fit_states = transform(fit_features, model)
    radius = decision_boundary_radius(fit_states, fit_targets, model.centroids)
    if radius != float(receipt["trust_radius"]):
        raise RuntimeError("HARNESS_INVALID: trust radius did not reproduce exactly")
    development_result = load_json(DEV_RESULT_PATH)
    if model_digests(model) != development_result.get("fitted_model_digests"):
        raise RuntimeError("HARNESS_INVALID: fitted model digests did not reproduce")

    official_rows = list(test["official"])
    official_features = encode_texts(
        encoder, [str(row["text"]) for row in official_rows]
    )
    id_to_index = {str(row["row_id"]): index for index, row in enumerate(official_rows)}
    primary_rows = list(test["primary"])
    primary_indices = torch.tensor(
        [id_to_index[str(row["row_id"])] for row in primary_rows], dtype=torch.int64
    )
    primary_features = official_features[primary_indices]
    run_dynamics = receipt["claim_scope"] == "alignment_and_dynamics"
    evaluation, raw_rows = evaluate_population(
        primary_rows,
        primary_features,
        model,
        controls,
        str(selection["selected_unaligned_control"]),
        radius,
        None,
        run_dynamics=run_dynamics,
    )
    alignment_comparison = evaluation["comparisons"]["alignment_minus_unaligned"]
    alignment_bars = {
        "macro_gain": alignment_comparison["point"]
        >= float(SCIENTIFIC_CONSTANTS["alignment_gain_bar"]),
        "interval_lower": alignment_comparison["low"]
        > float(SCIENTIFIC_CONSTANTS["alignment_interval_lower_bar"]),
        "maximum_intent_share": evaluation["arms"]["aligned_static"][
            "maximum_intent_share"
        ]
        <= float(SCIENTIFIC_CONSTANTS["maximum_intent_share_bar"]),
    }
    dynamics_bars: dict[str, bool] | None = None
    if run_dynamics:
        _, dynamics_bars = scientific_bars(evaluation)
    test_validity = {
        "registration_and_hashes": True,
        "population": len(primary_rows)
        == int(SCIENTIFIC_CONSTANTS["primary_test_count"]),
        "finite_and_positive_covariance": bool(
            all_finite(evaluation) and float(model.within_eigenvalues.min()) > 0.0
        ),
        "trust_radius": bool(np.isfinite(radius) and radius > 0.0),
    }
    if run_dynamics:
        diagnostics = evaluation["dynamics_diagnostics"]
        test_validity.update(
            {
                "energy_and_armijo": bool(
                    diagnostics["maximum_energy_increase"]
                    <= float(SCIENTIFIC_CONSTANTS["energy_tolerance"])
                    and diagnostics["maximum_armijo_residual"]
                    <= float(SCIENTIFIC_CONSTANTS["energy_tolerance"])
                ),
                "cap": bool(
                    diagnostics["maximum_cap_error"]
                    <= float(SCIENTIFIC_CONSTANTS["cap_tolerance"])
                ),
                "identity": diagnostics["identity_max_abs_error"] == 0.0,
                "independent_matched_optimizer": bool(
                    diagnostics["matched_max_abs_error"]
                    <= float(SCIENTIFIC_CONSTANTS["matched_state_tolerance"])
                    and diagnostics["matched_prediction_mismatches"] == 0
                ),
            }
        )
    science_pass = all(alignment_bars.values()) and (
        dynamics_bars is None or all(dynamics_bars.values())
    )
    verdict = (
        "HARNESS_INVALID"
        if not all(test_validity.values())
        else ("PASS" if science_pass else "FAIL")
    )
    write_jsonl(TEST_ROWS_PATH, raw_rows)
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": SCIENTIFIC_CONSTANTS["experiment_id"],
        "phase": "test",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "claim_scope": receipt["claim_scope"],
        "verdict": verdict,
        "registration_sha256": sha256_file(TEST_REGISTRATION_PATH),
        "test_manifest_sha256": sha256_file(TEST_MANIFEST_PATH),
        "official_feature_digest": tensor_digest(official_features),
        "primary_evaluation": evaluation,
        "validity": test_validity,
        "alignment_bars": alignment_bars,
        "dynamics_bars": dynamics_bars,
        "raw_rows_sha256": sha256_file(TEST_ROWS_PATH),
    }
    result["result_digest"] = canonical_json_sha256(result)
    write_json(TEST_RESULT_PATH, result)
    TEST_VERDICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_VERDICT_PATH.write_text(
        f"# B77-E1 Test Verdict\n\n**Claim scope:** `{receipt['claim_scope']}`\n\n"
        f"**Formal verdict:** `{verdict}`\n\n"
        "The primary population excludes the seven registered normalized "
        "train overlaps. See the result JSON and raw rows for complete metrics.\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Test verdict: {verdict}")
    print(f"Result: {TEST_RESULT_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--register-development", action="store_true")
    action.add_argument("--run-development", action="store_true")
    action.add_argument("--register-test", action="store_true")
    action.add_argument("--run-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args = parse_args()
    if args.preflight:
        run_preflight()
    elif args.register_development:
        create_development_registration()
    elif args.run_development:
        execute_development()
    elif args.register_test:
        create_test_registration()
    elif args.run_test:
        execute_test()


if __name__ == "__main__":
    main()

"""Preflight, register, and execute the HLM5 I1 intent-basin gate."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    DEV_RAW_ROWS_PATH,
    DEV_REGISTERED_COMMAND,
    DEV_REGISTRATION_PATH,
    DEV_RESULT_PATH,
    DEV_VERDICT_PATH,
    HASHED_IMPLEMENTATION_PATHS,
    MODEL_ID,
    MODEL_REVISION,
    PREFLIGHT_PATH,
    ROOT,
    SCIENTIFIC_CONSTANTS,
    SOURCE_COMMIT,
    SOURCE_DATA_SHA256,
    TEST_MANIFEST_PATH,
    TEST_RAW_ROWS_PATH,
    TEST_REGISTERED_COMMAND,
    TEST_REGISTRATION_PATH,
    TEST_RESULT_PATH,
    TEST_VERDICT_PATH,
)
from dynamics import (
    LineSearchError,
    angular_distance,
    centroid_scores,
    class_degree_scores,
    constrained_settle,
    independent_matched_settle,
    unit,
)
from metrics import (
    average_rank_auc,
    classify_development_verdict,
    classify_test_verdict,
    intent_accuracies,
    macro_accuracy,
    maximum_intent_prediction_share,
    paired_intent_bootstrap,
    select_static_control,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def implementation_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in HASHED_IMPLEMENTATION_PATHS:
        if not path.is_file():
            raise FileNotFoundError(f"required registered file is missing: {path}")
        hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return dict(sorted(hashes.items()))


def environment_info() -> dict[str, str]:
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
    allowed_paths = {
        path.resolve().relative_to(top_level).as_posix()
        for path in (
            PREFLIGHT_PATH,
            DEV_REGISTRATION_PATH,
            DEV_RAW_ROWS_PATH,
            DEV_RESULT_PATH,
            DEV_VERDICT_PATH,
            TEST_REGISTRATION_PATH,
            TEST_RAW_ROWS_PATH,
            TEST_RESULT_PATH,
            TEST_VERDICT_PATH,
        )
    }
    unexpected = [
        line
        for line in status
        if len(line) < 4 or line[3:].replace("\\", "/") not in allowed_paths
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
        raise RuntimeError(f"model snapshot contains no files: {snapshot}")
    return hashes


def load_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        device="cpu",
        local_files_only=True,
    )


def encode_texts(model, texts: list[str]) -> torch.Tensor:
    embeddings = model.encode(
        texts,
        batch_size=int(SCIENTIFIC_CONSTANTS["encoder_batch_size"]),
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return unit(embeddings.detach().cpu().to(torch.float64))


def manifest_errors(manifest: dict[str, Any], phase: str) -> list[str]:
    constants = SCIENTIFIC_CONSTANTS
    errors: list[str] = []
    if manifest.get("phase") != phase:
        errors.append(f"manifest phase {manifest.get('phase')!r} != {phase!r}")
    source = manifest.get("source", {})
    if source.get("commit") != SOURCE_COMMIT:
        errors.append("source commit mismatch")
    if source.get("data_sha256") != SOURCE_DATA_SHA256:
        errors.append("source data hash mismatch")
    intents = manifest.get("intents", [])
    if intents != sorted(intents) or len(intents) != constants["intent_count"]:
        errors.append("intent list is not the registered 150-name lexicographic order")
    if len(set(intents)) != len(intents):
        errors.append("intent names are not unique")

    if phase == "development":
        expected_counts = {
            "intents": constants["intent_count"],
            "train": constants["intent_count"] * constants["train_per_intent"],
            "in_scope": constants["validation_in_scope_count"],
            "oos": constants["validation_oos_count"],
            "excluded_train_overlaps": constants[
                "validation_train_overlap_exclusions"
            ],
        }
        train = manifest.get("train", [])
        if len(train) != expected_counts["train"]:
            errors.append("development train row count mismatch")
        train_counts = {index: 0 for index in range(int(constants["intent_count"]))}
        for row in train:
            target = row.get("target_index")
            if target not in train_counts:
                errors.append("development train target index out of range")
                break
            train_counts[target] += 1
        if any(count != constants["train_per_intent"] for count in train_counts.values()):
            errors.append("development train support is not exactly balanced")
        train_texts = {row.get("text") for row in train}
    else:
        expected_counts = {
            "intents": constants["intent_count"],
            "in_scope": constants["test_in_scope_count"],
            "oos": constants["test_oos_count"],
            "excluded_train_overlaps": constants["test_train_overlap_exclusions"],
        }
        train_texts = set()

    if manifest.get("counts") != expected_counts:
        errors.append(f"manifest counts differ: {manifest.get('counts')} != {expected_counts}")
    in_scope = manifest.get("in_scope", [])
    oos = manifest.get("oos", [])
    exclusions = manifest.get("exclusions", [])
    if len(in_scope) != expected_counts["in_scope"]:
        errors.append("in-scope row count mismatch")
    if len(oos) != expected_counts["oos"]:
        errors.append("OOS row count mismatch")
    if len(exclusions) != expected_counts["excluded_train_overlaps"]:
        errors.append("overlap exclusion count mismatch")
    row_ids = [row.get("row_id") for row in in_scope + oos]
    if len(row_ids) != len(set(row_ids)):
        errors.append("evaluation row IDs are not unique")
    evaluation_texts = [row.get("text") for row in in_scope + oos]
    if len(evaluation_texts) != len(set(evaluation_texts)):
        errors.append("evaluation texts are not unique")
    for row in in_scope:
        target = row.get("target_index")
        if not isinstance(target, int) or not 0 <= target < constants["intent_count"]:
            errors.append("in-scope target index out of range")
            break
    if phase == "development" and train_texts.intersection(evaluation_texts):
        errors.append("development evaluation text overlaps training")
    return errors


def cross_manifest_errors(
    development: dict[str, Any],
    test: dict[str, Any],
) -> list[str]:
    """Validate the sealed test population against development training text."""

    errors: list[str] = []
    if development.get("intents") != test.get("intents"):
        errors.append("development and test intent orders differ")
    if development.get("source") != test.get("source"):
        errors.append("development and test source records differ")
    train_texts = {row.get("text") for row in development.get("train", [])}
    test_texts = {row.get("text") for row in test.get("in_scope", [])}
    if train_texts.intersection(test_texts):
        errors.append("sealed test text overlaps development training")
    return errors


def run_test_scripts() -> list[dict[str, Any]]:
    tests = (
        ROOT / "tests" / "test_dynamics.py",
        ROOT / "tests" / "test_prepare_clinc.py",
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
            raise RuntimeError(f"preflight test failed: {path}")
    return records


def tensor_digest(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(header + contiguous.numpy().tobytes(order="C")).hexdigest()


def result_digest_matches(result: dict[str, Any]) -> bool:
    payload = dict(result)
    recorded = payload.pop("result_digest", None)
    return canonical_json_sha256(payload) == recorded


def run_preflight() -> None:
    if DEV_REGISTRATION_PATH.exists() or TEST_REGISTRATION_PATH.exists():
        raise RuntimeError("a registration exists; preflight is frozen")
    development = load_json(DEVELOPMENT_MANIFEST_PATH)
    test = load_json(TEST_MANIFEST_PATH)
    errors = (
        manifest_errors(development, "development")
        + manifest_errors(test, "test")
        + cross_manifest_errors(development, test)
    )
    if errors:
        raise RuntimeError("; ".join(errors))
    repository = git_info()
    if repository["unexpected_status"]:
        raise RuntimeError(
            f"unexpected worktree changes before preflight: {repository['unexpected_status']}"
        )

    tests = run_test_scripts()
    snapshot = model_snapshot()
    model_hashes = model_file_hashes(snapshot)
    encoder = load_encoder()
    synthetic_embeddings = encode_texts(
        encoder,
        ["synthetic intent alpha", "synthetic intent beta"],
    )
    if tuple(synthetic_embeddings.shape) != (2, 384):
        raise RuntimeError(
            f"unexpected synthetic encoder shape: {tuple(synthetic_embeddings.shape)}"
        )

    prototypes = unit(
        torch.tensor(
            [
                [1.0, 0.2, 0.0, 0.0],
                [0.9, 0.3, 0.0, 0.0],
                [0.0, 0.1, 1.0, 0.2],
                [0.0, 0.0, 0.9, 0.3],
            ],
            dtype=torch.float64,
        )
    )
    queries = unit(
        torch.tensor(
            [[0.8, 0.4, 0.1, 0.0], [0.1, 0.0, 0.8, 0.4]],
            dtype=torch.float64,
        )
    )
    kwargs = {
        "degree": int(SCIENTIFIC_CONSTANTS["degree"]),
        "steps": int(SCIENTIFIC_CONSTANTS["steps"]),
        "initial_step_radius_fraction": float(
            SCIENTIFIC_CONSTANTS["initial_step_radius_fraction"]
        ),
        "armijo_c": float(SCIENTIFIC_CONSTANTS["armijo_c"]),
        "max_halvings": int(SCIENTIFIC_CONSTANTS["max_halvings"]),
        "stationary_tolerance": float(SCIENTIFIC_CONSTANTS["stationary_tolerance"]),
        "energy_tolerance": float(SCIENTIFIC_CONSTANTS["energy_tolerance"]),
        "cap_tolerance": float(SCIENTIFIC_CONSTANTS["cap_tolerance"]),
    }
    candidate, trace = constrained_settle(queries, prototypes, 0.45, **kwargs)
    matched = independent_matched_settle(queries, prototypes, 0.45, **kwargs)
    maximum_error = float(torch.abs(candidate - matched).max())
    maximum_energy_increase = float((trace.energies[:, 1:] - trace.energies[:, :-1]).max())
    maximum_cap_error = float((angular_distance(queries, candidate) - 0.45).max())
    maximum_armijo_residual = float(trace.armijo_residuals.max())
    if maximum_error > float(SCIENTIFIC_CONSTANTS["matched_state_tolerance"]):
        raise RuntimeError("synthetic matched implementation disagrees")
    if maximum_energy_increase > float(SCIENTIFIC_CONSTANTS["energy_tolerance"]):
        raise RuntimeError("synthetic energy certificate failed")
    if maximum_cap_error > float(SCIENTIFIC_CONSTANTS["cap_tolerance"]):
        raise RuntimeError("synthetic cap certificate failed")
    if maximum_armijo_residual > float(SCIENTIFIC_CONSTANTS["energy_tolerance"]):
        raise RuntimeError("synthetic Armijo certificate failed")

    receipt = {
        "schema_version": 1,
        "experiment_id": SCIENTIFIC_CONSTANTS["experiment_id"],
        "status": "SYNTHETIC_PREFLIGHT_ONLY",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "real_clinc_embeddings_computed": False,
        "test_embeddings_computed": False,
        "development_manifest_sha256": sha256_file(DEVELOPMENT_MANIFEST_PATH),
        "test_manifest_sha256": sha256_file(TEST_MANIFEST_PATH),
        "implementation_sha256": implementation_hashes(),
        "git": repository,
        "environment": environment_info(),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "snapshot_file_sha256": model_hashes,
            "snapshot_digest": canonical_json_sha256(model_hashes),
            "synthetic_shape": list(synthetic_embeddings.shape),
        },
        "synthetic_dynamics": {
            "matched_max_abs_error": maximum_error,
            "maximum_energy_increase": maximum_energy_increase,
            "maximum_cap_error": maximum_cap_error,
            "maximum_armijo_residual": maximum_armijo_residual,
        },
        "tests": tests,
    }
    write_json(PREFLIGHT_PATH, receipt)
    print(f"Synthetic-only preflight PASS: {PREFLIGHT_PATH}")


def create_development_registration() -> None:
    if DEV_REGISTRATION_PATH.exists():
        raise RuntimeError("development registration already exists")
    if not PREFLIGHT_PATH.is_file():
        raise RuntimeError("synthetic preflight receipt is missing")
    preflight = load_json(PREFLIGHT_PATH)
    current_hashes = implementation_hashes()
    current_environment = environment_info()
    current_model_hashes = model_file_hashes(model_snapshot())
    current_git = git_info()
    if preflight.get("real_clinc_embeddings_computed") is not False:
        raise RuntimeError("preflight real-outcome access flag is not false")
    if preflight.get("test_embeddings_computed") is not False:
        raise RuntimeError("preflight test-access flag is not false")
    if preflight.get("implementation_sha256") != current_hashes:
        raise RuntimeError("implementation changed after preflight")
    if preflight.get("environment") != current_environment:
        raise RuntimeError("environment changed after preflight")
    if preflight.get("git", {}).get("head") != current_git["head"]:
        raise RuntimeError("git HEAD changed after preflight")
    if current_git["unexpected_status"]:
        raise RuntimeError(
            f"unexpected worktree changes before registration: {current_git['unexpected_status']}"
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
        "environment": current_environment,
        "git": current_git,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "snapshot_file_sha256": current_model_hashes,
            "snapshot_digest": canonical_json_sha256(current_model_hashes),
        },
    }
    registration["scientific_digest"] = canonical_json_sha256(registration)
    write_json(DEV_REGISTRATION_PATH, registration)
    print(f"Registered before CLINC encoder outcomes: {DEV_REGISTRATION_PATH}")
    print(f"Scientific digest: {registration['scientific_digest']}")


def registration_digest_matches(registration: dict[str, Any]) -> bool:
    payload = dict(registration)
    recorded = payload.pop("scientific_digest", None)
    return canonical_json_sha256(payload) == recorded


def development_registration_errors(registration: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not registration_digest_matches(registration):
        errors.append("development registration scientific digest mismatch")
    if registration.get("phase") != "development":
        errors.append("development registration phase mismatch")
    if registration.get("status") != "PRE_OUTCOME_INTERNAL_REGISTRATION":
        errors.append("development registration status mismatch")
    if registration.get("registered_command") != DEV_REGISTERED_COMMAND:
        errors.append("development command mismatch")
    if registration.get("scientific_constants") != SCIENTIFIC_CONSTANTS:
        errors.append("scientific constants mismatch")
    if registration.get("implementation_sha256") != implementation_hashes():
        errors.append("implementation hash mismatch")
    if registration.get("development_manifest_sha256") != sha256_file(
        DEVELOPMENT_MANIFEST_PATH
    ):
        errors.append("development manifest hash mismatch")
    if registration.get("test_manifest_sha256") != sha256_file(TEST_MANIFEST_PATH):
        errors.append("test manifest hash mismatch")
    if registration.get("preflight_sha256") != sha256_file(PREFLIGHT_PATH):
        errors.append("preflight hash mismatch")
    if registration.get("environment") != environment_info():
        errors.append("environment mismatch")
    current_git = git_info()
    if registration.get("git", {}).get("head") != current_git["head"]:
        errors.append("git HEAD mismatch")
    if current_git["unexpected_status"]:
        errors.append(f"unexpected worktree changes: {current_git['unexpected_status']}")
    current_model_hashes = model_file_hashes(model_snapshot())
    registered_model = registration.get("model", {})
    if registered_model.get("id") != MODEL_ID:
        errors.append("model ID mismatch")
    if registered_model.get("revision") != MODEL_REVISION:
        errors.append("model revision mismatch")
    if registered_model.get("snapshot_file_sha256") != current_model_hashes:
        errors.append("model file hash mismatch")
    if registered_model.get("snapshot_digest") != canonical_json_sha256(
        current_model_hashes
    ):
        errors.append("model snapshot digest mismatch")
    return errors


def build_memory(encoder, development: dict[str, Any]) -> dict[str, Any]:
    train_rows = development["train"]
    raw_train = encode_texts(encoder, [row["text"] for row in train_rows])
    class_count = int(SCIENTIFIC_CONSTANTS["intent_count"])
    blocks = int(SCIENTIFIC_CONSTANTS["prototype_blocks_per_intent"])
    block_size = int(SCIENTIFIC_CONSTANTS["prototype_block_size"])
    grouped: list[list[int]] = [[] for _ in range(class_count)]
    for row_index, row in enumerate(train_rows):
        grouped[int(row["target_index"])].append(row_index)
    raw_prototypes = torch.empty(
        (class_count, blocks, raw_train.shape[1]),
        dtype=torch.float64,
    )
    for class_index, indices in enumerate(grouped):
        if len(indices) != blocks * block_size:
            raise RuntimeError("prototype source support count mismatch")
        for block in range(blocks):
            block_indices = indices[block * block_size : (block + 1) * block_size]
            raw_prototypes[class_index, block] = unit(
                raw_train[block_indices].mean(dim=0)
            )

    raw_centroids = unit(raw_prototypes.mean(dim=1))
    common_mean = raw_prototypes.reshape(-1, raw_prototypes.shape[-1]).mean(dim=0)
    centered_prototypes = unit(raw_prototypes - common_mean)
    centered_centroids = unit(centered_prototypes.mean(dim=1))
    centered_train = unit(raw_train - common_mean)
    train_targets = torch.tensor(
        [int(row["target_index"]) for row in train_rows],
        dtype=torch.int64,
    )
    train_angles = angular_distance(
        centered_train,
        centered_centroids[train_targets],
    )
    radius = float(
        torch.quantile(train_angles, 0.5, interpolation="linear")
    )
    flat = centered_prototypes.reshape(-1, centered_prototypes.shape[-1])
    return {
        "raw_train": raw_train,
        "raw_prototypes": raw_prototypes,
        "raw_centroids": raw_centroids,
        "common_mean": common_mean,
        "centered_prototypes": centered_prototypes,
        "centered_centroids": centered_centroids,
        "flat_prototypes": flat,
        "trust_radius": radius,
        "prototype_digest": tensor_digest(centered_prototypes),
        "common_mean_digest": tensor_digest(common_mean),
        "maximum_unit_error": max(
            float(torch.abs(raw_train.norm(dim=1) - 1.0).max()),
            float(torch.abs(flat.norm(dim=1) - 1.0).max()),
            float(torch.abs(centered_centroids.norm(dim=1) - 1.0).max()),
        ),
        "all_finite": bool(
            torch.isfinite(raw_train).all()
            and torch.isfinite(centered_prototypes).all()
            and torch.isfinite(centered_centroids).all()
        ),
    }


def dynamics_kwargs(steps: int) -> dict[str, int | float]:
    return {
        "degree": int(SCIENTIFIC_CONSTANTS["degree"]),
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


def evaluate_queries(
    *,
    kind: str,
    raw_queries: torch.Tensor,
    centered_queries: torch.Tensor,
    metadata: list[dict[str, Any]],
    memory: dict[str, Any],
    raw_handle,
) -> dict[str, Any]:
    batch_size = int(SCIENTIFIC_CONSTANTS["dynamics_batch_size"])
    control_slots = {
        "raw_centroid": [],
        "centered_centroid": [],
        "centered_degree5": [],
    }
    control_confidences = {name: [] for name in control_slots}
    one_step_slots: list[int] = []
    one_step_confidences: list[float] = []
    candidate_slots: list[int] = []
    candidate_confidences: list[float] = []
    maximum_matched_error = 0.0
    maximum_energy_increase = -float("inf")
    maximum_armijo_residual = -float("inf")
    maximum_unit_error = 0.0
    maximum_cap_error = -float("inf")
    all_finite = True
    trajectory_digests: list[str] = []

    for start in range(0, raw_queries.shape[0], batch_size):
        end = min(start + batch_size, raw_queries.shape[0])
        raw_batch = raw_queries[start:end]
        batch = centered_queries[start:end]

        raw_scores = centroid_scores(raw_batch, memory["raw_centroids"])
        centered_centroid = centroid_scores(batch, memory["centered_centroids"])
        centered_degree = class_degree_scores(
            batch,
            memory["centered_prototypes"],
            int(SCIENTIFIC_CONSTANTS["degree"]),
        )
        static_scores = {
            "raw_centroid": raw_scores,
            "centered_centroid": centered_centroid,
            "centered_degree5": centered_degree,
        }

        one_state, _ = constrained_settle(
            batch,
            memory["flat_prototypes"],
            memory["trust_radius"],
            **dynamics_kwargs(int(SCIENTIFIC_CONSTANTS["one_step_steps"])),
        )
        candidate_state, trace = constrained_settle(
            batch,
            memory["flat_prototypes"],
            memory["trust_radius"],
            **dynamics_kwargs(int(SCIENTIFIC_CONSTANTS["steps"])),
        )
        matched_state = independent_matched_settle(
            batch,
            memory["flat_prototypes"],
            memory["trust_radius"],
            **dynamics_kwargs(int(SCIENTIFIC_CONSTANTS["steps"])),
        )
        one_scores = class_degree_scores(
            one_state,
            memory["centered_prototypes"],
            int(SCIENTIFIC_CONSTANTS["degree"]),
        )
        candidate_scores = class_degree_scores(
            candidate_state,
            memory["centered_prototypes"],
            int(SCIENTIFIC_CONSTANTS["degree"]),
        )
        matched_scores = class_degree_scores(
            matched_state,
            memory["centered_prototypes"],
            int(SCIENTIFIC_CONSTANTS["degree"]),
        )
        one_slot = one_scores.argmax(dim=1)
        candidate_slot = candidate_scores.argmax(dim=1)
        matched_slot = matched_scores.argmax(dim=1)
        if not torch.equal(candidate_slot, matched_slot):
            raise RuntimeError("candidate and independent settle predictions differ")

        row_errors = torch.abs(candidate_state - matched_state).amax(dim=1)
        energy_deltas = trace.energies[:, 1:] - trace.energies[:, :-1]
        cap_error = angular_distance(batch, candidate_state) - memory["trust_radius"]
        maximum_matched_error = max(maximum_matched_error, float(row_errors.max()))
        maximum_energy_increase = max(maximum_energy_increase, float(energy_deltas.max()))
        maximum_armijo_residual = max(
            maximum_armijo_residual,
            float(trace.armijo_residuals.max()),
        )
        maximum_unit_error = max(
            maximum_unit_error,
            float(torch.abs(candidate_state.norm(dim=1) - 1.0).max()),
        )
        maximum_cap_error = max(maximum_cap_error, float(cap_error.max()))
        all_finite = all_finite and bool(
            torch.isfinite(candidate_state).all()
            and torch.isfinite(trace.energies).all()
        )
        trajectory_digests.append(
            canonical_json_sha256(
                {
                    "final": tensor_digest(candidate_state),
                    "energies": tensor_digest(trace.energies),
                    "steps": tensor_digest(trace.step_angles),
                }
            )
        )

        static_predictions: dict[str, torch.Tensor] = {}
        static_confidence: dict[str, torch.Tensor] = {}
        for name, scores in static_scores.items():
            static_predictions[name] = scores.argmax(dim=1)
            static_confidence[name] = scores.max(dim=1).values
            control_slots[name].extend(static_predictions[name].tolist())
            control_confidences[name].extend(static_confidence[name].tolist())

        one_step_slots.extend(one_slot.tolist())
        one_step_confidences.extend(one_scores.max(dim=1).values.tolist())
        candidate_slots.extend(candidate_slot.tolist())
        candidate_confidences.extend(candidate_scores.max(dim=1).values.tolist())

        for local_index, meta in enumerate(metadata[start:end]):
            record: dict[str, Any] = {
                "kind": kind,
                "row_id": meta["row_id"],
                "raw_centroid_slot": int(static_predictions["raw_centroid"][local_index]),
                "raw_centroid_confidence": float(
                    static_confidence["raw_centroid"][local_index]
                ),
                "centered_centroid_slot": int(
                    static_predictions["centered_centroid"][local_index]
                ),
                "centered_centroid_confidence": float(
                    static_confidence["centered_centroid"][local_index]
                ),
                "centered_degree5_slot": int(
                    static_predictions["centered_degree5"][local_index]
                ),
                "centered_degree5_confidence": float(
                    static_confidence["centered_degree5"][local_index]
                ),
                "one_step_slot": int(one_slot[local_index]),
                "one_step_confidence": float(one_scores[local_index].max()),
                "candidate_slot": int(candidate_slot[local_index]),
                "candidate_confidence": float(candidate_scores[local_index].max()),
                "matched_slot": int(matched_slot[local_index]),
                "matched_max_abs_error": float(row_errors[local_index]),
                "energies": trace.energies[local_index].tolist(),
                "step_angles": trace.step_angles[local_index].tolist(),
                "gradient_norms": trace.gradient_norms[local_index].tolist(),
                "directional_terms": trace.directional_terms[local_index].tolist(),
                "armijo_residuals": trace.armijo_residuals[local_index].tolist(),
                "halvings": trace.halvings[local_index].tolist(),
                "constrained_stationary": trace.constrained_stationary[
                    local_index
                ].tolist(),
            }
            if kind == "in_scope":
                record["intent"] = meta["intent"]
                record["target_index"] = int(meta["target_index"])
            raw_handle.write(json.dumps(record, sort_keys=True) + "\n")

    return {
        "control_slots": control_slots,
        "control_confidences": control_confidences,
        "one_step_slots": one_step_slots,
        "one_step_confidences": one_step_confidences,
        "candidate_slots": candidate_slots,
        "candidate_confidences": candidate_confidences,
        "maximum_matched_error": maximum_matched_error,
        "maximum_energy_increase": maximum_energy_increase,
        "maximum_armijo_residual": maximum_armijo_residual,
        "maximum_unit_error": maximum_unit_error,
        "maximum_cap_error": maximum_cap_error,
        "all_finite": all_finite,
        "trajectory_digest": canonical_json_sha256(trajectory_digests),
    }


def aggregate_result(
    *,
    phase: str,
    manifest: dict[str, Any],
    memory: dict[str, Any],
    raw_queries: torch.Tensor,
    centered_queries: torch.Tensor,
    raw_oos: torch.Tensor,
    centered_oos: torch.Tensor,
    supported: dict[str, Any],
    oos: dict[str, Any],
    registration: dict[str, Any],
    registration_path: Path,
    raw_path: Path,
    selected_static_override: str | None,
) -> dict[str, Any]:
    metadata = manifest["in_scope"]
    targets = [int(row["target_index"]) for row in metadata]
    intents = [str(row["intent"]) for row in metadata]

    static_per_intent: dict[str, dict[str, float]] = {}
    static_macros: dict[str, float] = {}
    for name, predictions in supported["control_slots"].items():
        per_intent = intent_accuracies(predictions, targets, intents)
        static_per_intent[name] = per_intent
        static_macros[name] = macro_accuracy(per_intent)
    selected_static = selected_static_override or select_static_control(static_macros)
    if selected_static not in static_macros:
        raise RuntimeError(f"registered static control {selected_static!r} is unavailable")

    candidate_per_intent = intent_accuracies(
        supported["candidate_slots"], targets, intents
    )
    one_step_per_intent = intent_accuracies(
        supported["one_step_slots"], targets, intents
    )
    selected_per_intent = static_per_intent[selected_static]
    static_interval = paired_intent_bootstrap(
        candidate_per_intent,
        selected_per_intent,
        resamples=int(SCIENTIFIC_CONSTANTS["bootstrap_resamples"]),
        seed=int(SCIENTIFIC_CONSTANTS["seed"]),
    )
    one_step_interval = paired_intent_bootstrap(
        candidate_per_intent,
        one_step_per_intent,
        resamples=int(SCIENTIFIC_CONSTANTS["bootstrap_resamples"]),
        seed=int(SCIENTIFIC_CONSTANTS["seed"]),
    )
    selected_auc = average_rank_auc(
        supported["control_confidences"][selected_static],
        oos["control_confidences"][selected_static],
    )
    candidate_auc = average_rank_auc(
        supported["candidate_confidences"],
        oos["candidate_confidences"],
    )
    maximum_share = maximum_intent_prediction_share(supported["candidate_slots"])

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(SCIENTIFIC_CONSTANTS["seed"]))
    permutation = torch.randperm(
        int(SCIENTIFIC_CONSTANTS["intent_count"]), generator=generator
    ).tolist()
    shuffled_predictions = [
        permutation[prediction] for prediction in supported["candidate_slots"]
    ]
    shuffled_per_intent = intent_accuracies(shuffled_predictions, targets, intents)
    shuffled_macro = macro_accuracy(shuffled_per_intent)

    identity_state, _ = constrained_settle(
        centered_queries,
        memory["flat_prototypes"],
        memory["trust_radius"],
        **dynamics_kwargs(0),
    )
    identity_scores = class_degree_scores(
        identity_state,
        memory["centered_prototypes"],
        int(SCIENTIFIC_CONSTANTS["degree"]),
    )
    identity_bit_exact = torch.equal(identity_state, centered_queries)
    identity_static_exact = torch.equal(
        identity_scores.argmax(dim=1),
        torch.tensor(supported["control_slots"]["centered_degree5"]),
    )

    unit_error = max(
        memory["maximum_unit_error"],
        float(torch.abs(raw_queries.norm(dim=1) - 1.0).max()),
        float(torch.abs(centered_queries.norm(dim=1) - 1.0).max()),
        float(torch.abs(raw_oos.norm(dim=1) - 1.0).max()),
        float(torch.abs(centered_oos.norm(dim=1) - 1.0).max()),
        supported["maximum_unit_error"],
        oos["maximum_unit_error"],
    )
    matched_error = max(
        supported["maximum_matched_error"],
        oos["maximum_matched_error"],
    )
    maximum_energy_increase = max(
        supported["maximum_energy_increase"],
        oos["maximum_energy_increase"],
    )
    maximum_armijo_residual = max(
        supported["maximum_armijo_residual"],
        oos["maximum_armijo_residual"],
    )
    maximum_cap_error = max(
        supported["maximum_cap_error"],
        oos["maximum_cap_error"],
    )
    finite = bool(
        memory["all_finite"]
        and supported["all_finite"]
        and oos["all_finite"]
        and torch.isfinite(raw_queries).all()
        and torch.isfinite(raw_oos).all()
    )
    development_splits = {"train"}
    development_splits.update(row["split"] for row in manifest.get("train", []))
    development_splits.update(row["split"] for row in metadata)
    development_splits.update(row["split"] for row in manifest["oos"])
    no_test_access = phase != "development" or not bool(
        development_splits.intersection({"test", "oos_test"})
    )

    validity = {
        "registration_verified": True,
        "population_exact": not manifest_errors(manifest, phase),
        "finite_and_unit": finite
        and unit_error <= float(SCIENTIFIC_CONSTANTS["unit_tolerance"]),
        "energy_nonincreasing": maximum_energy_increase
        <= float(SCIENTIFIC_CONSTANTS["energy_tolerance"]),
        "armijo_bound": maximum_armijo_residual
        <= float(SCIENTIFIC_CONSTANTS["energy_tolerance"]),
        "inside_training_cap": maximum_cap_error
        <= float(SCIENTIFIC_CONSTANTS["cap_tolerance"]),
        "identity_bit_exact": identity_bit_exact and identity_static_exact,
        "independent_matched": matched_error
        <= float(SCIENTIFIC_CONSTANTS["matched_state_tolerance"]),
        "shuffled_label_null": shuffled_macro
        < float(SCIENTIFIC_CONSTANTS["valid_shuffled_macro_max"]),
        "trust_radius_valid": 0.0 < memory["trust_radius"] < float(torch.pi / 2),
        "development_did_not_encode_test": no_test_access,
    }
    scientific = {
        "static_gain": static_interval["point"]
        >= float(SCIENTIFIC_CONSTANTS["pass_static_gain"]),
        "static_bootstrap_lower": static_interval["low"]
        > float(SCIENTIFIC_CONSTANTS["pass_static_bootstrap_lower"]),
        "one_step_gain": one_step_interval["point"]
        >= float(SCIENTIFIC_CONSTANTS["pass_one_step_gain"]),
        "one_step_bootstrap_lower": one_step_interval["low"]
        > float(SCIENTIFIC_CONSTANTS["pass_one_step_bootstrap_lower"]),
        "auc_noninferiority": candidate_auc - selected_auc
        >= float(SCIENTIFIC_CONSTANTS["pass_auc_noninferiority"]),
        "no_intent_collapse": maximum_share
        <= float(SCIENTIFIC_CONSTANTS["pass_max_intent_share"]),
    }
    verdict = (
        classify_development_verdict(validity, scientific)
        if phase == "development"
        else classify_test_verdict(validity, scientific)
    )

    singular_values = torch.linalg.svdvals(memory["flat_prototypes"])
    rank_tolerance = (
        max(memory["flat_prototypes"].shape)
        * torch.finfo(torch.float64).eps
        * singular_values[0]
    )
    nonzero = singular_values[singular_values > rank_tolerance]
    result = {
        "schema_version": 1,
        "experiment_id": SCIENTIFIC_CONSTANTS["experiment_id"],
        "phase": phase,
        "formal_verdict": verdict,
        "registration_scientific_digest": registration["scientific_digest"],
        "registration_sha256": sha256_file(registration_path),
        "raw_rows_sha256": sha256_file(raw_path),
        "selected_static_control": selected_static,
        "test_embeddings_computed": phase == "test",
        "encoded_splits": sorted(development_splits),
        "counts": manifest["counts"],
        "validity": validity,
        "scientific_bars": scientific,
        "metrics": {
            "static_macro_accuracy": static_macros,
            "selected_static_macro_accuracy": static_macros[selected_static],
            "one_step_macro_accuracy": macro_accuracy(one_step_per_intent),
            "candidate_macro_accuracy": macro_accuracy(candidate_per_intent),
            "candidate_minus_selected_static": static_interval,
            "candidate_minus_one_step": one_step_interval,
            "selected_static_supported_oos_auc": selected_auc,
            "candidate_supported_oos_auc": candidate_auc,
            "candidate_minus_static_auc": candidate_auc - selected_auc,
            "maximum_candidate_intent_share": maximum_share,
            "shuffled_label_macro_accuracy": shuffled_macro,
            "trust_radius_radians": memory["trust_radius"],
            "unit_error": unit_error,
            "matched_max_abs_error": matched_error,
            "maximum_energy_increase": maximum_energy_increase,
            "maximum_armijo_residual": maximum_armijo_residual,
            "maximum_cap_error": maximum_cap_error,
        },
        "per_intent": {
            intent: {
                **{
                    f"{name}_accuracy": static_per_intent[name][intent]
                    for name in sorted(static_per_intent)
                },
                "one_step_accuracy": one_step_per_intent[intent],
                "candidate_accuracy": candidate_per_intent[intent],
                "candidate_minus_selected": (
                    candidate_per_intent[intent] - selected_per_intent[intent]
                ),
                "candidate_minus_one_step": (
                    candidate_per_intent[intent] - one_step_per_intent[intent]
                ),
                "shuffled_accuracy": shuffled_per_intent[intent],
            }
            for intent in sorted(candidate_per_intent)
        },
        "input_digests": {
            "raw_queries": tensor_digest(raw_queries),
            "centered_queries": tensor_digest(centered_queries),
            "raw_oos": tensor_digest(raw_oos),
            "centered_oos": tensor_digest(centered_oos),
            "prototypes": memory["prototype_digest"],
            "common_mean": memory["common_mean_digest"],
            "supported_trajectories": supported["trajectory_digest"],
            "oos_trajectories": oos["trajectory_digest"],
            "shuffled_label_permutation": canonical_json_sha256(permutation),
        },
        "prototype_geometry": {
            "coherence": float(
                torch.abs(
                    memory["flat_prototypes"] @ memory["flat_prototypes"].T
                    - torch.eye(
                        memory["flat_prototypes"].shape[0], dtype=torch.float64
                    )
                ).max()
            ),
            "singular_values": singular_values.tolist(),
            "numerical_rank": int(nonzero.numel()),
            "rank_tolerance": float(rank_tolerance),
            "condition_nonzero_spectrum": (
                float(nonzero[0] / nonzero[-1]) if nonzero.numel() else float("inf")
            ),
        },
        "claim_boundary": (
            "This assay measures fixed class-balanced constrained dynamics on frozen "
            "CLINC sentence embeddings. It does not establish semantic understanding, "
            "language-model improvement, production readiness, compliance, scaling "
            "authorization, or an HLM-specific implementation advantage."
        ),
    }
    result["result_digest"] = canonical_json_sha256(result)
    return result


def execute_phase(phase: str) -> None:
    if phase == "development":
        registration_path = DEV_REGISTRATION_PATH
        raw_path = DEV_RAW_ROWS_PATH
        result_path = DEV_RESULT_PATH
        manifest_path = DEVELOPMENT_MANIFEST_PATH
        registration = load_json(registration_path)
        errors = development_registration_errors(registration)
        selected_override = None
    else:
        registration_path = TEST_REGISTRATION_PATH
        raw_path = TEST_RAW_ROWS_PATH
        result_path = TEST_RESULT_PATH
        manifest_path = TEST_MANIFEST_PATH
        registration = load_json(registration_path)
        errors = test_registration_errors(registration)
        selected_override = str(registration["development_selection"]["static_control"])
    if errors:
        raise RuntimeError("registration verification failed: " + "; ".join(errors))
    if result_path.exists() or raw_path.exists():
        raise RuntimeError("registered output already exists; refusing to overwrite")

    manifest = load_json(manifest_path)
    manifest_issue = manifest_errors(manifest, phase)
    if manifest_issue:
        raise RuntimeError("manifest verification failed: " + "; ".join(manifest_issue))
    development = load_json(DEVELOPMENT_MANIFEST_PATH)
    if phase == "test":
        cross_errors = cross_manifest_errors(development, manifest)
        if cross_errors:
            raise RuntimeError("cross-manifest verification failed: " + "; ".join(cross_errors))

    torch.set_num_threads(1)
    torch.manual_seed(int(SCIENTIFIC_CONSTANTS["seed"]))
    torch.use_deterministic_algorithms(True)
    encoder = load_encoder()
    memory = build_memory(encoder, development)
    if phase == "test":
        selection = registration["development_selection"]
        if memory["prototype_digest"] != selection["prototype_digest"]:
            raise RuntimeError("recomputed prototype digest differs from test receipt")
        if memory["trust_radius"] != selection["trust_radius_radians"]:
            raise RuntimeError("recomputed trust radius differs from test receipt")

    raw_queries = encode_texts(encoder, [row["text"] for row in manifest["in_scope"]])
    raw_oos = encode_texts(encoder, [row["text"] for row in manifest["oos"]])
    centered_queries = unit(raw_queries - memory["common_mean"])
    centered_oos = unit(raw_oos - memory["common_mean"])

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with raw_path.open("x", encoding="utf-8", newline="\n") as raw_handle:
            supported = evaluate_queries(
                kind="in_scope",
                raw_queries=raw_queries,
                centered_queries=centered_queries,
                metadata=manifest["in_scope"],
                memory=memory,
                raw_handle=raw_handle,
            )
            oos = evaluate_queries(
                kind="oos",
                raw_queries=raw_oos,
                centered_queries=centered_oos,
                metadata=manifest["oos"],
                memory=memory,
                raw_handle=raw_handle,
            )
    except LineSearchError as exc:
        invalid = {
            "schema_version": 1,
            "experiment_id": SCIENTIFIC_CONSTANTS["experiment_id"],
            "phase": phase,
            "formal_verdict": "HARNESS_INVALID",
            "reason": str(exc),
            "registration_scientific_digest": registration["scientific_digest"],
        }
        write_json(result_path, invalid)
        raise

    result = aggregate_result(
        phase=phase,
        manifest=manifest,
        memory=memory,
        raw_queries=raw_queries,
        centered_queries=centered_queries,
        raw_oos=raw_oos,
        centered_oos=centered_oos,
        supported=supported,
        oos=oos,
        registration=registration,
        registration_path=registration_path,
        raw_path=raw_path,
        selected_static_override=selected_override,
    )
    write_json(result_path, result)
    print(f"{result['formal_verdict']}: {result_path}")
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


def create_test_registration() -> None:
    if TEST_REGISTRATION_PATH.exists():
        raise RuntimeError("test registration already exists")
    development_result = load_json(DEV_RESULT_PATH)
    if not result_digest_matches(development_result):
        raise RuntimeError("development result digest mismatch")
    if development_result.get("formal_verdict") != "PASS_TO_TEST":
        raise RuntimeError("development gate did not authorize test registration")
    if sha256_file(DEV_RAW_ROWS_PATH) != development_result.get("raw_rows_sha256"):
        raise RuntimeError("development raw rows hash mismatch")
    current_git = git_info()
    if current_git["unexpected_status"]:
        raise RuntimeError(
            f"unexpected worktree changes before test registration: {current_git['unexpected_status']}"
        )
    development_registration = load_json(DEV_REGISTRATION_PATH)
    errors = development_registration_errors(development_registration)
    if errors:
        raise RuntimeError("development lineage is invalid: " + "; ".join(errors))

    registration = {
        "schema_version": 1,
        "experiment_id": SCIENTIFIC_CONSTANTS["experiment_id"],
        "phase": "test",
        "status": "PRE_TEST_OUTCOME_INTERNAL_REGISTRATION",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "registered_command": TEST_REGISTERED_COMMAND,
        "scientific_constants": SCIENTIFIC_CONSTANTS,
        "implementation_sha256": implementation_hashes(),
        "development_manifest_sha256": sha256_file(DEVELOPMENT_MANIFEST_PATH),
        "test_manifest_sha256": sha256_file(TEST_MANIFEST_PATH),
        "preflight_sha256": sha256_file(PREFLIGHT_PATH),
        "development_registration_sha256": sha256_file(DEV_REGISTRATION_PATH),
        "development_result_sha256": sha256_file(DEV_RESULT_PATH),
        "development_result_digest": development_result["result_digest"],
        "development_raw_rows_sha256": development_result["raw_rows_sha256"],
        "development_selection": {
            "static_control": development_result["selected_static_control"],
            "trust_radius_radians": development_result["metrics"][
                "trust_radius_radians"
            ],
            "prototype_digest": development_result["input_digests"]["prototypes"],
        },
        "environment": environment_info(),
        "git": current_git,
        "model": development_registration["model"],
        "test_embeddings_computed": False,
    }
    registration["scientific_digest"] = canonical_json_sha256(registration)
    write_json(TEST_REGISTRATION_PATH, registration)
    print(f"Registered before test encoder outcomes: {TEST_REGISTRATION_PATH}")


def test_registration_errors(registration: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not registration_digest_matches(registration):
        errors.append("test registration scientific digest mismatch")
    if registration.get("phase") != "test":
        errors.append("test registration phase mismatch")
    if registration.get("status") != "PRE_TEST_OUTCOME_INTERNAL_REGISTRATION":
        errors.append("test registration status mismatch")
    if registration.get("registered_command") != TEST_REGISTERED_COMMAND:
        errors.append("test command mismatch")
    if registration.get("scientific_constants") != SCIENTIFIC_CONSTANTS:
        errors.append("scientific constants mismatch")
    if registration.get("implementation_sha256") != implementation_hashes():
        errors.append("implementation hash mismatch")
    if registration.get("development_manifest_sha256") != sha256_file(
        DEVELOPMENT_MANIFEST_PATH
    ):
        errors.append("development manifest hash mismatch")
    if registration.get("test_manifest_sha256") != sha256_file(TEST_MANIFEST_PATH):
        errors.append("test manifest hash mismatch")
    if registration.get("preflight_sha256") != sha256_file(PREFLIGHT_PATH):
        errors.append("preflight hash mismatch")
    if registration.get("development_registration_sha256") != sha256_file(
        DEV_REGISTRATION_PATH
    ):
        errors.append("development registration hash mismatch")
    if registration.get("development_result_sha256") != sha256_file(DEV_RESULT_PATH):
        errors.append("development result file hash mismatch")
    development_result = load_json(DEV_RESULT_PATH)
    if not result_digest_matches(development_result):
        errors.append("development result digest mismatch")
    if development_result.get("formal_verdict") != "PASS_TO_TEST":
        errors.append("development result does not authorize test")
    if registration.get("development_raw_rows_sha256") != sha256_file(
        DEV_RAW_ROWS_PATH
    ):
        errors.append("development raw rows hash mismatch")
    if registration.get("development_result_digest") != development_result.get(
        "result_digest"
    ):
        errors.append("development result lineage mismatch")
    selection = registration.get("development_selection", {})
    if selection.get("static_control") != development_result.get(
        "selected_static_control"
    ):
        errors.append("selected static control mismatch")
    if selection.get("trust_radius_radians") != development_result.get(
        "metrics", {}
    ).get("trust_radius_radians"):
        errors.append("registered trust radius mismatch")
    if selection.get("prototype_digest") != development_result.get(
        "input_digests", {}
    ).get("prototypes"):
        errors.append("registered prototype digest mismatch")
    if registration.get("environment") != environment_info():
        errors.append("environment mismatch")
    current_git = git_info()
    if registration.get("git", {}).get("head") != current_git["head"]:
        errors.append("git HEAD mismatch")
    if current_git["unexpected_status"]:
        errors.append(f"unexpected worktree changes: {current_git['unexpected_status']}")
    current_model_hashes = model_file_hashes(model_snapshot())
    registered_model = registration.get("model", {})
    if registered_model.get("id") != MODEL_ID:
        errors.append("model ID mismatch")
    if registered_model.get("revision") != MODEL_REVISION:
        errors.append("model revision mismatch")
    if registered_model.get("snapshot_file_sha256") != current_model_hashes:
        errors.append("model file hash mismatch")
    if registered_model.get("snapshot_digest") != canonical_json_sha256(
        current_model_hashes
    ):
        errors.append("model snapshot digest mismatch")
    if registration.get("test_embeddings_computed") is not False:
        errors.append("test registration outcome-access flag is not false")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--register-development", action="store_true")
    modes.add_argument("--run-development", action="store_true")
    modes.add_argument("--register-test", action="store_true")
    modes.add_argument("--run-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preflight:
        run_preflight()
    elif args.register_development:
        create_development_registration()
    elif args.run_development:
        execute_phase("development")
    elif args.register_test:
        create_test_registration()
    else:
        execute_phase("test")


if __name__ == "__main__":
    main()

"""Preflight, register, and execute the HLM5 S1 semantic attractor assay."""

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

from attractor import (
    LineSearchError,
    certified_settle,
    independent_matched_settle,
    static_cosine_predictions,
    static_degree_predictions,
    unit,
)
from contract import (
    HASHED_IMPLEMENTATION_PATHS,
    MANIFEST_PATH,
    MODEL_ID,
    MODEL_REVISION,
    PREFLIGHT_PATH,
    RAW_ROWS_PATH,
    REGISTERED_COMMAND,
    REGISTRATION_PATH,
    RESULT_PATH,
    ROOT,
    SCIENTIFIC_CONSTANTS,
)
from metrics import (
    average_rank_auc,
    classify_verdict,
    macro_accuracy,
    maximum_relation_prediction_share,
    paired_relation_bootstrap,
    relation_accuracies,
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
            REGISTRATION_PATH,
            RAW_ROWS_PATH,
            RESULT_PATH,
            ROOT / "docs" / "semantic-attractor-gate-verdict-2026-08-10.md",
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


def manifest_errors(manifest: dict[str, Any]) -> list[str]:
    counts = manifest.get("counts", {})
    expected = {
        "eligible_relations": SCIENTIFIC_CONSTANTS["eligible_relation_count"],
        "eligible_patterns": SCIENTIFIC_CONSTANTS["pattern_count"],
        "stored_addresses": SCIENTIFIC_CONSTANTS["stored_address_count"],
        "supported_queries": SCIENTIFIC_CONSTANTS["supported_query_count"],
        "off_support_queries": SCIENTIFIC_CONSTANTS["off_support_query_count"],
    }
    errors: list[str] = []
    if counts != expected:
        errors.append(f"manifest counts differ: {counts} != {expected}")
    slots = manifest.get("slots", [])
    queries = manifest.get("queries", [])
    if len(slots) != expected["stored_addresses"]:
        errors.append("slot row count differs")
    if len(queries) != expected["supported_queries"]:
        errors.append("query row count differs")
    if [row.get("slot") for row in slots] != list(range(len(slots))):
        errors.append("slot indices are not contiguous")
    slot_pairs = [(row.get("relation"), row.get("subject")) for row in slots]
    if len(slot_pairs) != len(set(slot_pairs)):
        errors.append("stored subject-relation pairs are not unique")
    slot_texts = [row.get("text") for row in slots]
    if len(slot_texts) != len(set(slot_texts)):
        errors.append("rendered canonical address texts are not unique")
    query_ids = [row.get("query_id") for row in queries]
    if len(query_ids) != len(set(query_ids)):
        errors.append("query IDs are not unique")
    query_texts = [row.get("text") for row in queries]
    if len(query_texts) != len(set(query_texts)):
        errors.append("rendered supported query texts are not unique")
    off_support_texts = [row.get("off_support_text") for row in queries]
    if len(off_support_texts) != len(set(off_support_texts)):
        errors.append("rendered off-support query texts are not unique")
    for row in queries:
        target = row.get("target_slot")
        if not isinstance(target, int) or not 0 <= target < len(slots):
            errors.append(f"invalid target slot for {row.get('query_id')}")
            break
    return errors


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


def run_test_scripts() -> list[dict[str, Any]]:
    tests = (
        ROOT / "tests" / "test_attractor.py",
        ROOT / "tests" / "test_prepare_pararel.py",
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


def run_preflight() -> None:
    if REGISTRATION_PATH.exists():
        raise RuntimeError("registration exists; preflight is frozen")
    manifest = load_json(MANIFEST_PATH)
    errors = manifest_errors(manifest)
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
    synthetic = encode_texts(encoder, ["synthetic alpha", "synthetic beta"])
    if tuple(synthetic.shape) != (2, 384):
        raise RuntimeError(f"unexpected synthetic encoder shape: {tuple(synthetic.shape)}")

    keys = torch.eye(4, dtype=torch.float64)
    queries = unit(
        torch.tensor(
            [[0.90, 0.30, 0.10, 0.00], [0.10, 0.85, 0.25, 0.05]],
            dtype=torch.float64,
        )
    )
    candidate = certified_settle(queries, keys)
    matched = independent_matched_settle(queries, keys)
    maximum_error = float(torch.abs(candidate.final_state - matched).max())
    maximum_energy_increase = float(
        (candidate.energies[:, 1:] - candidate.energies[:, :-1]).max()
    )
    if maximum_error > float(SCIENTIFIC_CONSTANTS["matched_state_tolerance"]):
        raise RuntimeError("synthetic matched implementation disagrees")
    if maximum_energy_increase > float(SCIENTIFIC_CONSTANTS["energy_tolerance"]):
        raise RuntimeError("synthetic energy certificate failed")

    receipt = {
        "schema_version": 1,
        "experiment_id": SCIENTIFIC_CONSTANTS["experiment_id"],
        "status": "SYNTHETIC_PREFLIGHT_ONLY",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "real_pararel_embeddings_computed": False,
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "manifest_counts": manifest["counts"],
        "implementation_sha256": implementation_hashes(),
        "git": repository,
        "environment": environment_info(),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "snapshot_file_sha256": model_hashes,
            "snapshot_digest": canonical_json_sha256(model_hashes),
            "synthetic_shape": list(synthetic.shape),
            "synthetic_unit_error": float(
                torch.abs(synthetic.norm(dim=1) - 1.0).max()
            ),
        },
        "synthetic_dynamics": {
            "matched_max_abs_error": maximum_error,
            "max_energy_increase": maximum_energy_increase,
        },
        "tests": tests,
    }
    write_json(PREFLIGHT_PATH, receipt)
    print(f"Synthetic-only preflight PASS: {PREFLIGHT_PATH}")


def create_registration() -> None:
    if REGISTRATION_PATH.exists():
        raise RuntimeError("registration already exists")
    if not PREFLIGHT_PATH.is_file():
        raise RuntimeError("synthetic preflight receipt is missing")
    preflight = load_json(PREFLIGHT_PATH)
    current_hashes = implementation_hashes()
    snapshot_hashes = model_file_hashes(model_snapshot())
    current_environment = environment_info()
    current_git = git_info()
    if preflight.get("real_pararel_embeddings_computed") is not False:
        raise RuntimeError("preflight outcome-access flag is not false")
    if preflight.get("implementation_sha256") != current_hashes:
        raise RuntimeError("implementation changed after preflight")
    if preflight.get("manifest_sha256") != sha256_file(MANIFEST_PATH):
        raise RuntimeError("manifest changed after preflight")
    if preflight.get("environment") != current_environment:
        raise RuntimeError("environment changed after preflight")
    if current_git["unexpected_status"]:
        raise RuntimeError(
            f"unexpected worktree changes before registration: {current_git['unexpected_status']}"
        )
    if preflight.get("git", {}).get("head") != current_git["head"]:
        raise RuntimeError("git HEAD changed after preflight")
    if preflight.get("model", {}).get("snapshot_file_sha256") != snapshot_hashes:
        raise RuntimeError("model snapshot changed after preflight")

    registration = {
        "schema_version": 1,
        "experiment_id": SCIENTIFIC_CONSTANTS["experiment_id"],
        "status": "PRE_OUTCOME_INTERNAL_REGISTRATION",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "registered_command": REGISTERED_COMMAND,
        "scientific_constants": SCIENTIFIC_CONSTANTS,
        "implementation_sha256": current_hashes,
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "preflight_sha256": sha256_file(PREFLIGHT_PATH),
        "environment": current_environment,
        "git": current_git,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "snapshot_file_sha256": snapshot_hashes,
            "snapshot_digest": canonical_json_sha256(snapshot_hashes),
        },
    }
    registration["scientific_digest"] = canonical_json_sha256(registration)
    write_json(REGISTRATION_PATH, registration)
    print(f"Registered before real outcome access: {REGISTRATION_PATH}")
    print(f"Scientific digest: {registration['scientific_digest']}")


def registration_digest_matches(registration: dict[str, Any]) -> bool:
    digest_payload = dict(registration)
    recorded_digest = digest_payload.pop("scientific_digest", None)
    return canonical_json_sha256(digest_payload) == recorded_digest


def registration_errors(registration: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not registration_digest_matches(registration):
        errors.append("registration scientific digest mismatch")
    if registration.get("status") != "PRE_OUTCOME_INTERNAL_REGISTRATION":
        errors.append("registration status mismatch")
    if registration.get("registered_command") != REGISTERED_COMMAND:
        errors.append("registered command mismatch")
    if registration.get("scientific_constants") != SCIENTIFIC_CONSTANTS:
        errors.append("scientific constants mismatch")
    if registration.get("implementation_sha256") != implementation_hashes():
        errors.append("implementation hash mismatch")
    if registration.get("manifest_sha256") != sha256_file(MANIFEST_PATH):
        errors.append("manifest hash mismatch")
    if registration.get("preflight_sha256") != sha256_file(PREFLIGHT_PATH):
        errors.append("preflight hash mismatch")
    if registration.get("environment") != environment_info():
        errors.append("environment mismatch")
    current_git = git_info()
    registered_git = registration.get("git", {})
    if registered_git.get("head") != current_git["head"]:
        errors.append("git HEAD mismatch")
    if current_git["unexpected_status"]:
        errors.append(f"unexpected worktree changes: {current_git['unexpected_status']}")
    model = registration.get("model", {})
    if model.get("id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        errors.append("model identity mismatch")
    current_model_hashes = model_file_hashes(model_snapshot())
    if model.get("snapshot_file_sha256") != current_model_hashes:
        errors.append("model file hash mismatch")
    if model.get("snapshot_digest") != canonical_json_sha256(current_model_hashes):
        errors.append("model digest mismatch")
    return errors


def tensor_digest(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(header + contiguous.numpy().tobytes(order="C")).hexdigest()


def evaluate_rows(
    *,
    kind: str,
    queries: torch.Tensor,
    metadata: list[dict[str, Any]],
    keys: torch.Tensor,
    raw_handle,
) -> dict[str, Any]:
    batch_size = int(SCIENTIFIC_CONSTANTS["dynamics_batch_size"])
    static_slots: list[int] = []
    static_confidences: list[float] = []
    degree_slots: list[int] = []
    candidate_slots: list[int] = []
    candidate_confidences: list[float] = []
    maximum_matched_error = 0.0
    maximum_energy_increase = -float("inf")
    maximum_armijo_residual = -float("inf")
    maximum_final_unit_error = 0.0
    all_final_finite = True
    trajectory_digests: list[str] = []

    for start in range(0, queries.shape[0], batch_size):
        end = min(start + batch_size, queries.shape[0])
        batch = queries[start:end]
        static_slot, static_confidence = static_cosine_predictions(batch, keys)
        degree_slot, _ = static_degree_predictions(
            batch,
            keys,
            degree=int(SCIENTIFIC_CONSTANTS["degree"]),
        )
        trace = certified_settle(
            batch,
            keys,
            degree=int(SCIENTIFIC_CONSTANTS["degree"]),
            steps=int(SCIENTIFIC_CONSTANTS["steps"]),
            initial_step=float(SCIENTIFIC_CONSTANTS["initial_step"]),
            armijo_c=float(SCIENTIFIC_CONSTANTS["armijo_c"]),
            max_halvings=int(SCIENTIFIC_CONSTANTS["max_halvings"]),
            stationary_tolerance=float(
                SCIENTIFIC_CONSTANTS["stationary_tolerance"]
            ),
            energy_tolerance=float(SCIENTIFIC_CONSTANTS["energy_tolerance"]),
        )
        matched = independent_matched_settle(
            batch,
            keys,
            degree=int(SCIENTIFIC_CONSTANTS["degree"]),
            steps=int(SCIENTIFIC_CONSTANTS["steps"]),
            initial_step=float(SCIENTIFIC_CONSTANTS["initial_step"]),
            armijo_c=float(SCIENTIFIC_CONSTANTS["armijo_c"]),
            max_halvings=int(SCIENTIFIC_CONSTANTS["max_halvings"]),
            stationary_tolerance=float(
                SCIENTIFIC_CONSTANTS["stationary_tolerance"]
            ),
            energy_tolerance=float(SCIENTIFIC_CONSTANTS["energy_tolerance"]),
        )
        candidate_slot, candidate_confidence = static_cosine_predictions(
            trace.final_state,
            keys,
        )
        matched_slot, _ = static_cosine_predictions(matched, keys)
        row_errors = torch.abs(trace.final_state - matched).amax(dim=1)
        maximum_matched_error = max(maximum_matched_error, float(row_errors.max()))
        energy_deltas = trace.energies[:, 1:] - trace.energies[:, :-1]
        maximum_energy_increase = max(
            maximum_energy_increase,
            float(energy_deltas.max()),
        )
        maximum_armijo_residual = max(
            maximum_armijo_residual,
            float(trace.armijo_residuals.max()),
        )
        maximum_final_unit_error = max(
            maximum_final_unit_error,
            float(torch.abs(trace.final_state.norm(dim=1) - 1.0).max()),
        )
        all_final_finite = all_final_finite and bool(torch.isfinite(trace.final_state).all())
        trajectory_digests.append(tensor_digest(trace.energies))

        if not torch.equal(candidate_slot, matched_slot):
            raise RuntimeError("candidate and matched settle selected different slots")

        for local_index, meta in enumerate(metadata[start:end]):
            record: dict[str, Any] = {
                "kind": kind,
                "query_id": meta["query_id"],
                "relation": meta["relation"],
                "static_slot": int(static_slot[local_index]),
                "static_confidence": float(static_confidence[local_index]),
                "degree5_slot": int(degree_slot[local_index]),
                "candidate_slot": int(candidate_slot[local_index]),
                "candidate_confidence": float(candidate_confidence[local_index]),
                "matched_slot": int(matched_slot[local_index]),
                "matched_max_abs_error": float(row_errors[local_index]),
                "energies": trace.energies[local_index].tolist(),
                "step_sizes": trace.step_sizes[local_index].tolist(),
                "gradient_norms": trace.gradient_norms[local_index].tolist(),
                "armijo_residuals": trace.armijo_residuals[local_index].tolist(),
                "halvings": trace.halvings[local_index].tolist(),
            }
            if kind == "supported":
                record["target_slot"] = int(meta["target_slot"])
                record["subject"] = meta["subject"]
            else:
                record["off_support_subject"] = meta["off_support_subject"]
            raw_handle.write(json.dumps(record, sort_keys=True) + "\n")

        static_slots.extend(static_slot.tolist())
        static_confidences.extend(static_confidence.tolist())
        degree_slots.extend(degree_slot.tolist())
        candidate_slots.extend(candidate_slot.tolist())
        candidate_confidences.extend(candidate_confidence.tolist())

    return {
        "static_slots": static_slots,
        "static_confidences": static_confidences,
        "degree_slots": degree_slots,
        "candidate_slots": candidate_slots,
        "candidate_confidences": candidate_confidences,
        "maximum_matched_error": maximum_matched_error,
        "maximum_energy_increase": maximum_energy_increase,
        "maximum_armijo_residual": maximum_armijo_residual,
        "maximum_final_unit_error": maximum_final_unit_error,
        "all_final_finite": all_final_finite,
        "trajectory_digest": canonical_json_sha256(trajectory_digests),
    }


def execute_registered() -> None:
    if RESULT_PATH.exists() or RAW_ROWS_PATH.exists():
        raise RuntimeError("registered output already exists; refusing to overwrite")
    registration = load_json(REGISTRATION_PATH)
    errors = registration_errors(registration)
    if errors:
        raise RuntimeError("registration verification failed: " + "; ".join(errors))
    manifest = load_json(MANIFEST_PATH)
    errors = manifest_errors(manifest)
    if errors:
        raise RuntimeError("manifest verification failed: " + "; ".join(errors))

    torch.set_num_threads(1)
    torch.manual_seed(int(SCIENTIFIC_CONSTANTS["seed"]))
    torch.use_deterministic_algorithms(True)
    encoder = load_encoder()
    slots = manifest["slots"]
    metadata = manifest["queries"]
    keys = encode_texts(encoder, [row["text"] for row in slots])
    supported_queries = encode_texts(encoder, [row["text"] for row in metadata])
    off_support_queries = encode_texts(
        encoder,
        [row["off_support_text"] for row in metadata],
    )

    unit_error = max(
        float(torch.abs(keys.norm(dim=1) - 1.0).max()),
        float(torch.abs(supported_queries.norm(dim=1) - 1.0).max()),
        float(torch.abs(off_support_queries.norm(dim=1) - 1.0).max()),
    )
    finite = bool(
        torch.isfinite(keys).all()
        and torch.isfinite(supported_queries).all()
        and torch.isfinite(off_support_queries).all()
    )
    identity = certified_settle(supported_queries, keys, steps=0).final_state
    identity_bit_exact = torch.equal(identity, supported_queries)

    RAW_ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with RAW_ROWS_PATH.open("x", encoding="utf-8", newline="\n") as raw_handle:
            supported = evaluate_rows(
                kind="supported",
                queries=supported_queries,
                metadata=metadata,
                keys=keys,
                raw_handle=raw_handle,
            )
            off_support = evaluate_rows(
                kind="off_support",
                queries=off_support_queries,
                metadata=metadata,
                keys=keys,
                raw_handle=raw_handle,
            )
    except LineSearchError as exc:
        invalid = {
            "schema_version": 1,
            "experiment_id": SCIENTIFIC_CONSTANTS["experiment_id"],
            "formal_verdict": "HARNESS_INVALID",
            "reason": str(exc),
            "registration_scientific_digest": registration["scientific_digest"],
        }
        write_json(RESULT_PATH, invalid)
        raise

    targets = [int(row["target_slot"]) for row in metadata]
    relations = [str(row["relation"]) for row in metadata]
    slot_relations = [str(row["relation"]) for row in slots]
    baseline_by_relation = relation_accuracies(
        supported["static_slots"],
        targets,
        relations,
    )
    candidate_by_relation = relation_accuracies(
        supported["candidate_slots"],
        targets,
        relations,
    )
    interval = paired_relation_bootstrap(
        candidate_by_relation,
        baseline_by_relation,
        resamples=int(SCIENTIFIC_CONSTANTS["bootstrap_resamples"]),
        seed=int(SCIENTIFIC_CONSTANTS["seed"]),
    )
    static_auc = average_rank_auc(
        supported["static_confidences"],
        off_support["static_confidences"],
    )
    candidate_auc = average_rank_auc(
        supported["candidate_confidences"],
        off_support["candidate_confidences"],
    )
    maximum_share = maximum_relation_prediction_share(
        supported["candidate_slots"],
        slot_relations,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(SCIENTIFIC_CONSTANTS["seed"]))
    permutation = torch.randperm(len(slots), generator=generator).tolist()
    shuffled_targets = [permutation[target] for target in targets]
    shuffled_by_relation = relation_accuracies(
        supported["candidate_slots"],
        shuffled_targets,
        relations,
    )
    shuffled_macro = macro_accuracy(shuffled_by_relation)

    singular_values = torch.linalg.svdvals(keys)
    rank_tolerance = max(keys.shape) * torch.finfo(torch.float64).eps * singular_values[0]
    nonzero = singular_values[singular_values > rank_tolerance]
    numerical_rank = int(nonzero.numel())
    condition = (
        float(nonzero[0] / nonzero[-1]) if numerical_rank > 0 else float("inf")
    )
    static_degree_equivalent = supported["static_slots"] == supported["degree_slots"]
    matched_error = max(
        supported["maximum_matched_error"],
        off_support["maximum_matched_error"],
    )
    maximum_energy_increase = max(
        supported["maximum_energy_increase"],
        off_support["maximum_energy_increase"],
    )
    maximum_armijo_residual = max(
        supported["maximum_armijo_residual"],
        off_support["maximum_armijo_residual"],
    )
    maximum_final_unit_error = max(
        supported["maximum_final_unit_error"],
        off_support["maximum_final_unit_error"],
    )
    all_final_finite = bool(
        supported["all_final_finite"] and off_support["all_final_finite"]
    )

    validity = {
        "registration_verified": True,
        "population_exact": not manifest_errors(manifest),
        "finite_and_unit": finite
        and all_final_finite
        and max(unit_error, maximum_final_unit_error)
        <= float(SCIENTIFIC_CONSTANTS["unit_tolerance"]),
        "energy_nonincreasing": maximum_energy_increase
        <= float(SCIENTIFIC_CONSTANTS["energy_tolerance"]),
        "armijo_bound": maximum_armijo_residual
        <= float(SCIENTIFIC_CONSTANTS["energy_tolerance"]),
        "identity_bit_exact": identity_bit_exact,
        "static_degree_equivalent": static_degree_equivalent,
        "independent_matched": matched_error
        <= float(SCIENTIFIC_CONSTANTS["matched_state_tolerance"]),
        "shuffled_target_null": shuffled_macro
        < float(SCIENTIFIC_CONSTANTS["valid_shuffled_macro_max"]),
    }
    scientific = {
        "accuracy_gain": interval["point"]
        >= float(SCIENTIFIC_CONSTANTS["pass_accuracy_gain"]),
        "bootstrap_lower": interval["low"]
        > float(SCIENTIFIC_CONSTANTS["pass_bootstrap_lower"]),
        "auc_noninferiority": candidate_auc - static_auc
        >= float(SCIENTIFIC_CONSTANTS["pass_auc_noninferiority"]),
        "no_relation_collapse": maximum_share
        <= float(SCIENTIFIC_CONSTANTS["pass_max_relation_share"]),
    }
    verdict = classify_verdict(validity, scientific)

    result = {
        "schema_version": 1,
        "experiment_id": SCIENTIFIC_CONSTANTS["experiment_id"],
        "formal_verdict": verdict,
        "registration_scientific_digest": registration["scientific_digest"],
        "registration_sha256": sha256_file(REGISTRATION_PATH),
        "raw_rows_sha256": sha256_file(RAW_ROWS_PATH),
        "input_digests": {
            "keys": tensor_digest(keys),
            "supported_queries": tensor_digest(supported_queries),
            "off_support_queries": tensor_digest(off_support_queries),
            "supported_trajectories": supported["trajectory_digest"],
            "off_support_trajectories": off_support["trajectory_digest"],
            "shuffled_target_permutation": canonical_json_sha256(permutation),
        },
        "counts": manifest["counts"],
        "validity": validity,
        "scientific_bars": scientific,
        "metrics": {
            "static_macro_accuracy": macro_accuracy(baseline_by_relation),
            "candidate_macro_accuracy": macro_accuracy(candidate_by_relation),
            "candidate_minus_static": interval,
            "static_supported_off_support_auc": static_auc,
            "candidate_supported_off_support_auc": candidate_auc,
            "candidate_minus_static_auc": candidate_auc - static_auc,
            "maximum_candidate_relation_share": maximum_share,
            "shuffled_target_macro_accuracy": shuffled_macro,
            "unit_error": unit_error,
            "matched_max_abs_error": matched_error,
            "maximum_energy_increase": maximum_energy_increase,
            "maximum_armijo_residual": maximum_armijo_residual,
            "maximum_final_unit_error": maximum_final_unit_error,
        },
        "per_relation": {
            relation: {
                "static_accuracy": baseline_by_relation[relation],
                "candidate_accuracy": candidate_by_relation[relation],
                "difference": (
                    candidate_by_relation[relation] - baseline_by_relation[relation]
                ),
                "shuffled_accuracy": shuffled_by_relation[relation],
            }
            for relation in sorted(baseline_by_relation)
        },
        "key_geometry": {
            "coherence": float(
                torch.abs(keys @ keys.T - torch.eye(len(keys), dtype=torch.float64)).max()
            ),
            "singular_values": singular_values.tolist(),
            "numerical_rank": numerical_rank,
            "rank_tolerance": float(rank_tolerance),
            "condition_nonzero_spectrum": condition,
        },
        "claim_boundary": (
            "This assay measures fixed polynomial attractor dynamics on frozen ParaRel "
            "sentence embeddings. It does not establish semantic understanding, factual "
            "correctness, LM improvement, production readiness, compliance, scaling "
            "authorization, or an implementation advantage over a matched cache optimizer."
        ),
    }
    result["result_digest"] = canonical_json_sha256(result)
    write_json(RESULT_PATH, result)
    print(f"{verdict}: {RESULT_PATH}")
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--register", action="store_true")
    modes.add_argument("--run-registered", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preflight:
        run_preflight()
    elif args.register:
        create_registration()
    else:
        execute_registered()


if __name__ == "__main__":
    main()

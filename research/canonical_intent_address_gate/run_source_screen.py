"""Register, run, and replay the frozen CA-1 source-only screen."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.natural_key_transfer_gate.geometry import tensor_sha256
from research.natural_key_transfer_gate.run_natural_key_transfer_gate import (
    configure_determinism,
    extract_hiddens,
    load_model_and_tokenizer,
)
from research.two_axis_admission_gate.geometry import apply_arm, fit_key_operators
from research.two_axis_admission_gate.prepare_hwu64 import (
    build_populations,
    population_sha256,
    verify_manifest,
)
from research.two_axis_admission_gate.run_two_axis_admission_gate import (
    SOURCE_HIDDEN_POPULATION_NAMES,
    load_hidden_cache,
    select_hiddens,
)

from .contract import (
    ALLOWED_GROUPS,
    ARM_NAMES,
    AUDIT_GROUP,
    CALIBRATION_GROUP,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DATA_ROOT,
    DEFAULT_SOURCE_CACHE_PATH,
    DEFAULT_TARGET_CACHE_PATH,
    DEFAULT_TOKENIZER_PATH,
    DEPENDENCY_PATHS,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_MINILM_TREE_SHA256,
    EXPECTED_PERMUTATION,
    EXPECTED_PERMUTATION_SHA256,
    EXPECTED_PROTOCOL_SHA256,
    EXPECTED_SOURCE_CACHE_SHA256,
    EXPECTED_SOURCE_HIDDEN_SHA256,
    HLM_ADDRESS_BATCH_SIZE,
    HLM_ADDRESS_CACHE_PATH,
    HLM_ARM_NAMES,
    IMPLEMENTATION_PATHS,
    MANIFEST_PATH,
    MINILM_ARM,
    MINILM_BATCH_SIZE,
    MINILM_CACHE_PATH,
    MINILM_REVISION,
    MINILM_SNAPSHOT_PATH,
    NEGATIVE_QUANTILE,
    PACKAGE_ROOT,
    PERMUTATION_SEED,
    PRIMARY_ARM,
    PROTOCOL_COMMIT,
    PROTOCOL_PATH,
    QUANTILE_METHOD,
    REGISTRATION_PATH,
    REPLAY_PATH,
    REPORT_ARM_NAMES,
    REPO_ROOT,
    RESULT_PATH,
    ROWS_PATH,
    SHUFFLED_ARM,
    SOURCE_BARS,
    SOURCE_QUERY_POPULATIONS,
    directory_tree_record,
    read_json,
    relative_hashes,
    sha256_bytes,
    sha256_file,
    stable_sha256,
    verify_model_file,
    write_json,
)
from .geometry import (
    all_finite,
    canonical_text,
    evaluate_arm,
    hlm_route,
    minilm_route,
    registered_permutation,
    source_verdict,
)


class HarnessInvalid(RuntimeError):
    """Raised when a binding source-screen contract fails."""


def now_local() -> str:
    return dt.datetime.now().astimezone().isoformat()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def git_worktree_is_clean() -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
    ).strip()


def protocol_commit_is_ancestor() -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def source_labels(built: dict[str, Any]) -> dict[str, list[str]]:
    groups = built["metadata"]["groups"]
    labels = {
        "calibration": sorted(groups[CALIBRATION_GROUP]),
        "audit": sorted(groups[AUDIT_GROUP]),
    }
    if len(labels["calibration"]) != 12 or len(labels["audit"]) != 12:
        raise HarnessInvalid("source canonical groups must each contain 12 labels")
    if set(labels["calibration"]) & set(labels["audit"]):
        raise HarnessInvalid("source canonical groups overlap")
    return labels


def scientific_constants() -> dict[str, Any]:
    return {
        "schema": "hlm5-ca1-source-constants-v1",
        "allowed_groups": list(ALLOWED_GROUPS),
        "source_query_populations": list(SOURCE_QUERY_POPULATIONS),
        "canonical_template": 'intent: {label.replace("_", " ")}',
        "hlm_arms": list(HLM_ARM_NAMES),
        "report_arms": list(REPORT_ARM_NAMES),
        "primary_arm": PRIMARY_ARM,
        "shuffled_arm": SHUFFLED_ARM,
        "minilm_arm": MINILM_ARM,
        "permutation_seed": PERMUTATION_SEED,
        "permutation": list(EXPECTED_PERMUTATION),
        "permutation_sha256": EXPECTED_PERMUTATION_SHA256,
        "negative_quantile": NEGATIVE_QUANTILE,
        "quantile_method": QUANTILE_METHOD,
        "comparison": "strict_greater_than",
        "hlm_address_batch_size": HLM_ADDRESS_BATCH_SIZE,
        "minilm_batch_size": MINILM_BATCH_SIZE,
        "minilm_revision": MINILM_REVISION,
        "source_bars": SOURCE_BARS,
        "target_names_encoded": False,
        "target_utterances_encoded": False,
    }


def environment_record() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "tokenizers": importlib.metadata.version("tokenizers"),
        "sentence_transformers": importlib.metadata.version(
            "sentence-transformers"
        ),
        "transformers": importlib.metadata.version("transformers"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name("cuda") if torch.cuda.is_available() else None,
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
        command, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if completed.returncode:
        raise HarnessInvalid(
            "canonical source tests failed\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )
    return {
        "command": command,
        "stdout_sha256": stable_sha256(completed.stdout),
        "stderr_sha256": stable_sha256(completed.stderr),
        "summary": completed.stderr.strip().splitlines()[-1],
    }


def artifact_record() -> dict[str, Any]:
    if sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise HarnessInvalid("source protocol bytes changed")
    if not protocol_commit_is_ancestor():
        raise HarnessInvalid("source protocol commit is not ancestral")
    if sha256_file(MANIFEST_PATH) != EXPECTED_MANIFEST_SHA256["file"]:
        raise HarnessInvalid("SCA-2 HWU64 manifest bytes changed")
    if sha256_file(DEFAULT_SOURCE_CACHE_PATH) != EXPECTED_SOURCE_CACHE_SHA256:
        raise HarnessInvalid("SCA-2 source cache file hash changed")
    minilm = directory_tree_record(MINILM_SNAPSHOT_PATH)
    if minilm["tree_sha256"] != EXPECTED_MINILM_TREE_SHA256:
        raise HarnessInvalid("MiniLM local snapshot tree changed")
    return {
        "protocol": {
            "path": PROTOCOL_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(PROTOCOL_PATH),
            "commit": PROTOCOL_COMMIT,
        },
        "manifest": {
            "path": MANIFEST_PATH.relative_to(REPO_ROOT).as_posix(),
            "file_sha256": sha256_file(MANIFEST_PATH),
            "scientific_sha256": EXPECTED_MANIFEST_SHA256["scientific"],
        },
        "source_cache": {
            "path": str(DEFAULT_SOURCE_CACHE_PATH.resolve()),
            "sha256": sha256_file(DEFAULT_SOURCE_CACHE_PATH),
            "hidden_sha256": EXPECTED_SOURCE_HIDDEN_SHA256,
        },
        "checkpoint": {
            "path": str(DEFAULT_CHECKPOINT_PATH.resolve()),
            "sha256": verify_model_file(DEFAULT_CHECKPOINT_PATH, "checkpoint"),
        },
        "tokenizer": {
            "path": str(DEFAULT_TOKENIZER_PATH.resolve()),
            "sha256": verify_model_file(DEFAULT_TOKENIZER_PATH, "tokenizer"),
        },
        "minilm": minilm,
        "implementation": relative_hashes(IMPLEMENTATION_PATHS),
        "dependencies": relative_hashes(DEPENDENCY_PATHS),
    }


def _forbidden_paths() -> tuple[Path, ...]:
    return (
        REGISTRATION_PATH,
        RESULT_PATH,
        ROWS_PATH,
        REPLAY_PATH,
        HLM_ADDRESS_CACHE_PATH,
        MINILM_CACHE_PATH,
    )


def register_source() -> dict[str, Any]:
    configure_determinism()
    if not git_worktree_is_clean():
        raise HarnessInvalid("registration requires a clean Git worktree")
    if any(path.exists() for path in _forbidden_paths()):
        raise HarnessInvalid("stale canonical source artifact exists")
    manifest = verify_manifest(DEFAULT_DATA_ROOT)
    built = build_populations(DEFAULT_DATA_ROOT)
    labels = source_labels(built)
    target = set(built["metadata"]["groups"]["target_supported"])
    if target & (set(labels["calibration"]) | set(labels["audit"])):
        raise HarnessInvalid("target label entered the source address set")
    permutation = registered_permutation()
    record = {
        "schema": "hlm5-ca1-source-registration-v1",
        "status": "REGISTERED_BEFORE_CANONICAL_ENCODING",
        "generated_at_local": now_local(),
        "git_head": git_head(),
        "environment": environment_record(),
        "artifacts": artifact_record(),
        "scientific_constants": scientific_constants(),
        "scientific_constants_sha256": stable_sha256(scientific_constants()),
        "source_labels": labels,
        "canonical_text_sha256": {
            label: stable_sha256(canonical_text(label))
            for label in labels["calibration"] + labels["audit"]
        },
        "permutation": {
            "values": permutation.tolist(),
            "sha256": tensor_sha256(permutation),
        },
        "manifest_scientific_sha256": manifest["scientific_sha256"],
        "tests": run_unit_tests(),
        "attestation": {
            "canonical_hlm5_labels_encoded": False,
            "canonical_minilm_labels_encoded": False,
            "source_queries_encoded_by_minilm": False,
            "target_names_encoded": False,
            "target_utterances_encoded": False,
            "test_utterances_encoded": False,
        },
    }
    write_json(REGISTRATION_PATH, record)
    return record


def verify_registration(registration: dict[str, Any]) -> dict[str, Any]:
    configure_determinism()
    if registration.get("schema") != "hlm5-ca1-source-registration-v1":
        raise HarnessInvalid("source registration schema changed")
    if registration.get("status") != "REGISTERED_BEFORE_CANONICAL_ENCODING":
        raise HarnessInvalid("source registration status changed")
    if registration.get("git_head") != git_head():
        raise HarnessInvalid("source registration Git head is stale")
    if registration.get("environment") != environment_record():
        raise HarnessInvalid("source registration environment changed")
    if registration.get("artifacts") != artifact_record():
        raise HarnessInvalid("source registration artifact hashes changed")
    if registration.get("scientific_constants") != scientific_constants():
        raise HarnessInvalid("source scientific constants changed")
    if registration.get("scientific_constants_sha256") != stable_sha256(
        scientific_constants()
    ):
        raise HarnessInvalid("source scientific constants hash changed")
    if any(registration.get("attestation", {}).values()):
        raise HarnessInvalid("registration attests to pre-registration outcome access")
    manifest = verify_manifest(DEFAULT_DATA_ROOT)
    if registration.get("manifest_scientific_sha256") != manifest[
        "scientific_sha256"
    ]:
        raise HarnessInvalid("registered manifest changed")
    return manifest


def _canonical_rows(labels: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "label": label,
            "utterance": canonical_text(label),
            "text_sha256": stable_sha256(canonical_text(label)),
        }
        for label in labels
    ]


def create_hlm_address_cache(
    labels: dict[str, list[str]],
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
) -> dict[str, Any]:
    if HLM_ADDRESS_CACHE_PATH.exists():
        raise HarnessInvalid("HLM canonical-address cache already exists")
    combined = sorted(labels["calibration"] + labels["audit"])
    rows = _canonical_rows(combined)
    hiddens, extraction = extract_hiddens(
        model, tokenizer, rows, device, HLM_ADDRESS_BATCH_SIZE
    )
    hiddens = hiddens.to(device="cpu", dtype=torch.float32).contiguous()
    if tuple(hiddens.shape) != (24, 768) or not bool(torch.isfinite(hiddens).all()):
        raise HarnessInvalid("HLM canonical-address tensor is invalid")
    cache = {
        "schema": "hlm5-ca1-hlm-address-cache-v1",
        "labels": combined,
        "canonical_text_sha256": {
            row["label"]: row["text_sha256"] for row in rows
        },
        "hidden_sha256": tensor_sha256(hiddens),
        "hiddens": hiddens,
        "extraction": extraction,
        "target_names_encoded": False,
        "target_utterances_encoded": False,
    }
    HLM_ADDRESS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, HLM_ADDRESS_CACHE_PATH)
    return {
        "path": str(HLM_ADDRESS_CACHE_PATH),
        "sha256": sha256_file(HLM_ADDRESS_CACHE_PATH),
        "hidden_sha256": cache["hidden_sha256"],
        "labels": combined,
        "canonical_text_sha256": cache["canonical_text_sha256"],
        "extraction": extraction,
    }


def load_hlm_address_cache(labels: dict[str, list[str]]) -> tuple[torch.Tensor, dict[str, int], dict[str, Any]]:
    if not HLM_ADDRESS_CACHE_PATH.is_file():
        raise HarnessInvalid("HLM canonical-address cache is missing")
    cache = torch.load(HLM_ADDRESS_CACHE_PATH, map_location="cpu", weights_only=False)
    combined = sorted(labels["calibration"] + labels["audit"])
    expected_text = {label: stable_sha256(canonical_text(label)) for label in combined}
    if cache.get("schema") != "hlm5-ca1-hlm-address-cache-v1":
        raise HarnessInvalid("HLM canonical cache schema changed")
    if cache.get("labels") != combined or cache.get("canonical_text_sha256") != expected_text:
        raise HarnessInvalid("HLM canonical cache labels or texts changed")
    if cache.get("target_names_encoded") or cache.get("target_utterances_encoded"):
        raise HarnessInvalid("HLM canonical cache reports target access")
    hiddens = cache.get("hiddens")
    if not isinstance(hiddens, torch.Tensor):
        raise HarnessInvalid("HLM canonical cache tensor is missing")
    hiddens = hiddens.to(device="cpu", dtype=torch.float32).contiguous()
    if tuple(hiddens.shape) != (24, 768) or tensor_sha256(hiddens) != cache.get(
        "hidden_sha256"
    ):
        raise HarnessInvalid("HLM canonical cache tensor changed")
    index = {label: position for position, label in enumerate(combined)}
    record = {
        "path": str(HLM_ADDRESS_CACHE_PATH),
        "sha256": sha256_file(HLM_ADDRESS_CACHE_PATH),
        "hidden_sha256": cache["hidden_sha256"],
        "labels": combined,
        "canonical_text_sha256": expected_text,
        "extraction": cache.get("extraction"),
    }
    return hiddens, index, record


def create_minilm_cache(
    labels: dict[str, list[str]], populations: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    if MINILM_CACHE_PATH.exists():
        raise HarnessInvalid("MiniLM source cache already exists")
    from sentence_transformers import SentenceTransformer

    combined_labels = sorted(labels["calibration"] + labels["audit"])
    canonical_texts = [canonical_text(label) for label in combined_labels]
    query_rows_by_id: dict[str, dict[str, Any]] = {}
    for name in SOURCE_QUERY_POPULATIONS:
        for row in populations[name]:
            query_rows_by_id[row["row_id"]] = row
    query_rows = [query_rows_by_id[row_id] for row_id in sorted(query_rows_by_id)]
    model = SentenceTransformer(
        str(MINILM_SNAPSHOT_PATH), device="cpu", local_files_only=True
    )
    label_embeddings = model.encode(
        canonical_texts,
        batch_size=MINILM_BATCH_SIZE,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).to(device="cpu", dtype=torch.float32)
    query_embeddings = model.encode(
        [row["utterance"] for row in query_rows],
        batch_size=MINILM_BATCH_SIZE,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).to(device="cpu", dtype=torch.float32)
    if tuple(label_embeddings.shape) != (24, 384):
        raise HarnessInvalid("MiniLM canonical-label shape changed")
    if tuple(query_embeddings.shape) != (384, 384):
        raise HarnessInvalid("MiniLM source-query shape changed")
    cache = {
        "schema": "hlm5-ca1-minilm-source-cache-v1",
        "revision": MINILM_REVISION,
        "tree_sha256": EXPECTED_MINILM_TREE_SHA256,
        "labels": combined_labels,
        "canonical_text_sha256": {
            label: stable_sha256(canonical_text(label)) for label in combined_labels
        },
        "query_row_ids": [row["row_id"] for row in query_rows],
        "query_population_sha256": {
            name: population_sha256(populations[name])
            for name in SOURCE_QUERY_POPULATIONS
        },
        "label_embeddings_sha256": tensor_sha256(label_embeddings),
        "query_embeddings_sha256": tensor_sha256(query_embeddings),
        "label_embeddings": label_embeddings,
        "query_embeddings": query_embeddings,
        "target_names_encoded": False,
        "target_utterances_encoded": False,
    }
    MINILM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, MINILM_CACHE_PATH)
    return {
        "path": str(MINILM_CACHE_PATH),
        "sha256": sha256_file(MINILM_CACHE_PATH),
        "label_embeddings_sha256": cache["label_embeddings_sha256"],
        "query_embeddings_sha256": cache["query_embeddings_sha256"],
        "labels": combined_labels,
        "query_row_ids": cache["query_row_ids"],
    }


def load_minilm_cache(
    labels: dict[str, list[str]], populations: dict[str, list[dict[str, Any]]]
) -> tuple[torch.Tensor, dict[str, int], torch.Tensor, dict[str, int], dict[str, Any]]:
    if not MINILM_CACHE_PATH.is_file():
        raise HarnessInvalid("MiniLM source cache is missing")
    cache = torch.load(MINILM_CACHE_PATH, map_location="cpu", weights_only=False)
    combined_labels = sorted(labels["calibration"] + labels["audit"])
    query_rows_by_id = {
        row["row_id"]: row
        for name in SOURCE_QUERY_POPULATIONS
        for row in populations[name]
    }
    query_ids = sorted(query_rows_by_id)
    if cache.get("schema") != "hlm5-ca1-minilm-source-cache-v1":
        raise HarnessInvalid("MiniLM source cache schema changed")
    if cache.get("revision") != MINILM_REVISION or cache.get(
        "tree_sha256"
    ) != EXPECTED_MINILM_TREE_SHA256:
        raise HarnessInvalid("MiniLM source cache model changed")
    if cache.get("labels") != combined_labels or cache.get("query_row_ids") != query_ids:
        raise HarnessInvalid("MiniLM source cache population changed")
    expected_populations = {
        name: population_sha256(populations[name]) for name in SOURCE_QUERY_POPULATIONS
    }
    if cache.get("query_population_sha256") != expected_populations:
        raise HarnessInvalid("MiniLM source cache query hashes changed")
    if cache.get("target_names_encoded") or cache.get("target_utterances_encoded"):
        raise HarnessInvalid("MiniLM source cache reports target access")
    label_embeddings = cache.get("label_embeddings")
    query_embeddings = cache.get("query_embeddings")
    if not isinstance(label_embeddings, torch.Tensor) or not isinstance(
        query_embeddings, torch.Tensor
    ):
        raise HarnessInvalid("MiniLM cache tensors are missing")
    label_embeddings = label_embeddings.float().contiguous()
    query_embeddings = query_embeddings.float().contiguous()
    if tuple(label_embeddings.shape) != (24, 384) or tuple(query_embeddings.shape) != (
        384,
        384,
    ):
        raise HarnessInvalid("MiniLM cache tensor shape changed")
    if tensor_sha256(label_embeddings) != cache.get(
        "label_embeddings_sha256"
    ) or tensor_sha256(query_embeddings) != cache.get("query_embeddings_sha256"):
        raise HarnessInvalid("MiniLM cache tensor hash changed")
    record = {
        "path": str(MINILM_CACHE_PATH),
        "sha256": sha256_file(MINILM_CACHE_PATH),
        "label_embeddings_sha256": cache["label_embeddings_sha256"],
        "query_embeddings_sha256": cache["query_embeddings_sha256"],
        "labels": combined_labels,
        "query_row_ids": query_ids,
    }
    return (
        label_embeddings,
        {label: index for index, label in enumerate(combined_labels)},
        query_embeddings,
        {row_id: index for index, row_id in enumerate(query_ids)},
        record,
    )


def _select_labels(
    values: torch.Tensor, index: dict[str, int], labels: list[str]
) -> torch.Tensor:
    return values[torch.tensor([index[label] for label in labels], dtype=torch.long)].double()


def _select_queries(
    values: torch.Tensor, index: dict[str, int], rows: list[dict[str, Any]]
) -> torch.Tensor:
    return values[
        torch.tensor([index[row["row_id"]] for row in rows], dtype=torch.long)
    ].double()


def compute_science(
    populations: dict[str, list[dict[str, Any]]],
    built: dict[str, Any],
    source_hiddens: torch.Tensor,
    source_index: dict[str, int],
    hlm_addresses: torch.Tensor,
    hlm_address_index: dict[str, int],
    minilm_labels: torch.Tensor,
    minilm_label_index: dict[str, int],
    minilm_queries: torch.Tensor,
    minilm_query_index: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = source_labels(built)
    source_values = {
        name: select_hiddens(source_hiddens, source_index, populations[name])
        for name in ("source_fit",) + SOURCE_QUERY_POPULATIONS
    }
    operators = fit_key_operators(source_values["source_fit"])
    truth = {
        name: [row["intent"] for row in populations[name]]
        for name in SOURCE_QUERY_POPULATIONS
    }
    hlm_keys = {
        "calibration": _select_labels(
            hlm_addresses, hlm_address_index, labels["calibration"]
        ),
        "audit": _select_labels(hlm_addresses, hlm_address_index, labels["audit"]),
    }
    arms: dict[str, Any] = {}
    arrays: dict[str, Any] = {}
    raw_routes: dict[str, Any] = {}
    for base_arm in ARM_NAMES:
        arm = f"hlm5_{base_arm}_canonical"
        calibration_keys = apply_arm(hlm_keys["calibration"], operators, base_arm)
        audit_keys = apply_arm(hlm_keys["audit"], operators, base_arm)
        calibration_positive = hlm_route(
            apply_arm(source_values["calibration_positive"], operators, base_arm),
            calibration_keys,
            labels["calibration"],
        )
        calibration_negative = hlm_route(
            apply_arm(source_values["calibration_negative"], operators, base_arm),
            calibration_keys,
            labels["calibration"],
        )
        audit_positive = hlm_route(
            apply_arm(source_values["audit_positive"], operators, base_arm),
            audit_keys,
            labels["audit"],
        )
        audit_negative = hlm_route(
            apply_arm(source_values["audit_negative"], operators, base_arm),
            audit_keys,
            labels["audit"],
        )
        metrics, arm_arrays = evaluate_arm(
            calibration_positive,
            truth["calibration_positive"],
            calibration_negative,
            audit_positive,
            truth["audit_positive"],
            audit_negative,
        )
        metrics["arm"] = arm
        arms[arm] = metrics
        arrays[arm] = arm_arrays
        if base_arm == "raw":
            raw_routes = {
                "calibration_positive": calibration_positive,
                "calibration_negative": calibration_negative,
                "audit_positive": audit_positive,
                "audit_negative": audit_negative,
            }

    permutation = registered_permutation()
    shuffled_calibration = [labels["calibration"][i] for i in permutation.tolist()]
    shuffled_audit = [labels["audit"][i] for i in permutation.tolist()]
    shuffled_routes = {
        "calibration_positive": hlm_route(
            apply_arm(source_values["calibration_positive"], operators, "raw"),
            apply_arm(hlm_keys["calibration"], operators, "raw"),
            shuffled_calibration,
        ),
        "calibration_negative": hlm_route(
            apply_arm(source_values["calibration_negative"], operators, "raw"),
            apply_arm(hlm_keys["calibration"], operators, "raw"),
            shuffled_calibration,
        ),
        "audit_positive": hlm_route(
            apply_arm(source_values["audit_positive"], operators, "raw"),
            apply_arm(hlm_keys["audit"], operators, "raw"),
            shuffled_audit,
        ),
        "audit_negative": hlm_route(
            apply_arm(source_values["audit_negative"], operators, "raw"),
            apply_arm(hlm_keys["audit"], operators, "raw"),
            shuffled_audit,
        ),
    }
    shuffled_metrics, shuffled_arrays = evaluate_arm(
        shuffled_routes["calibration_positive"],
        truth["calibration_positive"],
        shuffled_routes["calibration_negative"],
        shuffled_routes["audit_positive"],
        truth["audit_positive"],
        shuffled_routes["audit_negative"],
    )
    shuffled_metrics["arm"] = SHUFFLED_ARM
    shuffled_metrics["label_assignment"] = {
        "calibration": shuffled_calibration,
        "audit": shuffled_audit,
    }
    arms[SHUFFLED_ARM] = shuffled_metrics
    arrays[SHUFFLED_ARM] = shuffled_arrays

    minilm_keys = {
        name: _select_labels(minilm_labels, minilm_label_index, labels[name])
        for name in ("calibration", "audit")
    }
    minilm_source = {
        name: _select_queries(minilm_queries, minilm_query_index, populations[name])
        for name in SOURCE_QUERY_POPULATIONS
    }
    minilm_routes = {
        "calibration_positive": minilm_route(
            minilm_source["calibration_positive"],
            minilm_keys["calibration"],
            labels["calibration"],
        ),
        "calibration_negative": minilm_route(
            minilm_source["calibration_negative"],
            minilm_keys["calibration"],
            labels["calibration"],
        ),
        "audit_positive": minilm_route(
            minilm_source["audit_positive"],
            minilm_keys["audit"],
            labels["audit"],
        ),
        "audit_negative": minilm_route(
            minilm_source["audit_negative"],
            minilm_keys["audit"],
            labels["audit"],
        ),
    }
    minilm_metrics, minilm_arrays = evaluate_arm(
        minilm_routes["calibration_positive"],
        truth["calibration_positive"],
        minilm_routes["calibration_negative"],
        minilm_routes["audit_positive"],
        truth["audit_positive"],
        minilm_routes["audit_negative"],
    )
    minilm_metrics["arm"] = MINILM_ARM
    arms[MINILM_ARM] = minilm_metrics
    arrays[MINILM_ARM] = minilm_arrays

    verdict, bars, comparisons = source_verdict(arms)
    raw_shuffle_score_identity = {
        name: raw_routes[name]["scores_sha256"]
        == shuffled_routes[name]["scores_sha256"]
        for name in raw_routes
    }
    science = {
        "labels": labels,
        "canonical_text_sha256": {
            label: stable_sha256(canonical_text(label))
            for label in labels["calibration"] + labels["audit"]
        },
        "operators": operators.metadata,
        "arms": arms,
        "comparisons": comparisons,
        "bars": bars,
        "verdict": verdict,
        "raw_shuffle_score_identity": raw_shuffle_score_identity,
    }
    return science, arrays


def build_rows(
    populations: dict[str, list[dict[str, Any]]],
    science: dict[str, Any],
    arrays: dict[str, Any],
) -> list[dict[str, Any]]:
    stage_map = {
        "calibration_positive": "calibration_supported",
        "calibration_negative": "calibration_off_support",
        "audit_positive": "audit_supported",
        "audit_negative": "audit_off_support",
    }
    rows: list[dict[str, Any]] = []
    for population_name, metric_name in stage_map.items():
        supported = population_name.endswith("positive")
        for row_index, row in enumerate(populations[population_name]):
            arms: dict[str, Any] = {}
            for arm in REPORT_ARM_NAMES:
                data = arrays[arm][metric_name]
                record = {
                    "predicted_intent": data["predicted"][row_index],
                    "maximum_score": float(data["maximum_scores"][row_index].item()),
                    "margin": float(data["margins"][row_index].item()),
                    "tau_absolute": science["arms"][arm]["threshold"][
                        "tau_absolute"
                    ],
                    "admitted": bool(data["admitted"][row_index].item()),
                }
                if supported:
                    record["correct"] = bool(data["correct"][row_index].item())
                arms[arm] = record
            output = {
                "population": population_name,
                "kind": "supported" if supported else "off_support",
                "row_id": row["row_id"],
                "fold": row["fold"],
                "source_index": row["source_index"],
                "utterance_sha256": row["utterance_sha256"],
                "arms": arms,
            }
            if supported:
                output["true_intent"] = row["intent"]
            else:
                output["source_intent"] = row["intent"]
            rows.append(output)
    return rows


def rows_text(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    )


def science_validity(
    science: dict[str, Any],
    built: dict[str, Any],
    hlm_addresses: torch.Tensor,
    minilm_labels: torch.Tensor,
    minilm_queries: torch.Tensor,
) -> dict[str, bool]:
    target = set(built["metadata"]["groups"]["target_supported"])
    allowed = set(science["labels"]["calibration"]) | set(
        science["labels"]["audit"]
    )
    independent_errors = [
        metrics[stage]["independent_max_abs_error"]
        for metrics in science["arms"].values()
        for stage in (
            "calibration_supported",
            "calibration_off_support",
            "audit_supported",
            "audit_off_support",
        )
    ]
    calibration_rates = [
        metrics["threshold"]["negative_admission_rate"]
        for metrics in science["arms"].values()
    ]
    label_norm_error = float(
        torch.max(torch.abs(torch.linalg.vector_norm(minilm_labels.double(), dim=1) - 1.0)).item()
    )
    query_norm_error = float(
        torch.max(torch.abs(torch.linalg.vector_norm(minilm_queries.double(), dim=1) - 1.0)).item()
    )
    return {
        "protocol_committed_before_canonical_encoding": protocol_commit_is_ancestor(),
        "source_cache_file_hash_exact": sha256_file(DEFAULT_SOURCE_CACHE_PATH)
        == EXPECTED_SOURCE_CACHE_SHA256,
        "target_cache_absent": not DEFAULT_TARGET_CACHE_PATH.exists(),
        "target_labels_excluded": not bool(target & allowed),
        "exactly_24_source_labels": len(allowed) == 24,
        "hlm_address_shape_finite": tuple(hlm_addresses.shape) == (24, 768)
        and bool(torch.isfinite(hlm_addresses).all()),
        "minilm_shapes_finite": tuple(minilm_labels.shape) == (24, 384)
        and tuple(minilm_queries.shape) == (384, 384)
        and bool(torch.isfinite(minilm_labels).all())
        and bool(torch.isfinite(minilm_queries).all()),
        "minilm_embeddings_normalized": label_norm_error <= 1e-6
        and query_norm_error <= 1e-6,
        "permutation_exact": tuple(registered_permutation().tolist())
        == EXPECTED_PERMUTATION,
        "independent_numpy_readers_match": all(
            error <= 1e-10 for error in independent_errors
        ),
        "calibration_rates_valid": all(rate <= 0.05 for rate in calibration_rates),
        "raw_shuffle_scores_exact": all(
            science["raw_shuffle_score_identity"].values()
        ),
        "scientific_values_finite": all_finite(science),
    }


def run_source() -> dict[str, Any]:
    configure_determinism()
    if not REGISTRATION_PATH.is_file():
        raise HarnessInvalid("source registration is missing")
    if any(path.exists() for path in (RESULT_PATH, ROWS_PATH, REPLAY_PATH)):
        raise HarnessInvalid("canonical source result already exists")
    if HLM_ADDRESS_CACHE_PATH.exists() or MINILM_CACHE_PATH.exists():
        raise HarnessInvalid("canonical cache exists before first source run")
    if not torch.cuda.is_available():
        raise HarnessInvalid("canonical HLM5 source encoding requires local CUDA")
    registration = read_json(REGISTRATION_PATH)
    manifest = verify_registration(registration)
    built = build_populations(DEFAULT_DATA_ROOT)
    populations = built["rows"]
    labels = source_labels(built)
    source_hiddens, source_index, source_cache_record = load_hidden_cache(
        "source_audit",
        DEFAULT_SOURCE_CACHE_PATH,
        populations,
        SOURCE_HIDDEN_POPULATION_NAMES,
    )
    if source_cache_record["hidden_sha256"] != EXPECTED_SOURCE_HIDDEN_SHA256:
        raise HarnessInvalid("SCA-2 source hidden tensor changed")
    model, tokenizer, model_metadata = load_model_and_tokenizer(
        DEFAULT_CHECKPOINT_PATH, DEFAULT_TOKENIZER_PATH, torch.device("cuda")
    )
    hlm_cache_record = create_hlm_address_cache(
        labels, model, tokenizer, torch.device("cuda")
    )
    minilm_cache_record = create_minilm_cache(labels, populations)
    hlm_addresses, hlm_index, loaded_hlm_record = load_hlm_address_cache(labels)
    (
        minilm_labels,
        minilm_label_index,
        minilm_queries,
        minilm_query_index,
        loaded_minilm_record,
    ) = load_minilm_cache(labels, populations)
    if hlm_cache_record != loaded_hlm_record or minilm_cache_record != loaded_minilm_record:
        raise HarnessInvalid("new canonical cache failed immediate reload")
    science, arrays = compute_science(
        populations,
        built,
        source_hiddens,
        source_index,
        hlm_addresses,
        hlm_index,
        minilm_labels,
        minilm_label_index,
        minilm_queries,
        minilm_query_index,
    )
    validity = science_validity(
        science, built, hlm_addresses, minilm_labels, minilm_queries
    )
    validity.update(
        {
            "registration_verified": True,
            "manifest_verified": manifest["scientific_sha256"]
            == EXPECTED_MANIFEST_SHA256["scientific"],
            "source_hidden_tensor_exact": source_cache_record["hidden_sha256"]
            == EXPECTED_SOURCE_HIDDEN_SHA256,
            "target_names_not_encoded": True,
            "target_utterances_not_encoded": True,
            "test_utterances_not_encoded": True,
        }
    )
    if not all(validity.values()):
        raise HarnessInvalid(f"canonical source validity failed: {validity}")
    rows = build_rows(populations, science, arrays)
    serialized_rows = rows_text(rows)
    scientific = {
        "population": {
            "manifest_scientific_sha256": manifest["scientific_sha256"],
            "query_hashes": {
                name: population_sha256(populations[name])
                for name in SOURCE_QUERY_POPULATIONS
            },
            "target_names_encoded": False,
            "target_utterances_encoded": False,
            "test_utterances_encoded": False,
        },
        **science,
    }
    result = {
        "schema": "hlm5-ca1-source-result-v1",
        "status": "VALID",
        "verdict": science["verdict"],
        "generated_at_local": now_local(),
        "git_head": git_head(),
        "registration": {
            "path": REGISTRATION_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(REGISTRATION_PATH),
        },
        "model": model_metadata,
        "source_cache": source_cache_record,
        "canonical_caches": {
            "hlm5": loaded_hlm_record,
            "minilm": loaded_minilm_record,
        },
        "validity": validity,
        "scientific": scientific,
        "scientific_sha256": stable_sha256(scientific),
        "rows_object_sha256": stable_sha256(rows),
        "rows_jsonl_sha256": sha256_bytes(serialized_rows.encode("utf-8")),
        "claim_scope": (
            "Source-only deterministic intent-name routing in frozen HLM5-136M "
            "final-hidden geometry, with a pinned MiniLM control; no target, "
            "test, recurrence, product, explanation, compliance, or scaling claim."
        ),
    }
    ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ROWS_PATH.write_text(serialized_rows, encoding="utf-8", newline="\n")
    write_json(RESULT_PATH, result)
    return result


def replay_source() -> dict[str, Any]:
    configure_determinism()
    if not RESULT_PATH.is_file() or not ROWS_PATH.is_file():
        raise HarnessInvalid("canonical source result is missing")
    if REPLAY_PATH.exists():
        raise HarnessInvalid("canonical source replay already exists")
    registration = read_json(REGISTRATION_PATH)
    manifest = verify_registration(registration)
    canonical = read_json(RESULT_PATH)
    if canonical.get("registration", {}).get("sha256") != sha256_file(
        REGISTRATION_PATH
    ):
        raise HarnessInvalid("canonical result registration binding changed")
    built = build_populations(DEFAULT_DATA_ROOT)
    populations = built["rows"]
    labels = source_labels(built)
    source_hiddens, source_index, source_cache_record = load_hidden_cache(
        "source_audit",
        DEFAULT_SOURCE_CACHE_PATH,
        populations,
        SOURCE_HIDDEN_POPULATION_NAMES,
    )
    hlm_addresses, hlm_index, hlm_record = load_hlm_address_cache(labels)
    (
        minilm_labels,
        minilm_label_index,
        minilm_queries,
        minilm_query_index,
        minilm_record,
    ) = load_minilm_cache(labels, populations)
    science, arrays = compute_science(
        populations,
        built,
        source_hiddens,
        source_index,
        hlm_addresses,
        hlm_index,
        minilm_labels,
        minilm_label_index,
        minilm_queries,
        minilm_query_index,
    )
    scientific = {
        "population": {
            "manifest_scientific_sha256": manifest["scientific_sha256"],
            "query_hashes": {
                name: population_sha256(populations[name])
                for name in SOURCE_QUERY_POPULATIONS
            },
            "target_names_encoded": False,
            "target_utterances_encoded": False,
            "test_utterances_encoded": False,
        },
        **science,
    }
    rows = build_rows(populations, science, arrays)
    serialized_rows = rows_text(rows)
    checks = {
        "scientific_sha256_exact": stable_sha256(scientific)
        == canonical["scientific_sha256"],
        "rows_object_sha256_exact": stable_sha256(rows)
        == canonical["rows_object_sha256"],
        "rows_jsonl_sha256_exact": sha256_bytes(serialized_rows.encode("utf-8"))
        == canonical["rows_jsonl_sha256"]
        == sha256_file(ROWS_PATH),
        "source_cache_exact": source_cache_record["sha256"]
        == canonical["source_cache"]["sha256"],
        "hlm_cache_exact": hlm_record["sha256"]
        == canonical["canonical_caches"]["hlm5"]["sha256"],
        "minilm_cache_exact": minilm_record["sha256"]
        == canonical["canonical_caches"]["minilm"]["sha256"],
        "verdict_exact": science["verdict"] == canonical["verdict"],
        "target_cache_absent": not DEFAULT_TARGET_CACHE_PATH.exists(),
    }
    if not all(checks.values()):
        raise HarnessInvalid(f"canonical source replay mismatch: {checks}")
    receipt = {
        "schema": "hlm5-ca1-source-replay-v1",
        "status": "EXACT_REPLAY",
        "generated_at_local": now_local(),
        "git_head": git_head(),
        "canonical_result": {
            "path": RESULT_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(RESULT_PATH),
        },
        "canonical_rows": {
            "path": ROWS_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(ROWS_PATH),
        },
        "checks": checks,
    }
    write_json(REPLAY_PATH, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "run", "replay"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "register":
        result = register_source()
        print(f"status={result['status']}")
        print(f"registration_sha256={sha256_file(REGISTRATION_PATH)}")
        print("canonical_labels_encoded=false")
    elif args.mode == "run":
        result = run_source()
        print(f"status={result['status']}")
        print(f"verdict={result['verdict']}")
        print(f"scientific_sha256={result['scientific_sha256']}")
        print("target_names_encoded=false")
        print("target_utterances_encoded=false")
    elif args.mode == "replay":
        result = replay_source()
        print(f"status={result['status']}")
        print(f"replay_sha256={sha256_file(REPLAY_PATH)}")
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()

"""Frozen constants and provenance helpers for the CA-1 source screen."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from research.two_axis_admission_gate.contract import (
    ARM_NAMES,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DATA_ROOT,
    DEFAULT_SOURCE_CACHE_PATH,
    DEFAULT_TARGET_CACHE_PATH,
    DEFAULT_TOKENIZER_PATH,
    EXPECTED_MANIFEST_SHA256,
    MANIFEST_PATH,
    REPO_ROOT,
    SNAPSHOT_ROOT,
    read_json,
    relative_hashes,
    sha256_bytes,
    sha256_file,
    stable_json,
    stable_sha256,
    verify_model_file,
    write_json,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = (
    PACKAGE_ROOT
    / "docs"
    / "canonical-intent-address-source-protocol-2026-08-10.md"
)
REGISTRATION_PATH = PACKAGE_ROOT / "source_registration.json"
RESULT_PATH = PACKAGE_ROOT / "results" / "canonical_intent_source_result.json"
ROWS_PATH = PACKAGE_ROOT / "results" / "canonical_intent_source_rows.jsonl"
REPLAY_PATH = PACKAGE_ROOT / "results" / "canonical_intent_source_replay.json"

HLM_ADDRESS_CACHE_PATH = Path(
    r"D:\HLM2\runtime\ca1\hwu64-source-canonical-hlm5.pt"
)
MINILM_CACHE_PATH = Path(
    r"D:\HLM2\runtime\ca1\hwu64-source-canonical-minilm.pt"
)
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MINILM_SNAPSHOT_PATH = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--sentence-transformers--all-MiniLM-L6-v2"
    / "snapshots"
    / MINILM_REVISION
)

PROTOCOL_COMMIT = "b3bd4297fc76e8282b506736a3f5f3564513ffb9"
EXPECTED_PROTOCOL_SHA256 = (
    "312c37dd3488c1bf7f296d13636c20586192f6b4a75f5bbd3df635eac5c67082"
)
EXPECTED_SOURCE_CACHE_SHA256 = (
    "dbf9bc07133c2029095ddad20a990586b73bf354d331dad4e035f9400d1be6f7"
)
EXPECTED_SOURCE_HIDDEN_SHA256 = (
    "9c77843334cdc54450f6c861e2aaf29cec91662f88f847bec6f1337e49e20667"
)
EXPECTED_MINILM_TREE_SHA256 = (
    "200f0f4334d089279904ae34725fab198753c92ee2c0aaf930923cac9ebeee73"
)

CANONICAL_PREFIX = "intent: "
CALIBRATION_GROUP = "calibration_supported"
AUDIT_GROUP = "audit_supported"
ALLOWED_GROUPS = (CALIBRATION_GROUP, AUDIT_GROUP)
SOURCE_QUERY_POPULATIONS = (
    "calibration_positive",
    "calibration_negative",
    "audit_positive",
    "audit_negative",
)

PRIMARY_ARM = "hlm5_raw_canonical"
SHUFFLED_ARM = "hlm5_raw_shuffled_labels"
MINILM_ARM = "minilm_canonical"
HLM_ARM_NAMES = tuple(f"hlm5_{name}_canonical" for name in ARM_NAMES)
REPORT_ARM_NAMES = HLM_ARM_NAMES + (SHUFFLED_ARM, MINILM_ARM)

PERMUTATION_SEED = 20260813
EXPECTED_PERMUTATION = (7, 6, 8, 9, 2, 4, 0, 10, 1, 3, 11, 5)
EXPECTED_PERMUTATION_SHA256 = (
    "503112e320778589b2770c809476161f9c24fed621c69766c4e4b5f1674c6a9b"
)

NEGATIVE_QUANTILE = 0.95
QUANTILE_METHOD = "higher"
HLM_ADDRESS_BATCH_SIZE = 24
MINILM_BATCH_SIZE = 64

SOURCE_BARS = {
    "macro_top1_accuracy": 0.40,
    "macro_coverage": 0.20,
    "macro_correct_admission_rate": 0.15,
    "selective_accuracy": 0.65,
    "off_support_false_admission_rate": 0.05,
    "maximum_predicted_intent_share": 0.25,
    "top1_gain_over_shuffled": 0.20,
    "correct_admission_gain_over_shuffled": 0.10,
    "top1_drop_vs_minilm": 0.10,
}

IMPLEMENTATION_PATHS = (
    PACKAGE_ROOT / "__init__.py",
    PACKAGE_ROOT / "contract.py",
    PACKAGE_ROOT / "geometry.py",
    PACKAGE_ROOT / "run_source_screen.py",
    PACKAGE_ROOT / "tests" / "test_geometry.py",
    PACKAGE_ROOT / "tests" / "test_registration.py",
)

DEPENDENCY_PATHS = (
    REPO_ROOT / "research" / "two_axis_admission_gate" / "contract.py",
    REPO_ROOT / "research" / "two_axis_admission_gate" / "geometry.py",
    REPO_ROOT / "research" / "two_axis_admission_gate" / "prepare_hwu64.py",
    REPO_ROOT
    / "research"
    / "two_axis_admission_gate"
    / "run_two_axis_admission_gate.py",
    REPO_ROOT / "research" / "source_calibrated_admission_gate" / "geometry.py",
    REPO_ROOT / "research" / "natural_key_transfer_gate" / "geometry.py",
    REPO_ROOT
    / "research"
    / "natural_key_transfer_gate"
    / "run_natural_key_transfer_gate.py",
    SNAPSHOT_ROOT / "hlm5_lm.py",
)


def directory_tree_record(root: Path) -> dict[str, Any]:
    """Hash a fixed local model snapshot by relative name, bytes, and size."""

    if not root.is_dir():
        raise FileNotFoundError(root)
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    payload = "".join(
        f"{item['path']}\0{item['sha256']}\0{item['size']}\n" for item in files
    ).encode("utf-8")
    return {
        "root": str(root.resolve()),
        "revision": MINILM_REVISION,
        "files": files,
        "tree_sha256": hashlib.sha256(payload).hexdigest(),
    }


__all__ = [
    "ALLOWED_GROUPS",
    "ARM_NAMES",
    "AUDIT_GROUP",
    "CANONICAL_PREFIX",
    "CALIBRATION_GROUP",
    "DEFAULT_CHECKPOINT_PATH",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_SOURCE_CACHE_PATH",
    "DEFAULT_TARGET_CACHE_PATH",
    "DEFAULT_TOKENIZER_PATH",
    "DEPENDENCY_PATHS",
    "EXPECTED_MANIFEST_SHA256",
    "EXPECTED_MINILM_TREE_SHA256",
    "EXPECTED_PERMUTATION",
    "EXPECTED_PERMUTATION_SHA256",
    "EXPECTED_PROTOCOL_SHA256",
    "EXPECTED_SOURCE_CACHE_SHA256",
    "EXPECTED_SOURCE_HIDDEN_SHA256",
    "HLM_ADDRESS_BATCH_SIZE",
    "HLM_ADDRESS_CACHE_PATH",
    "HLM_ARM_NAMES",
    "IMPLEMENTATION_PATHS",
    "MANIFEST_PATH",
    "MINILM_ARM",
    "MINILM_BATCH_SIZE",
    "MINILM_CACHE_PATH",
    "MINILM_REVISION",
    "MINILM_SNAPSHOT_PATH",
    "NEGATIVE_QUANTILE",
    "PACKAGE_ROOT",
    "PERMUTATION_SEED",
    "PRIMARY_ARM",
    "PROTOCOL_COMMIT",
    "PROTOCOL_PATH",
    "QUANTILE_METHOD",
    "REGISTRATION_PATH",
    "REPLAY_PATH",
    "REPORT_ARM_NAMES",
    "REPO_ROOT",
    "RESULT_PATH",
    "ROWS_PATH",
    "SHUFFLED_ARM",
    "SOURCE_BARS",
    "SOURCE_QUERY_POPULATIONS",
    "directory_tree_record",
    "read_json",
    "relative_hashes",
    "sha256_bytes",
    "sha256_file",
    "stable_json",
    "stable_sha256",
    "verify_model_file",
    "write_json",
]

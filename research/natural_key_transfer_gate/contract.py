"""Frozen paths, constants, and hashing helpers for NKT-1."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
SNAPSHOT_ROOT = REPO_ROOT / "research" / "validated_memory_primitives"

PREREG_PATH = PACKAGE_ROOT / "docs" / "natural-key-transfer-prereg-2026-08-10.md"
MANIFEST_PATH = PACKAGE_ROOT / "data" / "massive_development_manifest.json"
PREFLIGHT_PATH = PACKAGE_ROOT / "preflight.json"
REGISTRATION_PATH = PACKAGE_ROOT / "development_registration.json"
RESULT_PATH = PACKAGE_ROOT / "results" / "natural_key_transfer_development_result.json"
ROWS_PATH = PACKAGE_ROOT / "results" / "natural_key_transfer_development_rows.jsonl"

DEFAULT_DATA_PATH = Path(r"D:\HLM2\runtime\datasets\massive-1.1\1.1\data\en-US.jsonl")
DEFAULT_LICENSE_PATH = Path(r"D:\HLM2\runtime\datasets\massive-1.1\1.1\LICENSE")
DEFAULT_CHECKPOINT_PATH = Path(
    r"D:\HLM2\artifacts\models\demo\hlm5-136m-fineweb-g2-2026-06-10\model.pt"
)
DEFAULT_TOKENIZER_PATH = Path(
    r"D:\HLM2\artifacts\models\demo\hlm5-136m-fineweb-g2-2026-06-10\tokenizer.json"
)
DEFAULT_CACHE_PATH = Path(r"D:\HLM2\runtime\nkt1\massive-development-hiddens.pt")

EXPECTED_SHA256 = {
    "data": "c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e",
    "license": "c2e6ea015269147de02117ebdd91f30ef09831251f5345fa8365273b1db1d435",
    "checkpoint": "2addfa88808be6847e29f70c9182093c15ff420c168f9dd8b41b72c4757a8eda",
    "tokenizer": "15993635191a1c5f1a5dc7aeaacbdf9a44a45d90abef954fc77b686f4fbbe588",
}

TARGET_DOMAIN_SALT = "hlm5-natural-key-transfer-v1"
SUPPORT_SALT = "nkt1-support-v1"
SHUFFLE_SALT = "nkt1-shuffle-v1"
TARGET_DOMAIN_COUNT = 6
SUPPORT_PER_INTENT = 3
DEGREE = 5
SHRINKAGE = 0.05
BOOTSTRAP_SEED = 20260810
BOOTSTRAP_RESAMPLES = 10_000
DEPLOYED_TAU_COS = 0.95

PASS_BARS = {
    "candidate_minus_raw": 0.020,
    "candidate_minus_blind_zca": 0.010,
    "candidate_minus_shuffled_within": 0.010,
    "interval_lower_strictly_greater_than": 0.0,
    "candidate_max_predicted_intent_share": 0.100,
    "candidate_auroc_drop_vs_raw": 0.010,
}

IMPLEMENTATION_PATHS = (
    PACKAGE_ROOT / "__init__.py",
    PACKAGE_ROOT / "contract.py",
    PACKAGE_ROOT / "prepare_massive.py",
    PACKAGE_ROOT / "geometry.py",
    PACKAGE_ROOT / "run_natural_key_transfer_gate.py",
    PACKAGE_ROOT / "tests" / "test_prepare_massive.py",
    PACKAGE_ROOT / "tests" / "test_geometry.py",
    PACKAGE_ROOT / "tests" / "test_registration.py",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_sha256(value: Any) -> str:
    return sha256_bytes(stable_json(value).encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def file_hashes(paths: tuple[Path, ...] = IMPLEMENTATION_PATHS) -> dict[str, str]:
    return {path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in paths}


def verify_expected_file(path: Path, key: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    expected = EXPECTED_SHA256[key]
    if observed != expected:
        raise RuntimeError(
            f"{key} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return observed

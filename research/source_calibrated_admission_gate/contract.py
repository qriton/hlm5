"""Frozen paths, constants, and hashing helpers for SCA-1."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
SNAPSHOT_ROOT = REPO_ROOT / "research" / "validated_memory_primitives"

PREREG_PATH = PACKAGE_ROOT / "docs" / "source-calibrated-admission-prereg-2026-08-10.md"
MANIFEST_PATH = PACKAGE_ROOT / "data" / "topv2_development_manifest.json"
PREFLIGHT_PATH = PACKAGE_ROOT / "preflight.json"
REGISTRATION_PATH = PACKAGE_ROOT / "development_registration.json"
RESULT_PATH = (
    PACKAGE_ROOT / "results" / "source_calibrated_admission_development_result.json"
)
ROWS_PATH = (
    PACKAGE_ROOT / "results" / "source_calibrated_admission_development_rows.jsonl"
)

DEFAULT_DATA_ROOT = Path(r"D:\HLM2\runtime\datasets\topv2-1.1\official")
DEFAULT_CHECKPOINT_PATH = Path(
    r"D:\HLM2\artifacts\models\demo\hlm5-136m-fineweb-g2-2026-06-10\model.pt"
)
DEFAULT_TOKENIZER_PATH = Path(
    r"D:\HLM2\artifacts\models\demo\hlm5-136m-fineweb-g2-2026-06-10\tokenizer.json"
)
DEFAULT_CACHE_PATH = Path(r"D:\HLM2\runtime\sca1\topv2-development-hiddens.pt")

EXPECTED_MODEL_SHA256 = {
    "checkpoint": "2addfa88808be6847e29f70c9182093c15ff420c168f9dd8b41b72c4757a8eda",
    "tokenizer": "15993635191a1c5f1a5dc7aeaacbdf9a44a45d90abef954fc77b686f4fbbe588",
}

EXPECTED_MANIFEST_SHA256 = {
    "file": "e2575806939206cca4c2ffe506d00f9894d3c9b2c70463082a1a5a7dae243159",
    "scientific": "90293546a57991bf046df32a87e20798b8da089bbe5c0fe88d07b98fcea8682c",
}
EXPECTED_RANDOM_BASIS_SHA256 = (
    "3d24eb4803ae658587e314df46dc48a22850c76df1559598849a06f054945683"
)

EXPECTED_DATA_SHA256 = {
    "alarm_eval.tsv": "9b80c3b3430c6378fa566f76f6ef5aa31156599f362a8896922839ebeedaa4a3",
    "alarm_test.tsv": "96ce58ce1944f2f94b37477a18aa31b70b54aad0505eb7688d613fbd6415818c",
    "alarm_train.tsv": "920483ba7ae051f95f69692deb77c80e66a105005ed5e49a3a043e9ffd155db5",
    "event_eval.tsv": "1fb48c8aeb5bf3c6bb35ed206979adb2c8ef55fe7b0caf49424ebd75041e05d8",
    "event_test.tsv": "9b67d4ed5cfe63c7899be66f523ec33de04352e6a16c35b9c2a5c322ec4548fc",
    "event_train.tsv": "8c81d6680e7b72ca43be52702654839400678025ac1e6df9f352cb348180bdd8",
    "LICENSE": "3b2890eacd851373001c4a14623458e3adaf1b1967939aa9c38a318e28d61c00",
    "messaging_eval.tsv": "a9f8761b0add1e8e1b46a8a53915b5101e48135983f199948cef9fed346e6b63",
    "messaging_test.tsv": "e3d25346286bb9a514b325e18ec0045cef2eb58f45273d1deff088aa677cefec",
    "messaging_train.tsv": "1f23f321ce36884d05931ca78eeb0823aaae13047d8099ba68e32e8656c2cf02",
    "music_eval.tsv": "27d3ce7a63b13a91a1dd86292a269ff26e20c7895cd9789da2cc6f0a03de67cc",
    "music_test.tsv": "c080a8f350cf5df6642d0f0e6e0a19c5ac682a8d9c912a0b57a5f0d1f01b44a0",
    "music_train.tsv": "0abb87d11edc98d4be9b20879ec892d3fdcb0cd8d2bf4da06eef5e997c578d3c",
    "navigation_eval.tsv": "308c95b88b9bad1d5cd5d3904f702dc03160829ada32f7eaff131d60f85588b7",
    "navigation_test.tsv": "bf1bd7915932fac8452149f8b69f290e253b3eb1574e1557a7f7a9a156174fd2",
    "navigation_train.tsv": "b486903314e4d2f011eeeef95f31f1eccdd8bf7aa538a2b003a23f9dd44b0b5a",
    "README": "5f050132aee99d4901793f9a9a47bc37aa7d68266b44f5c41f0eedbb94da62ff",
    "reminder_eval.tsv": "dea5c9deb676676eeeede7036aa07f664f9d2e5a6b37b7fd8f3d2c6b171ecebf",
    "reminder_test.tsv": "e6bc68ac371e7755753f455dcef5e42d4bf960c08f72beb094e4bc0940b7f925",
    "reminder_train.tsv": "689f0934456b9b85d8d0d256425b86362f2bd349756d531a71958edd1b6e6ac1",
    "timer_eval.tsv": "463bfeef6621b90a1c471916653fc420c219f8f65b38359a442485caa9e794a9",
    "timer_test.tsv": "aa04c18cce9d02456f2a21f5fe8b8088a3470529b88db5ee35640110bddd74c9",
    "timer_train.tsv": "d630abdde12cade9230434714f3f2f0ca8f75c8c00a6b6a5f5dfeb3100cf996f",
    "weather_eval.tsv": "f6672529ab85c1415cd5a5d3045551176799605fba9e132fdd9d11adb93b908e",
    "weather_test.tsv": "e4aa02ae0d3bad627c499d58365f34f37939153c832d30ee90725b745d74e9df",
    "weather_train.tsv": "fd683b775602ec88d3fc7ca442c060ff33aacb710eaed3e517c8691835ae1018",
}

SOURCE_DOMAINS = ("alarm", "event", "messaging", "music", "navigation", "timer")
TARGET_DOMAINS = ("reminder", "weather")
PARTITIONS = ("train", "eval", "test")

SUPPORT_PER_INTENT = 3
ROWS_PER_INTENT = 20
SOURCE_FIT_PER_DOMAIN = 2_000
GROUP_INTENT_COUNT = 9
DEGREE = 5
SHRINKAGE = 0.05
RANDOM_BASIS_SEED = 20260811
BOOTSTRAP_SEED = 20260811
BOOTSTRAP_RESAMPLES = 10_000
CALIBRATION_NEGATIVE_QUANTILE = 0.95
CALIBRATION_QUANTILE_METHOD = "higher"

INTENT_ORDER_SALT = "sca1-source-intent-order-v1"
ROW_ORDER_SALT = "sca1-row-order-v1"
SOURCE_FIT_SALT = "sca1-source-fit-v1"

ARM_NAMES = (
    "raw",
    "centered",
    "diagonal_std",
    "random_basis_zca",
    "blind_zca",
)
CANDIDATE_ARM = "blind_zca"

PASS_BARS = {
    "candidate_correct_admission_macro": 0.20,
    "candidate_minus_raw_correct_admission_macro": 0.03,
    "candidate_minus_diagonal_correct_admission_macro": 0.015,
    "candidate_minus_random_basis_correct_admission_macro": 0.015,
    "paired_interval_lower_strictly_greater_than": 0.0,
    "candidate_target_coverage_macro": 0.30,
    "candidate_selective_accuracy": 0.65,
    "candidate_off_support_false_admission_rate": 0.05,
    "candidate_max_predicted_intent_share": 0.25,
    "candidate_top1_accuracy_drop_vs_raw": 0.01,
}

IMPLEMENTATION_PATHS = (
    PACKAGE_ROOT / "__init__.py",
    PACKAGE_ROOT / "contract.py",
    PACKAGE_ROOT / "prepare_topv2.py",
    PACKAGE_ROOT / "geometry.py",
    PACKAGE_ROOT / "run_source_calibrated_admission_gate.py",
    PACKAGE_ROOT / "tests" / "test_prepare_topv2.py",
    PACKAGE_ROOT / "tests" / "test_geometry.py",
    PACKAGE_ROOT / "tests" / "test_registration.py",
)

DEPENDENCY_PATHS = (
    REPO_ROOT / "research" / "natural_key_transfer_gate" / "contract.py",
    REPO_ROOT / "research" / "natural_key_transfer_gate" / "geometry.py",
    REPO_ROOT / "research" / "natural_key_transfer_gate" / "prepare_massive.py",
    REPO_ROOT
    / "research"
    / "natural_key_transfer_gate"
    / "run_natural_key_transfer_gate.py",
    SNAPSHOT_ROOT / "hlm5_lm.py",
    SNAPSHOT_ROOT / "hlm5_memory.py",
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


def relative_hashes(paths: tuple[Path, ...]) -> dict[str, str]:
    return {path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in paths}


def verify_data_files(data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, expected in EXPECTED_DATA_SHA256.items():
        path = data_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != expected:
            raise RuntimeError(
                f"TOPv2 file hash mismatch for {name}: expected {expected}, observed {digest}"
            )
        observed[name] = digest
    return observed


def verify_model_file(path: Path, key: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    expected = EXPECTED_MODEL_SHA256[key]
    if observed != expected:
        raise RuntimeError(
            f"{key} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return observed

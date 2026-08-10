"""Frozen paths, constants, and hashing helpers for SCA-2."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
SNAPSHOT_ROOT = REPO_ROOT / "research" / "validated_memory_primitives"

PREREG_PATH = PACKAGE_ROOT / "docs" / "two-axis-admission-prereg-2026-08-10.md"
MANIFEST_PATH = PACKAGE_ROOT / "data" / "hwu64_development_manifest.json"
PREFLIGHT_PATH = PACKAGE_ROOT / "preflight.json"
REGISTRATION_PATH = PACKAGE_ROOT / "development_registration.json"
RESULT_PATH = PACKAGE_ROOT / "results" / "two_axis_admission_development_result.json"
ROWS_PATH = PACKAGE_ROOT / "results" / "two_axis_admission_development_rows.jsonl"
REPLAY_PATH = PACKAGE_ROOT / "results" / "two_axis_admission_replay_receipt.json"

DEFAULT_DATA_ROOT = Path(r"D:\HLM2\runtime\datasets\hwu64-official")
DEFAULT_CHECKPOINT_PATH = Path(
    r"D:\HLM2\artifacts\models\demo\hlm5-136m-fineweb-g2-2026-06-10\model.pt"
)
DEFAULT_TOKENIZER_PATH = Path(
    r"D:\HLM2\artifacts\models\demo\hlm5-136m-fineweb-g2-2026-06-10\tokenizer.json"
)
DEFAULT_SOURCE_CACHE_PATH = Path(
    r"D:\HLM2\runtime\sca2\hwu64-source-audit-hiddens.pt"
)
DEFAULT_TARGET_CACHE_PATH = Path(
    r"D:\HLM2\runtime\sca2\hwu64-target-development-hiddens.pt"
)

PREREG_COMMIT = "56a3cb1818f814d0a8cae589512027228b8a2723"
EXPECTED_PREREG_SHA256 = (
    "83f3cefbfbf3a5f172940095d30a2da94e73669c16c49d040fdcda9a88fd84d2"
)

DATASET_GIT_COMMIT = "f6071b496b17d71e6eb43f543af0707f4ff30557"
DATASET_SCRIPTS_GIT_COMMIT = "356711b59f347532d0290f070ff9aad5af7ed02e"
RASA_CV_ROOT = Path(
    "CrossValidation"
    "/out4RasaReal"
    "/rasa_json_2018_03_22-13_01_25_169_80Train"
    "/CrossValidation"
)

EXPECTED_MODEL_SHA256 = {
    "checkpoint": "2addfa88808be6847e29f70c9182093c15ff420c168f9dd8b41b72c4757a8eda",
    "tokenizer": "15993635191a1c5f1a5dc7aeaacbdf9a44a45d90abef954fc77b686f4fbbe588",
}

EXPECTED_DATA_SHA256 = {
    "LICENSE": "d07988f4be47912f75a73b44da8a9c0b602bf6c8ced69de858b848f379ecd973",
    "README.md": "cb424dede0ac86135c6a5198523df8b8be7efe3fdd0053a12f4946d3199f3370",
    "fold_01": "0b0396e8029376b4860a094eb381a93aa41800f0331152ee91f9e8c71ea2c5c0",
    "fold_02": "e246f1b430f21e6c0ff17d606271fac2f472585d5794592ac701e1fcba3ed7da",
    "fold_03": "361ab24fae47386b9d264f834d7458ce64a83235cf1ee30cf71e29583f4329ed",
    "fold_04": "f75087940ff9ffcaac79c95dad709b99ac0118416cfaaf53d889186acf1e55ab",
    "fold_05": "e31c8abbff904d568c5c257d8e66bd5b326c75df92b4e4286191f66afa1f7cc8",
    "fold_06": "18626d98b9372e65b84024e89031efb5e3ee37a1cd151609c6ade232c02deb72",
    "fold_07": "1bff32c7c74d9e06b92209ac5afa9456cae7a1e2dfe4231c8aaa76b3bc7f3be9",
    "fold_08": "16b3052352642d6988dd625e0d0a6f6adc0075b47525689fbe91acbdac24136b",
    "fold_09": "2d9a5ed6c88cf759b3574d40105ee3e0e37b65328990bb1cc52217f5bb6cac6b",
    "fold_10": "2620910fc909d85e5072c49f0a2cfd8afcd2b9eea628020357b69409073f8177",
}

EXPECTED_MANIFEST_SHA256 = {
    "file": "ae3af9eadb6511501a1c8695376b04c6176e3be8e465a68e07b0fd18477ae576",
    "scientific": "43ca391013300dd88a76c7e4a17b33503cb6682e295d5eac4071a50387590fd0",
}
EXPECTED_RANDOM_BASIS_SHA256 = (
    "3d24eb4803ae658587e314df46dc48a22850c76df1559598849a06f054945683"
)

DEVELOPMENT_FOLD = 1
TEST_FOLD = 2
TRAIN_FOLDS = tuple(range(3, 11))
SUPPORT_PER_INTENT = 3
ROWS_PER_INTENT = 8
GROUP_INTENT_COUNT = 12
GROUP_NAMES = (
    "calibration_supported",
    "calibration_negative",
    "audit_supported",
    "audit_negative",
    "target_supported",
)
INTENT_ORDER_SALT = "sca2-hwu64-intent-groups-v1"
ROW_ORDER_SALT = "sca2-hwu64-row-order-v1"

DEGREE = 5
SHRINKAGE = 0.05
RANDOM_BASIS_SEED = 20260811
BOOTSTRAP_SEED = 20260812
BOOTSTRAP_RESAMPLES = 10_000
ABSOLUTE_NEGATIVE_QUANTILE = 0.95
MARGIN_ERROR_QUANTILE = 0.90
QUANTILE_METHOD = "higher"
MINIMUM_MARGIN_CALIBRATION_ERRORS = 8

ARM_NAMES = (
    "raw",
    "centered",
    "diagonal_std",
    "random_basis_zca",
    "blind_zca",
)
CANDIDATE_ARM = "blind_zca"

AUDIT_BARS = {
    "candidate_macro_coverage": 0.20,
    "candidate_macro_correct_admission": 0.15,
    "candidate_selective_accuracy": 0.65,
    "candidate_selective_gain_vs_absolute_only": 0.05,
    "candidate_correct_admission_retention_vs_absolute_only": 0.60,
    "candidate_off_support_false_admission_rate": 0.05,
}

PASS_BARS = {
    "candidate_macro_coverage": 0.20,
    "candidate_macro_correct_admission": 0.15,
    "candidate_selective_accuracy": 0.65,
    "candidate_selective_gain_vs_absolute_only": 0.075,
    "candidate_selective_gain_vs_matched_proximity": 0.05,
    "candidate_correct_admission_retention_vs_absolute_only": 0.60,
    "candidate_off_support_false_admission_rate": 0.05,
    "candidate_max_predicted_intent_share": 0.25,
    "candidate_top1_accuracy_drop_vs_raw": 0.01,
}

IMPLEMENTATION_PATHS = (
    PACKAGE_ROOT / "__init__.py",
    PACKAGE_ROOT / "contract.py",
    PACKAGE_ROOT / "prepare_hwu64.py",
    PACKAGE_ROOT / "geometry.py",
    PACKAGE_ROOT / "run_two_axis_admission_gate.py",
    PACKAGE_ROOT / "tests" / "test_prepare_hwu64.py",
    PACKAGE_ROOT / "tests" / "test_geometry.py",
    PACKAGE_ROOT / "tests" / "test_registration.py",
)

DEPENDENCY_PATHS = (
    REPO_ROOT / "research" / "natural_key_transfer_gate" / "contract.py",
    REPO_ROOT / "research" / "natural_key_transfer_gate" / "geometry.py",
    REPO_ROOT
    / "research"
    / "natural_key_transfer_gate"
    / "run_natural_key_transfer_gate.py",
    REPO_ROOT / "research" / "source_calibrated_admission_gate" / "geometry.py",
    SNAPSHOT_ROOT / "hlm5_lm.py",
    SNAPSHOT_ROOT / "hlm5_memory.py",
)


def fold_test_relative_path(fold: int) -> Path:
    if fold not in range(1, 11):
        raise ValueError(f"fold must be in 1..10, got {fold}")
    return (
        RASA_CV_ROOT
        / f"KFold_{fold}"
        / "mergedTestset"
        / "RasaNluTestset_Merged_2018_03_22-13_01_25_169.json"
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
    try:
        dataset_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=data_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"HWU64 source is not a readable Git checkout: {data_root}"
        ) from error
    if dataset_head != DATASET_GIT_COMMIT:
        raise RuntimeError(
            "HWU64 Git commit mismatch: "
            f"expected {DATASET_GIT_COMMIT}, observed {dataset_head}"
        )
    paths = {
        "LICENSE": data_root / "LICENSE",
        "README.md": data_root / "README.md",
        **{
            f"fold_{fold:02d}": data_root / fold_test_relative_path(fold)
            for fold in range(1, 11)
        },
    }
    observed: dict[str, str] = {}
    for name, expected in EXPECTED_DATA_SHA256.items():
        path = paths[name]
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != expected:
            raise RuntimeError(
                f"HWU64 file hash mismatch for {name}: expected {expected}, observed {digest}"
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

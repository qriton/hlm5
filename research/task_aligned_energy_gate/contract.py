"""Immutable contract for the Banking77 task-aligned energy gate."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOC_PATH = ROOT / "docs" / "task-aligned-energy-prereg-2026-08-10.md"
DEVELOPMENT_MANIFEST_PATH = DATA_DIR / "banking77_development_manifest.json"
TEST_MANIFEST_PATH = DATA_DIR / "banking77_test_manifest.json"
PREFLIGHT_PATH = ROOT / "preflight.json"
REGISTRATION_PATH = ROOT / "development_registration.json"
DEV_ROWS_PATH = ROOT / "results" / "task_aligned_energy_development_rows.jsonl"
DEV_RESULT_PATH = ROOT / "results" / "task_aligned_energy_development_result.json"
DEV_VERDICT_PATH = (
    ROOT / "docs" / "task-aligned-energy-development-verdict-2026-08-10.md"
)
TEST_REGISTRATION_PATH = ROOT / "test_registration.json"
TEST_ROWS_PATH = ROOT / "results" / "task_aligned_energy_test_rows.jsonl"
TEST_RESULT_PATH = ROOT / "results" / "task_aligned_energy_test_result.json"
TEST_VERDICT_PATH = ROOT / "docs" / "task-aligned-energy-test-verdict-2026-08-10.md"

EXPERIMENT_ID = "hlm5-b77-task-aligned-energy-v1"
SOURCE_REPOSITORY = "https://github.com/PolyAI-LDN/task-specific-datasets"
SOURCE_COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE_HASHES = {
    "banking_data/train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "banking_data/test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    "banking_data/categories.json": "fa9960a8cf80ec3ae345a3dd71345e178523597eccf03396401d650d85eea1a1",
    "LICENSE": "48a83a6e39f7b2f166763b30776132c9a99aa816f17cb06f87ad5b8542a7b71f",
}
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SPLIT_SALT = "hlm5-b77-task-aligned-energy-v1-split"

SCIENTIFIC_CONSTANTS: dict[str, object] = {
    "experiment_id": EXPERIMENT_ID,
    "intent_count": 77,
    "official_train_count": 10_003,
    "deduplicated_train_count": 9_999,
    "normalized_train_duplicate_groups": 4,
    "fit_count": 8_459,
    "calibration_per_intent": 10,
    "calibration_count": 770,
    "development_per_intent": 10,
    "development_count": 770,
    "official_test_count": 3_080,
    "primary_test_count": 3_073,
    "normalized_train_test_overlap_count": 7,
    "projection_rank": 76,
    "shrinkage_grid": [0.05, 0.25, 0.5, 0.75, 1.0],
    "energy_temperature": 1.0,
    "steps": 8,
    "one_step_steps": 1,
    "initial_step_radius_fraction": 0.5,
    "armijo_c": 1e-4,
    "max_halvings": 24,
    "stationary_tolerance": 1e-12,
    "energy_tolerance": 1e-12,
    "cap_tolerance": 1e-10,
    "matched_state_tolerance": 1e-10,
    "bootstrap_resamples": 10_000,
    "seed": 20_260_810,
    "alignment_gain_bar": 0.020,
    "alignment_interval_lower_bar": 0.0,
    "dynamics_static_gain_bar": 0.010,
    "dynamics_static_interval_lower_bar": 0.0,
    "dynamics_one_step_gain_bar": 0.005,
    "dynamics_one_step_interval_lower_bar": 0.0,
    "maximum_intent_share_bar": 0.050,
    "shuffled_label_macro_max": 0.030,
    "dtype": "torch.float64",
    "device": "cpu",
    "encoder_batch_size": 128,
    "dynamics_batch_size": 128,
}

HASHED_IMPLEMENTATION_PATHS = (
    ROOT / "README.md",
    ROOT / "DATA_NOTICE.md",
    DOC_PATH,
    ROOT / "contract.py",
    ROOT / "prepare_banking77.py",
    ROOT / "energy.py",
    ROOT / "metrics.py",
    ROOT / "run_task_aligned_energy_gate.py",
    ROOT / "tests" / "test_prepare_banking77.py",
    ROOT / "tests" / "test_energy.py",
    ROOT / "tests" / "test_registration.py",
    DEVELOPMENT_MANIFEST_PATH,
    TEST_MANIFEST_PATH,
)

DEV_REGISTERED_COMMAND = "python -B run_task_aligned_energy_gate.py --run-development"
TEST_REGISTERED_COMMAND = "python -B run_task_aligned_energy_gate.py --run-test"

"""Immutable contract for the HLM5 I1 intent-basin gate."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOC_PATH = ROOT / "docs" / "intent-basin-gate-prereg-2026-08-10.md"
DEVELOPMENT_MANIFEST_PATH = DATA_DIR / "clinc_development_manifest.json"
TEST_MANIFEST_PATH = DATA_DIR / "clinc_test_manifest.json"

PREFLIGHT_PATH = ROOT / "preflight.json"
DEV_REGISTRATION_PATH = ROOT / "development_registration.json"
DEV_RAW_ROWS_PATH = ROOT / "results" / "intent_basin_development_rows.jsonl"
DEV_RESULT_PATH = ROOT / "results" / "intent_basin_development_result.json"
DEV_VERDICT_PATH = ROOT / "docs" / "intent-basin-development-verdict-2026-08-10.md"
TEST_REGISTRATION_PATH = ROOT / "test_registration.json"
TEST_RAW_ROWS_PATH = ROOT / "results" / "intent_basin_test_rows.jsonl"
TEST_RESULT_PATH = ROOT / "results" / "intent_basin_test_result.json"
TEST_VERDICT_PATH = ROOT / "docs" / "intent-basin-test-verdict-2026-08-10.md"

EXPERIMENT_ID = "hlm5-i1-class-balanced-intent-basin-v1"
SOURCE_REPOSITORY = "https://github.com/clinc/oos-eval"
SOURCE_COMMIT = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
SOURCE_DATA_SHA256 = (
    "1e53b1322dac050a0a601e4ec5157d510633202de752a747dfd9d4f7cf9c1d5c"
)
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

SCIENTIFIC_CONSTANTS: dict[str, int | float | str] = {
    "experiment_id": EXPERIMENT_ID,
    "intent_count": 150,
    "train_per_intent": 100,
    "validation_per_intent": 20,
    "test_per_intent": 30,
    "validation_in_scope_count": 2997,
    "test_in_scope_count": 4498,
    "validation_train_overlap_exclusions": 3,
    "test_train_overlap_exclusions": 2,
    "validation_oos_count": 100,
    "test_oos_count": 1000,
    "prototype_blocks_per_intent": 4,
    "prototype_block_size": 25,
    "prototype_count": 600,
    "degree": 5,
    "steps": 8,
    "one_step_steps": 1,
    "initial_step_radius_fraction": 0.5,
    "armijo_c": 1e-4,
    "max_halvings": 24,
    "stationary_tolerance": 1e-12,
    "energy_tolerance": 1e-12,
    "unit_tolerance": 1e-10,
    "cap_tolerance": 1e-10,
    "matched_state_tolerance": 1e-10,
    "bootstrap_resamples": 10_000,
    "seed": 20_260_810,
    "pass_static_gain": 0.020,
    "pass_static_bootstrap_lower": 0.0,
    "pass_one_step_gain": 0.005,
    "pass_one_step_bootstrap_lower": 0.0,
    "pass_auc_noninferiority": -0.010,
    "pass_max_intent_share": 0.030,
    "valid_shuffled_macro_max": 0.020,
    "dtype": "torch.float64",
    "device": "cpu",
    "encoder_batch_size": 128,
    "dynamics_batch_size": 64,
}

HASHED_IMPLEMENTATION_PATHS = (
    ROOT / "README.md",
    ROOT / "DATA_NOTICE.md",
    DOC_PATH,
    ROOT / "dynamics.py",
    ROOT / "contract.py",
    ROOT / "metrics.py",
    ROOT / "prepare_clinc.py",
    ROOT / "run_intent_basin_gate.py",
    ROOT / "tests" / "test_dynamics.py",
    ROOT / "tests" / "test_prepare_clinc.py",
    ROOT / "tests" / "test_registration.py",
    DEVELOPMENT_MANIFEST_PATH,
    TEST_MANIFEST_PATH,
)

DEV_REGISTERED_COMMAND = "python -B run_intent_basin_gate.py --run-development"
TEST_REGISTERED_COMMAND = "python -B run_intent_basin_gate.py --run-test"

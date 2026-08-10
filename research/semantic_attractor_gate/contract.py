"""Immutable scientific constants for the HLM5 S1 registered assay."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOC_PATH = ROOT / "docs" / "semantic-attractor-gate-prereg-2026-08-10.md"
MANIFEST_PATH = DATA_DIR / "pararel_manifest.json"
REGISTRATION_PATH = ROOT / "registration.json"
PREFLIGHT_PATH = ROOT / "preflight.json"
RAW_ROWS_PATH = ROOT / "results" / "semantic_attractor_gate_rows.jsonl"
RESULT_PATH = ROOT / "results" / "semantic_attractor_gate_result.json"
VERDICT_PATH = ROOT / "docs" / "semantic-attractor-gate-verdict-2026-08-10.md"

EXPERIMENT_ID = "hlm5-s1-certified-semantic-attractor-v1"
PARAREL_REPOSITORY = "https://github.com/yanaiela/pararel"
PARAREL_COMMIT = "cb5554678457beb5ac163d888f1ce8cf174b3f0b"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

SCIENTIFIC_CONSTANTS: dict[str, int | float | str] = {
    "experiment_id": EXPERIMENT_ID,
    "eligible_relation_count": 38,
    "pattern_count": 328,
    "facts_per_relation": 8,
    "stored_address_count": 304,
    "supported_query_count": 2320,
    "off_support_query_count": 2320,
    "mask_literal": "something",
    "degree": 5,
    "steps": 8,
    "initial_step": 1.0,
    "armijo_c": 1e-4,
    "max_halvings": 24,
    "stationary_tolerance": 1e-12,
    "energy_tolerance": 1e-12,
    "unit_tolerance": 1e-10,
    "matched_state_tolerance": 1e-10,
    "bootstrap_resamples": 10_000,
    "seed": 20_260_810,
    "pass_accuracy_gain": 0.050,
    "pass_bootstrap_lower": 0.0,
    "pass_auc_noninferiority": -0.010,
    "pass_max_relation_share": 0.150,
    "valid_shuffled_macro_max": 0.050,
    "dtype": "torch.float64",
    "device": "cpu",
    "encoder_batch_size": 64,
    "dynamics_batch_size": 64,
}

HASHED_IMPLEMENTATION_PATHS = (
    ROOT / "README.md",
    DOC_PATH,
    ROOT / "attractor.py",
    ROOT / "contract.py",
    ROOT / "metrics.py",
    ROOT / "prepare_pararel.py",
    ROOT / "run_semantic_attractor_gate.py",
    ROOT / "tests" / "test_attractor.py",
    ROOT / "tests" / "test_prepare_pararel.py",
    ROOT / "tests" / "test_registration.py",
    MANIFEST_PATH,
)

REGISTERED_COMMAND = "python -B run_semantic_attractor_gate.py --run-registered"

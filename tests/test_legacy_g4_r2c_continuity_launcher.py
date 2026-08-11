"""Static preregistration contracts for the unsubmitted R2c continuity assay."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm" / "legacy_g4_hybrid_522k_exact_resume_continuity.slurm"
TRAINER = ROOT / "research" / "legacy_g4_recovery" / "train_hlm5_lm_fineweb_stateful.py"
COMPARATOR = ROOT / "research" / "legacy_g4_recovery" / "compare_stateful_shards.py"
PROTOCOL = ROOT / "docs" / "legacy-g4-hybrid-r2c-exact-resume-contract-2026-08-11.md"


def launcher_text() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_assay_is_paired_bounded_and_uses_the_original_schedule() -> None:
    text = launcher_text()
    assert "#SBATCH --time=02:00:00" in text
    assert "schedule_horizon=610000" in text
    assert "source_step=522000" in text
    assert "midpoint_step=522500" in text
    assert "target_step=523000" in text
    assert "cadence=500" in text
    assert 'run_segment continuous "$continuous_dir" "$source_step" "$target_step"' in text
    assert 'run_segment split_pre_restart "$split_dir" "$source_step" "$midpoint_step"' in text
    assert 'run_segment split_post_restart "$split_dir" "$midpoint_step" "$target_step"' in text
    assert '--steps "$schedule_horizon"' in text
    assert '--save-every "$cadence"' in text
    assert '--eval-every "$cadence"' in text


def test_midpoint_control_precedes_the_restart_treatment() -> None:
    text = launcher_text()
    midpoint_compare = text.index("run_comparison midpoint")
    post_restart = text.index("run_segment split_post_restart")
    assert midpoint_compare < post_restart
    assert "INVALID_REPRODUCIBILITY" in text
    assert "FAIL_CONTINUITY" in text
    assert "PASS_CONTINUITY" in text
    assert "midpoint_ppl_exact" in text
    assert "midpoint_marker_exact" in text
    assert "final_ppl_exact" in text
    assert "final_marker_exact" in text


def test_every_segment_requires_complete_state_without_a_new_bridge() -> None:
    text = launcher_text()
    function = text[text.index("run_segment()") : text.index("extract_ppl()")]
    assert "--stateful-checkpoints" in function
    assert "--require-complete-state" in function
    assert '--expected-resume-step "$expected_resume"' in function
    assert "--restart-warmup-steps" not in function
    assert "--legacy-sampling-rng-offset" not in function
    assert "--no-prune-checkpoints" in function


def test_source_is_immutable_and_outputs_use_project_work_storage() -> None:
    text = launcher_text()
    assert "work_parent=/leonardo_work/AIFAC_L14_039" in text
    assert "minimum_free_bytes=260000000000" in text
    assert "43294668998" in text
    assert "e32a2b0945e2197cc080d585bd16c7a33a817d63a6e5d96fb436127f8fdecae8" in text
    assert "SOURCE_CHANGED_DURING_R2C" in text
    assert 'ln -s "$source_checkpoint"' in text
    assert re.search(r"^\s*rm\s", text, flags=re.MULTILINE) is None
    assert re.search(r"^\s*mv\s", text, flags=re.MULTILINE) is None


def test_registration_and_single_use_are_fail_closed() -> None:
    text = launcher_text()
    assert "R2C_REGISTERED_LAUNCHER_SHA256_REQUIRED" in text
    assert "R2C_REGISTERED_PROTOCOL_SHA256_REQUIRED" in text
    assert "R2C_WORK_ROOT_ALREADY_EXISTS" in text
    assert "CREATE_NEW_FAILED" in text
    assert "COMPARISON_IMPLEMENTATION_INVALID" in text
    assert 'environment_sha256' in text
    assert 'allow_bf16_reduced_precision_reduction' in text
    assert "trap record_failure_artifact ERR" in text
    assert text.index('create_new_json "$attempt"') < text.index(
        'run_segment continuous "$continuous_dir"'
    )


def test_launcher_pins_the_exact_trainer_and_comparator() -> None:
    text = launcher_text()
    trainer_sha = hashlib.sha256(TRAINER.read_bytes()).hexdigest()
    comparator_sha = hashlib.sha256(COMPARATOR.read_bytes()).hexdigest()
    assert trainer_sha == "1f10b45c9d35bbaba346e07756c38edb681ea9bb8daee1c4c08ab4d2e58aa0c4"
    assert f"comparator_sha_expected={comparator_sha}" in text
    assert f"trainer_sha_expected={trainer_sha}" in text


def test_protocol_states_the_bounded_claim_and_paid_boundary() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "No R2c" in text and "outcome has been accessed" in text
    assert "midpoint is a pre-intervention reproducibility control" in text
    assert "does not authorize that paid ladder" in text
    assert "Submission requires a new explicit user approval" in text

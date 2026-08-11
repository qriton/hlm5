"""Static safety contract for the unsubmitted legacy G4 R2b bridge."""

import hashlib

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm" / "legacy_g4_hybrid_520k_stateful_bridge.slurm"
TRAINER = (
    ROOT
    / "research"
    / "legacy_g4_recovery"
    / "train_hlm5_lm_fineweb_stateful.py"
)


def launcher_text() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_bridge_is_bounded_and_preserves_the_historical_schedule() -> None:
    text = launcher_text()
    assert "#SBATCH --time=02:00:00" in text
    assert "#SBATCH --dependency=singleton" in text
    assert "original_horizon=610000" in text
    assert '--steps "$original_horizon"' in text
    assert "--save-every 2000" in text
    assert "--eval-every 2000" in text
    assert "--expected-resume-step 520000" in text
    assert "--no-prune-checkpoints" in text
    assert 'target_marker="$out_dir/ckpt_step522000.sharded"' in text
    assert 'scancel --signal=TERM "$trainer_step"' in text
    assert "BRIDGE_OVERRAN_TARGET" in text


def test_bridge_uses_one_frozen_restart_intervention() -> None:
    text = launcher_text()
    assert "sampling_rng_offset=108000" in text
    assert "restart_warmup_steps=2000" in text
    assert "restart_warmup_initial_factor=0.1" in text
    assert '--legacy-sampling-rng-offset "$sampling_rng_offset"' in text
    assert '--restart-warmup-steps "$restart_warmup_steps"' in text
    assert '--restart-warmup-initial-factor "$restart_warmup_initial_factor"' in text
    assert '"sampling_rng_reconstructed":true' in text
    assert "1fddeb1b7259f70bb9936e595df427e2e84ecb33b25fd8b034e35986c0b1015d" in text
    assert '"optimizer_state_restored":false' in text
    assert '"cuda_rng_state_restored":false' in text


def test_bridge_is_single_use_and_fails_closed() -> None:
    text = launcher_text()
    assert "R2B_REGISTERED_LAUNCHER_SHA256 is required" in text
    assert "R2B_ATTEMPT_ALREADY_EXISTS" in text
    assert "CREATE_NEW_FAILED" in text
    assert "trap record_failure_artifact ERR" in text
    assert "failure_status=IMPLEMENTATION_INVALID" in text
    assert "failure_status=FAIL_STABILITY" in text
    assert "R2B_INCOMPLETE" in text
    assert text.index('create_new_json "$attempt"') < text.index(
        'srun --kill-on-bad-exit=1 python -B "$stateful_trainer"'
    )


def test_bridge_binds_the_validation_before_stateful_promotion() -> None:
    text = launcher_text()
    assert "max_printed_ppl=15.77" in text
    assert "eval at step 521999" in text
    assert "EXPECTED_ONE_BINDING_VALIDATION" in text
    assert "PPL_BAR_FAILED" in text
    assert text.index("PPL_BAR_FAILED") < text.index("--require-complete-state")
    assert "STATEFUL_RESUME_VALIDATED step=522000 optimizer=True runtime=True" in text
    assert '"stateful_reload_verified":true' in text


def test_bridge_preserves_and_rehashes_the_520k_source() -> None:
    text = launcher_text()
    assert "634a8f58e26e679323a637b4df2c726b8dd55e1964486009bf192fe8ad53f80b" in text
    assert "SOURCE_CHANGED_AFTER_BRIDGE" in text
    assert "SOURCE_CHANGED_AFTER_RELOAD" in text
    assert 'checkpoint_record "$source_checkpoint" 96' in text
    assert 'checkpoint_record "$target_checkpoint" 96' in text
    assert "rm " not in text
    assert "mv " not in text


def test_bridge_binds_the_exact_stateful_trainer() -> None:
    trainer_sha = hashlib.sha256(TRAINER.read_bytes()).hexdigest()
    assert trainer_sha == "1f10b45c9d35bbaba346e07756c38edb681ea9bb8daee1c4c08ab4d2e58aa0c4"
    assert trainer_sha in launcher_text()

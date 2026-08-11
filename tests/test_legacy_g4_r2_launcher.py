"""Static safety contract for the unsubmitted legacy G4 R2 launcher."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm" / "legacy_g4_hybrid_520k_to_524k_rung.slurm"


def launcher_text() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_r2_preserves_historical_schedule_and_hard_cap() -> None:
    text = launcher_text()
    assert "#SBATCH --time=02:00:00" in text
    assert "#SBATCH --dependency=singleton" in text
    assert "original_horizon=610000" in text
    assert '--steps "$original_horizon"' in text
    assert "--steps 524000" not in text
    assert "--save-every 4000" in text
    assert "--eval-every 2000" in text
    assert "--nonfinite-policy fail" in text
    assert "--max-nonfinite 0" in text


def test_r2_is_registered_single_use_and_stops_at_durable_target() -> None:
    text = launcher_text()
    assert "R2_REGISTERED_LAUNCHER_SHA256 is required" in text
    assert 'target_marker="$out_dir/ckpt_step524000.sharded"' in text
    assert '[[ -s "$target_marker" ]]' in text
    assert 'scancel --signal=TERM "$trainer_step"' in text
    assert 'if wait "$trainer_pid"; then' in text
    assert text.index('create_new_json "$attempt"') < text.index(
        "srun --kill-on-bad-exit=1 python -B train_hlm5_lm_fineweb.py"
    )
    assert "R2_ATTEMPT_ALREADY_EXISTS" in text
    assert "recovery_r2_failure.json" in text
    assert "trap record_failure_artifact ERR" in text
    assert "RUN_LOG_ALREADY_EXISTS" in text
    assert "failure_status=IMPLEMENTATION_INVALID" in text
    assert "failure_status=FAIL_STABILITY" in text
    assert "R2_INCOMPLETE" in text
    assert "TRAINER_OOM_BEFORE_TARGET" in text
    assert "TRAINER_IMPLEMENTATION_ERROR" in text
    assert '"optimizer_state_restored":false' in text
    assert '"data_rng_state_restored":false' in text


def test_r2_preserves_source_and_rejects_overrun() -> None:
    text = launcher_text()
    assert "634a8f58e26e679323a637b4df2c726b8dd55e1964486009bf192fe8ad53f80b" in text
    assert "SOURCE_CHANGED_AFTER_RUNG" in text
    assert "RUNG_OVERRAN_TARGET" in text
    assert 'checkpoint_record "$source_checkpoint" 96' in text
    assert 'checkpoint_record "$target_checkpoint" 96' in text
    assert "rm " not in text
    assert "mv " not in text


def test_r2_binds_current_validation_not_historical_best() -> None:
    text = launcher_text()
    assert "max_printed_ppl=15.77" in text
    assert "EXPECTED_TWO_VALIDATIONS" in text
    assert "PPL_BAR_FAILED" in text
    assert "ppl_522" in text
    assert "ppl_524" in text
    assert "printed_ppl_ceiling" in text


def test_r2_binds_exact_legacy_sources() -> None:
    text = launcher_text()
    assert "f74bf513719f8c3a110cbf45296889ad5914b3ca1cf7b5ba8538195d123d42f9" in text
    assert "e3080a257276b6d5b235a72cd9c7e18fb5a4d71e2a3fe8e30daebe5f029fd0d9" in text
    assert "3acd55ffc8789bcfafa2e72fc2aad6127e758865043f93767e28e9514952dd8e" in text

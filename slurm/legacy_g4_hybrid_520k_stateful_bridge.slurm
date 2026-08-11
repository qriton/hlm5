#!/bin/bash
#SBATCH --job-name=hlm5-g4-hybrid-r2b-bridge
#SBATCH --account=aifac_l14_039
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=24
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --dependency=singleton
#SBATCH --exclude=lrdn0289,lrdn1459,lrdn2516
#SBATCH --output=/leonardo_work/AIFAC_L14_039/hlm5-g4-hybrid-r2b-bridge-%j.out
#SBATCH --error=/leonardo_work/AIFAC_L14_039/hlm5-g4-hybrid-r2b-bridge-%j.err

# Registered warm-restart bridge from the model-only 520K checkpoint to a
# state-complete 522K checkpoint. Submission requires separate allocation
# approval plus the final launcher SHA-256 in R2B_REGISTERED_LAUNCHER_SHA256.

set -Eeuo pipefail

repo=/leonardo/home/userexternal/mdima000/HLM5
out_dir="$repo/runs/g4_hybrid"
source_checkpoint="$out_dir/ckpt_step520000"
source_marker="$out_dir/ckpt_step520000.sharded"
target_checkpoint="$out_dir/ckpt_step522000"
target_marker="$out_dir/ckpt_step522000.sharded"
attempt="$out_dir/recovery_r2b_attempt.json"
result="$out_dir/recovery_r2b_result.json"
failure="$out_dir/recovery_r2b_failure.json"
run_log="$out_dir/recovery_r2b_train_${SLURM_JOB_ID}.log"
launcher="$repo/slurm/legacy_g4_hybrid_520k_stateful_bridge.slurm"
stateful_trainer="$repo/research/legacy_g4_recovery/train_hlm5_lm_fineweb_stateful.py"
uninterrupted_log="$repo/logs/hlm5-g4-hybrid-resume-47988845.out"

anchor_ppl=15.581647041956762
max_printed_ppl=15.77
original_horizon=610000
sampling_rng_offset=108000
restart_warmup_steps=2000
restart_warmup_initial_factor=0.1
attempt_created=0
failure_status=IMPLEMENTATION_INVALID

record_failure_artifact() {
  if [[ "$attempt_created" -eq 1 && ! -e "$result" && ! -e "$failure" ]]; then
    local observed_attempt_sha
    observed_attempt_sha=$(sha256sum "$attempt" | cut -d' ' -f1)
    (set -o noclobber; printf '%s\n' \
      "{\"schema\":\"hlm5-g4-hybrid-r2b-failure-v1\",\"status\":\"${failure_status}\",\"job_id\":\"${SLURM_JOB_ID}\",\"attempt_sha256\":\"${observed_attempt_sha}\",\"reason\":\"see registered stderr\"}" \
      > "$failure") || true
  fi
}

fail() {
  record_failure_artifact
  echo "R2B_${failure_status} $*" >&2
  exit 2
}

scientific_fail() {
  failure_status=FAIL_STABILITY
  fail "$@"
}

incomplete() {
  echo "R2B_INCOMPLETE $*" >&2
  exit 3
}

trap record_failure_artifact ERR

expect_hash() {
  local expected="$1"
  local path="$2"
  local actual
  actual=$(sha256sum "$path" | cut -d' ' -f1)
  [[ "$actual" == "$expected" ]] || \
    fail "HASH_MISMATCH path=$path expected=$expected actual=$actual"
}

checkpoint_record() {
  local checkpoint="$1"
  local expected_count="$2"
  local count total aggregate
  count=$(find "$checkpoint" -maxdepth 1 -type f -name 'shard_rank*.pt' | wc -l)
  [[ "$count" -eq "$expected_count" ]] || \
    fail "SHARD_COUNT_MISMATCH checkpoint=$checkpoint count=$count"
  for rank in $(seq 0 $((expected_count - 1))); do
    [[ -s "$checkpoint/shard_rank${rank}.pt" ]] || \
      fail "MISSING_OR_EMPTY_SHARD checkpoint=$checkpoint rank=$rank"
  done
  total=$(find "$checkpoint" -maxdepth 1 -type f -name 'shard_rank*.pt' \
    -printf '%s\n' | awk '{sum += $1} END {printf "%.0f\n", sum}')
  aggregate=$(find "$checkpoint" -maxdepth 1 -type f -name 'shard_rank*.pt' \
    -print0 | sort -z -V | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
  printf '%s %s %s\n' "$count" "$total" "$aggregate"
}

create_new_json() {
  local path="$1"
  local body="$2"
  (set -o noclobber; printf '%s\n' "$body" > "$path") || \
    fail "CREATE_NEW_FAILED path=$path"
}

cd "$repo"

[[ -n "${R2B_REGISTERED_LAUNCHER_SHA256:-}" ]] || \
  fail "R2B_REGISTERED_LAUNCHER_SHA256 is required"
expect_hash "$R2B_REGISTERED_LAUNCHER_SHA256" "$launcher"
expect_hash 6f7c425ff46a7edc8cf382d3ca66b4e547dc5b71b7c39a9698fac3d0a41f5014 \
  "$source_marker"
expect_hash f74bf513719f8c3a110cbf45296889ad5914b3ca1cf7b5ba8538195d123d42f9 \
  "$repo/train_hlm5_lm_fineweb.py"
expect_hash e3080a257276b6d5b235a72cd9c7e18fb5a4d71e2a3fe8e30daebe5f029fd0d9 \
  "$repo/hlm5_lm.py"
expect_hash 3acd55ffc8789bcfafa2e72fc2aad6127e758865043f93767e28e9514952dd8e \
  "$repo/tools/validate_g4_fsdp_resume.py"
expect_hash 1f10b45c9d35bbaba346e07756c38edb681ea9bb8daee1c4c08ab4d2e58aa0c4 \
  "$stateful_trainer"
expect_hash 1fddeb1b7259f70bb9936e595df427e2e84ecb33b25fd8b034e35986c0b1015d \
  "$uninterrupted_log"

read -r source_count source_bytes source_aggregate < <(checkpoint_record "$source_checkpoint" 96)
[[ "$source_bytes" == 21670864639 ]] || \
  fail "SOURCE_BYTES_MISMATCH bytes=$source_bytes"
[[ "$source_aggregate" == 634a8f58e26e679323a637b4df2c726b8dd55e1964486009bf192fe8ad53f80b ]] || \
  fail "SOURCE_AGGREGATE_MISMATCH aggregate=$source_aggregate"

latest_marker=$(find "$out_dir" -maxdepth 1 -type f -name 'ckpt_step*.sharded' \
  -printf '%f\n' | sort -V | tail -1)
[[ "$latest_marker" == ckpt_step520000.sharded ]] || \
  fail "LATEST_MARKER_MISMATCH marker=$latest_marker"
[[ ! -e "$target_checkpoint" && ! -e "$target_marker" ]] || \
  fail "TARGET_ALREADY_EXISTS"
[[ ! -e "$attempt" && ! -e "$result" && ! -e "$failure" ]] || \
  fail "R2B_ATTEMPT_ALREADY_EXISTS"
[[ ! -e "$run_log" ]] || fail "RUN_LOG_ALREADY_EXISTS path=$run_log"
for stale in train_summary.json train_incomplete.json train_failure.json; do
  [[ ! -e "$out_dir/$stale" ]] || fail "STALE_TERMINAL_ARTIFACT path=$out_dir/$stale"
done

module purge
module load profile/deeplrn
module load cineca-ai/4.1.1
export PYTHONUNBUFFERED=1
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29528

# Read-only load preflight on the original 520K source before attempt creation.
srun --kill-on-bad-exit=1 python -B tools/validate_g4_fsdp_resume.py \
  --report-label hybrid_520000_r2b_preflight \
  --size huge3b \
  --out-dir runs/g4_hybrid \
  --grad-checkpoint \
  --memory-layer \
  --memory-size 256

launcher_sha=$(sha256sum "$launcher" | cut -d' ' -f1)
trainer_sha=$(sha256sum "$stateful_trainer" | cut -d' ' -f1)
attempt_body=$(printf '{"schema":"hlm5-g4-hybrid-r2b-attempt-v1","job_id":"%s","source_step":520000,"target_step":522000,"original_horizon":610000,"world_size":96,"launcher_sha256":"%s","stateful_trainer_sha256":"%s","source_aggregate_sha256":"%s","source_bytes":%s,"uninterrupted_log_sha256":"1fddeb1b7259f70bb9936e595df427e2e84ecb33b25fd8b034e35986c0b1015d","optimizer_state_restored":false,"cuda_rng_state_restored":false,"sampling_rng_reconstructed":true,"sampling_rng_offset":108000,"restart_warmup_steps":2000,"restart_warmup_initial_factor":0.1}\n' \
  "$SLURM_JOB_ID" "$launcher_sha" "$trainer_sha" "$source_aggregate" "$source_bytes")
create_new_json "$attempt" "$attempt_body"
attempt_created=1
attempt_sha=$(sha256sum "$attempt" | cut -d' ' -f1)
(set -o noclobber; : > "$run_log") || fail "CREATE_NEW_FAILED path=$run_log"

set +e
srun --kill-on-bad-exit=1 python -B "$stateful_trainer" \
  --size huge3b \
  --batch 1 \
  --grad-checkpoint \
  --fsdp \
  --fsdp-val-windows 20 \
  --lr 2e-4 \
  --warmup 2000 \
  --grad-clip 1.0 \
  --nonfinite-policy fail \
  --max-nonfinite 0 \
  --save-every 2000 \
  --eval-every 2000 \
  --steps "$original_horizon" \
  --out-dir runs/g4_hybrid \
  --resume auto \
  --memory-layer \
  --memory-size 256 \
  --stateful-checkpoints \
  --expected-resume-step 520000 \
  --legacy-sampling-rng-offset "$sampling_rng_offset" \
  --restart-warmup-steps "$restart_warmup_steps" \
  --restart-warmup-initial-factor "$restart_warmup_initial_factor" \
  --no-prune-checkpoints >> "$run_log" 2>&1 &
trainer_pid=$!
set -e

target_seen=0
while kill -0 "$trainer_pid" 2>/dev/null; do
  if [[ -s "$target_marker" ]]; then
    target_seen=1
    trainer_step=$(squeue --steps -j "$SLURM_JOB_ID" -h -o '%i' \
      | awk '/\.[0-9]+$/ {print; exit}')
    [[ -n "$trainer_step" ]] || fail "TRAINER_STEP_NOT_FOUND"
    echo "TARGET_MARKER_SEEN stopping_step=$trainer_step"
    scancel --signal=TERM "$trainer_step"
    break
  fi
  sleep 2
done

if wait "$trainer_pid"; then
  trainer_rc=0
else
  trainer_rc=$?
fi
[[ "$target_seen" -eq 1 ]] || {
  if [[ -e "$out_dir/train_failure.json" ]] || \
    grep -Eqi 'non-finite (loss|gradient)|nonfinite' "$run_log"; then
    scientific_fail "TRAINER_FAILED_BEFORE_TARGET rc=$trainer_rc log=$run_log"
  fi
  if grep -Eqi 'CUDA out of memory|OutOfMemoryError' "$run_log"; then
    scientific_fail "TRAINER_OOM_BEFORE_TARGET rc=$trainer_rc log=$run_log"
  fi
  if grep -Eq 'Traceback \(most recent call last\)|RuntimeError:|ValueError:' "$run_log"; then
    fail "TRAINER_IMPLEMENTATION_ERROR rc=$trainer_rc log=$run_log"
  fi
  incomplete "TRAINER_EXITED_BEFORE_TARGET rc=$trainer_rc log=$run_log"
}

read -r post_source_count post_source_bytes post_source_aggregate \
  < <(checkpoint_record "$source_checkpoint" 96)
[[ "$post_source_bytes" == "$source_bytes" && "$post_source_aggregate" == "$source_aggregate" ]] || \
  fail "SOURCE_CHANGED_AFTER_BRIDGE"
read -r target_count target_bytes target_aggregate < <(checkpoint_record "$target_checkpoint" 96)

python -B - "$target_marker" <<'PY'
import json
import sys
from pathlib import Path

marker = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "step": 521999,
    "world_size": 96,
    "checkpoint_schema": "hlm5-g4-state-complete-v1",
    "optimizer_state_saved": True,
    "runtime_state_saved": True,
}
for key, value in expected.items():
    if marker.get(key) != value:
        raise SystemExit(f"TARGET_MARKER_FIELD_MISMATCH {key}={marker.get(key)!r}")
PY

[[ ! -e "$out_dir/ckpt_step524000" && ! -e "$out_dir/ckpt_step524000.sharded" ]] || \
  fail "BRIDGE_OVERRAN_TARGET"
[[ ! -e "$out_dir/train_failure.json" ]] || scientific_fail "TRAIN_FAILURE_ARTIFACT_PRESENT"
if grep -Eqi 'non-finite (loss|gradient)|nonfinite' "$run_log"; then
  scientific_fail "NONFINITE_REPORTED log=$run_log"
fi
grep -Fq 'resumed sharded runs/g4_hybrid/ckpt_step520000' "$run_log" || \
  fail "EXPECTED_RESUME_LOG_MISSING"
grep -Fq 'advanced sampling RNG by 108000 legacy draws' "$run_log" || \
  fail "SAMPLING_RNG_REPLAY_LOG_MISSING"
grep -Fq 'restart LR bridge: steps=2000 initial_factor=0.100000 resume_step=520000' \
  "$run_log" || fail "RESTART_BRIDGE_LOG_MISSING"

mapfile -t observed_ppl < <(
  awk '/eval at step 521999/{want=1; next} want && /val PPL/{print $3; want=0}' "$run_log")
[[ "${#observed_ppl[@]}" -eq 1 ]] || \
  fail "EXPECTED_ONE_BINDING_VALIDATION count=${#observed_ppl[@]}"
ppl_522="${observed_ppl[0]}"
if ! python -B - "$ppl_522" "$max_printed_ppl" <<'PY'
import math
import sys

observed = float(sys.argv[1])
ceiling = float(sys.argv[2])
raise SystemExit(0 if math.isfinite(observed) and observed <= ceiling else 1)
PY
then
  scientific_fail "PPL_BAR_FAILED step=522000 observed=$ppl_522 ceiling=$max_printed_ppl"
fi

# A PASS requires a fresh all-rank load of optimizer and every rank-local RNG
# stream from the newly written 522K checkpoint before any further training.
srun --kill-on-bad-exit=1 python -B "$stateful_trainer" \
  --size huge3b \
  --batch 1 \
  --grad-checkpoint \
  --fsdp \
  --steps "$original_horizon" \
  --out-dir runs/g4_hybrid \
  --resume auto \
  --memory-layer \
  --memory-size 256 \
  --stateful-checkpoints \
  --require-complete-state \
  --expected-resume-step 522000 \
  --stop-after-resume \
  --no-prune-checkpoints >> "$run_log" 2>&1
grep -Fq 'STATEFUL_RESUME_VALIDATED step=522000 optimizer=True runtime=True' "$run_log" || \
  fail "STATEFUL_RELOAD_PROOF_MISSING"

read -r final_source_count final_source_bytes final_source_aggregate \
  < <(checkpoint_record "$source_checkpoint" 96)
[[ "$final_source_bytes" == "$source_bytes" && "$final_source_aggregate" == "$source_aggregate" ]] || \
  fail "SOURCE_CHANGED_AFTER_RELOAD"
read -r final_target_count final_target_bytes final_target_aggregate \
  < <(checkpoint_record "$target_checkpoint" 96)
[[ "$final_target_bytes" == "$target_bytes" && "$final_target_aggregate" == "$target_aggregate" ]] || \
  fail "TARGET_CHANGED_AFTER_RELOAD"

result_body=$(printf '{"schema":"hlm5-g4-hybrid-r2b-result-v1","status":"PASS_BRIDGE","job_id":"%s","attempt_sha256":"%s","source_step":520000,"target_step":522000,"printed_ppl":%s,"printed_ppl_ceiling":15.77,"anchor_ppl":15.581647041956762,"sampling_rng_offset":108000,"restart_warmup_steps":2000,"restart_warmup_initial_factor":0.1,"target_shards":%s,"target_bytes":%s,"target_aggregate_sha256":"%s","stateful_reload_verified":true}\n' \
  "$SLURM_JOB_ID" "$attempt_sha" "$ppl_522" "$target_count" "$target_bytes" "$target_aggregate")
create_new_json "$result" "$result_body"
echo "R2B_PASS_BRIDGE target=ckpt_step522000 ppl=$ppl_522 aggregate=$target_aggregate"

#!/bin/bash
#SBATCH --job-name=hlm5-g4-hybrid-r2-524k
#SBATCH --account=aifac_l14_039
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=24
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --dependency=singleton
#SBATCH --exclude=lrdn0289,lrdn1459,lrdn2516
#SBATCH --output=/leonardo_work/AIFAC_L14_039/hlm5-g4-hybrid-r2-524k-%j.out
#SBATCH --error=/leonardo_work/AIFAC_L14_039/hlm5-g4-hybrid-r2-524k-%j.err

# Registered R2 stability rung for the legacy 96-way G4 hybrid checkpoint.
# This preserves the original 610K cosine horizon and stops the exact legacy
# trainer immediately after its durable 524K save. It must never be submitted
# without a separately recorded allocation approval and launcher SHA-256.

set -Eeuo pipefail

repo=/leonardo/home/userexternal/mdima000/HLM5
out_dir="$repo/runs/g4_hybrid"
source_checkpoint="$out_dir/ckpt_step520000"
source_marker="$out_dir/ckpt_step520000.sharded"
target_checkpoint="$out_dir/ckpt_step524000"
target_marker="$out_dir/ckpt_step524000.sharded"
attempt="$out_dir/recovery_r2_attempt.json"
result="$out_dir/recovery_r2_result.json"
failure="$out_dir/recovery_r2_failure.json"
run_log="$out_dir/recovery_r2_train_${SLURM_JOB_ID}.log"
launcher="$repo/slurm/legacy_g4_hybrid_520k_to_524k_rung.slurm"

anchor_ppl=15.581647041956762
# The legacy source prints current validation PPL to two decimals and stores
# only historical best PPL in the marker. Requiring printed PPL <= 15.77 is a
# conservative implementation of the registered +0.20 absolute bar.
max_printed_ppl=15.77
original_horizon=610000
attempt_created=0
failure_status=IMPLEMENTATION_INVALID

record_failure_artifact() {
  if [[ "$attempt_created" -eq 1 && ! -e "$result" && ! -e "$failure" ]]; then
    local observed_attempt_sha
    observed_attempt_sha=$(sha256sum "$attempt" | cut -d' ' -f1)
    (set -o noclobber; printf '%s\n' \
      "{\"schema\":\"hlm5-g4-hybrid-r2-failure-v1\",\"status\":\"${failure_status}\",\"job_id\":\"${SLURM_JOB_ID}\",\"attempt_sha256\":\"${observed_attempt_sha}\",\"reason\":\"see registered stderr\"}" \
      > "$failure") || true
  fi
}

fail() {
  local message="$*"
  record_failure_artifact
  echo "R2_${failure_status} $message" >&2
  exit 2
}

scientific_fail() {
  failure_status=FAIL_STABILITY
  fail "$@"
}

incomplete() {
  echo "R2_INCOMPLETE $*" >&2
  exit 3
}

# ERR catches unhandled post-attempt command failures. Scheduler termination is
# not converted to INVALID; without a known failure artifact it stays incomplete.
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

[[ -n "${R2_REGISTERED_LAUNCHER_SHA256:-}" ]] || \
  fail "R2_REGISTERED_LAUNCHER_SHA256 is required"
expect_hash "$R2_REGISTERED_LAUNCHER_SHA256" "$launcher"
expect_hash 6f7c425ff46a7edc8cf382d3ca66b4e547dc5b71b7c39a9698fac3d0a41f5014 \
  "$source_marker"
expect_hash f74bf513719f8c3a110cbf45296889ad5914b3ca1cf7b5ba8538195d123d42f9 \
  "$repo/train_hlm5_lm_fineweb.py"
expect_hash e3080a257276b6d5b235a72cd9c7e18fb5a4d71e2a3fe8e30daebe5f029fd0d9 \
  "$repo/hlm5_lm.py"
expect_hash 3acd55ffc8789bcfafa2e72fc2aad6127e758865043f93767e28e9514952dd8e \
  "$repo/tools/validate_g4_fsdp_resume.py"

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
  fail "R2_ATTEMPT_ALREADY_EXISTS"
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
export MASTER_PORT=29527

# Re-run the already-passed load-only check in the actual allocation before the
# first mutating operation.
srun --kill-on-bad-exit=1 python -B tools/validate_g4_fsdp_resume.py \
  --report-label hybrid_520000_r2_preflight \
  --size huge3b \
  --out-dir runs/g4_hybrid \
  --grad-checkpoint \
  --memory-layer \
  --memory-size 256

launcher_sha=$(sha256sum "$launcher" | cut -d' ' -f1)
attempt_body=$(printf '{"schema":"hlm5-g4-hybrid-r2-attempt-v1","job_id":"%s","source_step":520000,"target_step":524000,"original_horizon":610000,"world_size":96,"launcher_sha256":"%s","source_aggregate_sha256":"%s","source_bytes":%s,"optimizer_state_restored":false,"data_rng_state_restored":false}\n' \
  "$SLURM_JOB_ID" "$launcher_sha" "$source_aggregate" "$source_bytes")
create_new_json "$attempt" "$attempt_body"
attempt_created=1
attempt_sha=$(sha256sum "$attempt" | cut -d' ' -f1)
(set -o noclobber; : > "$run_log") || fail "CREATE_NEW_FAILED path=$run_log"

# Keep --steps at the historical 610K horizon so lr_at(step) is unchanged.
# The monitor terminates only the training job step after the complete 524K
# marker appears; the batch allocation remains alive for validation.
set +e
srun --kill-on-bad-exit=1 python -B train_hlm5_lm_fineweb.py \
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
  --save-every 4000 \
  --eval-every 2000 \
  --steps "$original_horizon" \
  --out-dir runs/g4_hybrid \
  --resume auto \
  --memory-layer \
  --memory-size 256 >> "$run_log" 2>&1 &
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
  # Expected after scancel of the completed target job step. Keeping wait in a
  # conditional prevents the ERR trap from misclassifying this as INVALID.
  trainer_rc=$?
fi
[[ "$target_seen" -eq 1 ]] || \
  {
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

# The target marker is written only after every rank reports a successful
# shard save. Rebind both source and target after the expected termination.
read -r post_source_count post_source_bytes post_source_aggregate \
  < <(checkpoint_record "$source_checkpoint" 96)
[[ "$post_source_bytes" == "$source_bytes" && "$post_source_aggregate" == "$source_aggregate" ]] || \
  fail "SOURCE_CHANGED_AFTER_RUNG"
read -r target_count target_bytes target_aggregate < <(checkpoint_record "$target_checkpoint" 96)

latest_marker=$(find "$out_dir" -maxdepth 1 -type f -name 'ckpt_step*.sharded' \
  -printf '%f\n' | sort -V | tail -1)
[[ "$latest_marker" == ckpt_step524000.sharded ]] || \
  fail "POST_RUNG_LATEST_MARKER_MISMATCH marker=$latest_marker"
[[ ! -e "$out_dir/ckpt_step528000" && ! -e "$out_dir/ckpt_step528000.sharded" ]] || \
  fail "RUNG_OVERRAN_TARGET"
[[ ! -e "$out_dir/train_failure.json" ]] || scientific_fail "TRAIN_FAILURE_ARTIFACT_PRESENT"
if grep -Eqi 'non-finite (loss|gradient)|nonfinite' "$run_log"; then
  scientific_fail "NONFINITE_REPORTED log=$run_log"
fi
grep -Fq 'resumed sharded runs/g4_hybrid/ckpt_step520000' "$run_log" || \
  fail "EXACT_RESUME_LINE_MISSING"

mapfile -t ppls < <(grep -E 'val PPL [0-9]' "$run_log" | awk '{print $NF}')
[[ "${#ppls[@]}" -eq 2 ]] || fail "EXPECTED_TWO_VALIDATIONS count=${#ppls[@]}"
ppl_522=${ppls[0]}
ppl_524=${ppls[1]}
awk -v a="$ppl_522" -v b="$ppl_524" -v ceiling="$max_printed_ppl" \
  'BEGIN { if (a+0 != a || b+0 != b || a > ceiling || b > ceiling) exit 1 }' || \
  scientific_fail "PPL_BAR_FAILED ppl_522=$ppl_522 ppl_524=$ppl_524 printed_ceiling=$max_printed_ppl"

# The same all-rank loader now targets 524K because it is the latest marker.
export MASTER_PORT=29528
if ! srun --kill-on-bad-exit=1 python -B tools/validate_g4_fsdp_resume.py \
  --report-label hybrid_524000_r2_result \
  --size huge3b \
  --out-dir runs/g4_hybrid \
  --grad-checkpoint \
  --memory-layer \
  --memory-size 256; then
  scientific_fail "TARGET_LOAD_VALIDATION_FAILED"
fi

expect_hash 6f7c425ff46a7edc8cf382d3ca66b4e547dc5b71b7c39a9698fac3d0a41f5014 \
  "$source_marker"
target_marker_sha=$(sha256sum "$target_marker" | cut -d' ' -f1)
run_log_sha=$(sha256sum "$run_log" | cut -d' ' -f1)
result_body=$(printf '{"schema":"hlm5-g4-hybrid-r2-result-v1","status":"PASS","job_id":"%s","attempt_sha256":"%s","source_step":520000,"target_step":524000,"original_horizon":610000,"world_size":96,"optimizer_state_restored":false,"data_rng_state_restored":false,"trainer_exit_after_expected_term":%s,"anchor_ppl":%.15f,"printed_ppl_ceiling":%.2f,"ppl_522":%s,"ppl_524":%s,"source_aggregate_sha256":"%s","target_aggregate_sha256":"%s","target_bytes":%s,"target_marker_sha256":"%s","run_log_sha256":"%s"}\n' \
  "$SLURM_JOB_ID" "$attempt_sha" "$trainer_rc" "$anchor_ppl" "$max_printed_ppl" \
  "$ppl_522" "$ppl_524" "$source_aggregate" "$target_aggregate" \
  "$target_bytes" "$target_marker_sha" "$run_log_sha")
create_new_json "$result" "$result_body"
echo "R2_STABILITY_PASS source=520000 target=524000 ppl_522=$ppl_522 ppl_524=$ppl_524"

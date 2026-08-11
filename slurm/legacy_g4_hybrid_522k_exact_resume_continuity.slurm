#!/bin/bash
#SBATCH --job-name=hlm5-g4-hybrid-r2c-continuity
#SBATCH --account=aifac_l14_039
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=24
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --dependency=singleton
#SBATCH --exclude=lrdn0289,lrdn1459,lrdn2516
#SBATCH --output=/leonardo_work/AIFAC_L14_039/hlm5-g4-hybrid-r2c-%j.out
#SBATCH --error=/leonardo_work/AIFAC_L14_039/hlm5-g4-hybrid-r2c-%j.err

# Registered R2c paired continuity assay.  Arm A runs 522K -> 523K in one
# process.  Arm B runs the same updates with a save/process-restart at 522.5K.
# Both arms evaluate and save at the same two boundaries.  The midpoint must
# first be exact, otherwise the restart effect is not identified.

set -Eeuo pipefail

repo=/leonardo/home/userexternal/mdima000/HLM5
source_dir="$repo/runs/g4_hybrid"
source_checkpoint="$source_dir/ckpt_step522000"
source_marker="$source_dir/ckpt_step522000.sharded"
work_parent=/leonardo_work/AIFAC_L14_039
work_root="$work_parent/hlm5-g4-hybrid-r2c"
continuous_dir="$work_root/continuous"
split_dir="$work_root/split"
attempt="$work_root/attempt.json"
result="$work_root/result.json"
failure="$work_root/failure.json"
environment_record="$work_root/environment.json"
continuous_log="$work_root/continuous.log"
split_pre_log="$work_root/split-pre-restart.log"
split_post_log="$work_root/split-post-restart.log"
mid_report="$work_root/midpoint-comparison"
final_report="$work_root/final-comparison"
launcher="$repo/slurm/legacy_g4_hybrid_522k_exact_resume_continuity.slurm"
protocol="$repo/docs/legacy-g4-hybrid-r2c-exact-resume-contract-2026-08-11.md"
trainer="$repo/research/legacy_g4_recovery/train_hlm5_lm_fineweb_stateful.py"
comparator="$repo/research/legacy_g4_recovery/compare_stateful_shards.py"

source_step=522000
midpoint_step=522500
target_step=523000
schedule_horizon=610000
cadence=500
world_size=96
minimum_free_bytes=260000000000
source_bytes_expected=43294668998
source_aggregate_expected=e32a2b0945e2197cc080d585bd16c7a33a817d63a6e5d96fb436127f8fdecae8
source_marker_sha_expected=8e83af00aed53b74e39aaee0ca090aef81a8232db9d0fd46aad1a16f82fcfe42
trainer_sha_expected=1f10b45c9d35bbaba346e07756c38edb681ea9bb8daee1c4c08ab4d2e58aa0c4
comparator_sha_expected=52a676ce2288a087e5ee4aa8c078fd44d40ff75063188b248b92b8a5f9791482
model_sha_expected=e3080a257276b6d5b235a72cd9c7e18fb5a4d71e2a3fe8e30daebe5f029fd0d9

attempt_created=0
result_created=0
failure_status=IMPLEMENTATION_INVALID
failure_reason=unclassified

record_failure_artifact() {
  if [[ "$attempt_created" -eq 1 && "$result_created" -eq 0 && \
        ! -e "$result" && ! -e "$failure" ]]; then
    local attempt_sha
    attempt_sha=$(sha256sum "$attempt" | cut -d' ' -f1)
    (set -o noclobber; printf '%s\n' \
      "{\"schema\":\"hlm5-g4-hybrid-r2c-failure-v1\",\"status\":\"${failure_status}\",\"job_id\":\"${SLURM_JOB_ID}\",\"attempt_sha256\":\"${attempt_sha}\",\"reason\":\"${failure_reason}\"}" \
      > "$failure") || true
  fi
}

fail() {
  failure_status=IMPLEMENTATION_INVALID
  failure_reason="$1"
  record_failure_artifact
  echo "R2C_IMPLEMENTATION_INVALID $*" >&2
  exit 2
}

stability_fail() {
  failure_status=FAIL_STABILITY
  failure_reason="$1"
  record_failure_artifact
  echo "R2C_FAIL_STABILITY $*" >&2
  exit 1
}

incomplete() {
  failure_status=INCOMPLETE
  failure_reason="$1"
  record_failure_artifact
  echo "R2C_INCOMPLETE $*" >&2
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

validate_marker() {
  local marker="$1"
  local expected_payload_step="$2"
  python -B - "$marker" "$expected_payload_step" <<'PY'
import json
import sys
from pathlib import Path

marker = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "step": int(sys.argv[2]),
    "world_size": 96,
    "checkpoint_schema": "hlm5-g4-state-complete-v1",
    "optimizer_state_saved": True,
    "runtime_state_saved": True,
}
for key, value in expected.items():
    if marker.get(key) != value:
        raise SystemExit(
            f"TARGET_MARKER_FIELD_MISMATCH {key}={marker.get(key)!r} expected={value!r}")
PY
}

check_segment_log() {
  local log_path="$1"
  local expected_resume="$2"
  local expected_target="$3"
  grep -Fq "at step $expected_resume" "$log_path" || \
    fail "EXPECTED_RESUME_LOG_MISSING log=$log_path step=$expected_resume"
  grep -Fq "optimizer=True runtime=True" "$log_path" && \
    fail "UNEXPECTED_STOP_AFTER_RESUME_MARKER log=$log_path"
  ! grep -Fq 'restart LR bridge:' "$log_path" || \
    fail "UNEXPECTED_RESTART_WARMUP log=$log_path"
  if grep -Eqi 'non-finite (loss|gradient)|NON-FINITE|nan' "$log_path"; then
    stability_fail "NONFINITE_REPORTED log=$log_path"
  fi
  if grep -Eqi 'CUDA out of memory|OutOfMemoryError' "$log_path"; then
    stability_fail "OOM_REPORTED log=$log_path"
  fi
  grep -Fq "saved sharded ckpt_step${expected_target} (96 shards)" "$log_path" || \
    fail "TARGET_SAVE_LOG_MISSING log=$log_path step=$expected_target"
}

run_segment() {
  local label="$1"
  local out_dir="$2"
  local expected_resume="$3"
  local expected_target="$4"
  local log_path="$5"
  local target_checkpoint="$out_dir/ckpt_step${expected_target}"
  local target_marker="$out_dir/ckpt_step${expected_target}.sharded"
  local trainer_pid trainer_step trainer_rc target_seen

  [[ ! -e "$target_checkpoint" && ! -e "$target_marker" ]] || \
    fail "SEGMENT_TARGET_ALREADY_EXISTS label=$label target=$expected_target"
  (set -o noclobber; : > "$log_path") || fail "RUN_LOG_ALREADY_EXISTS path=$log_path"

  set +e
  srun --kill-on-bad-exit=1 python -B "$trainer" \
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
    --save-every "$cadence" \
    --eval-every "$cadence" \
    --steps "$schedule_horizon" \
    --out-dir "$out_dir" \
    --resume auto \
    --memory-layer \
    --memory-size 256 \
    --stateful-checkpoints \
    --require-complete-state \
    --expected-resume-step "$expected_resume" \
    --no-prune-checkpoints >> "$log_path" 2>&1 &
  trainer_pid=$!
  set -e

  target_seen=0
  while kill -0 "$trainer_pid" 2>/dev/null; do
    if [[ -s "$target_marker" ]]; then
      target_seen=1
      trainer_step=$(squeue --steps -j "$SLURM_JOB_ID" -h -o '%i' \
        | awk '/\.[0-9]+$/ {print; exit}')
      [[ -n "$trainer_step" ]] || fail "TRAINER_STEP_NOT_FOUND label=$label"
      echo "R2C_TARGET_MARKER_SEEN label=$label stopping_step=$trainer_step"
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
  if [[ "$target_seen" -ne 1 ]]; then
    if [[ -e "$out_dir/train_failure.json" ]] || \
       grep -Eqi 'non-finite (loss|gradient)|NON-FINITE|nan' "$log_path"; then
      stability_fail "SEGMENT_NONFINITE label=$label rc=$trainer_rc"
    fi
    if grep -Eqi 'CUDA out of memory|OutOfMemoryError' "$log_path"; then
      stability_fail "SEGMENT_OOM label=$label rc=$trainer_rc"
    fi
    if grep -Eq 'Traceback \(most recent call last\)|RuntimeError:|ValueError:' "$log_path"; then
      fail "SEGMENT_IMPLEMENTATION_ERROR label=$label rc=$trainer_rc"
    fi
    incomplete "SEGMENT_EXITED_BEFORE_TARGET label=$label rc=$trainer_rc"
  fi

  checkpoint_record "$target_checkpoint" "$world_size" >/dev/null
  validate_marker "$target_marker" $((expected_target - 1))
  check_segment_log "$log_path" "$expected_resume" "$expected_target"
}

extract_ppl() {
  local log_path="$1"
  local payload_step="$2"
  awk -v target="$payload_step" \
    '$0 ~ "eval at step " target {want=1; next} want && /val PPL/{print $3; want=0}' \
    "$log_path"
}

run_comparison() {
  local label="$1"
  local left="$2"
  local right="$3"
  local report_dir="$4"
  local payload_step="$5"
  local compare_log="$work_root/${label}-comparison.log"
  local rc status

  [[ ! -e "$report_dir" && ! -e "$compare_log" ]] || \
    fail "COMPARISON_OUTPUT_ALREADY_EXISTS label=$label"
  set +e
  srun --kill-on-bad-exit=1 python -B "$comparator" \
    --left "$left" \
    --right "$right" \
    --report-dir "$report_dir" \
    --label "$label" \
    --expected-world "$world_size" \
    --expected-step "$payload_step" > "$compare_log" 2>&1
  rc=$?
  set -e
  [[ -s "$report_dir/aggregate.json" ]] || \
    fail "COMPARISON_AGGREGATE_MISSING label=$label rc=$rc"
  status=$(python -B - "$report_dir/aggregate.json" <<'PY'
import json
import sys
from pathlib import Path

record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if record.get("rank_report_count") != 96 or record.get("world_size") != 96:
    raise SystemExit("comparison did not cover all 96 ranks")
print(record.get("status", "MISSING"))
PY
  ) || fail "COMPARISON_AGGREGATE_INVALID label=$label"
  case "$status" in
    PASS_EXACT)
      [[ "$rc" -eq 0 ]] || fail "COMPARISON_STATUS_RC_MISMATCH label=$label status=$status rc=$rc"
      ;;
    FAIL_MISMATCH)
      [[ "$rc" -eq 1 ]] || fail "COMPARISON_STATUS_RC_MISMATCH label=$label status=$status rc=$rc"
      ;;
    IMPLEMENTATION_INVALID)
      fail "COMPARISON_IMPLEMENTATION_INVALID label=$label"
      ;;
    *)
      fail "COMPARISON_UNKNOWN_STATUS label=$label status=$status"
      ;;
  esac
  return "$rc"
}

verify_source_unchanged() {
  local final_source_count final_source_bytes final_source_aggregate
  read -r final_source_count final_source_bytes final_source_aggregate \
    < <(checkpoint_record "$source_checkpoint" "$world_size")
  [[ "$final_source_bytes" == "$source_bytes" && \
     "$final_source_aggregate" == "$source_aggregate" ]] || \
    fail "SOURCE_CHANGED_DURING_R2C"
}

write_terminal_result() {
  local status="$1"
  local midpoint_overall="$2"
  local final_overall="$3"
  python -B - \
    "$result" "$status" "$SLURM_JOB_ID" "$attempt" "$environment_record" \
    "$mid_report/aggregate.json" "$final_report/aggregate.json" \
    "$continuous_log" "$split_pre_log" "$split_post_log" \
    "$midpoint_overall" "$final_overall" \
    "${midpoint_exact:-false}" "${midpoint_ppl_exact:-false}" \
    "${midpoint_marker_exact:-false}" "${final_exact:-false}" \
    "${final_ppl_exact:-false}" "${final_marker_exact:-false}" \
    "${continuous_mid_ppl:-}" "${split_mid_ppl:-}" \
    "${continuous_final_ppl:-}" "${split_final_ppl:-}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def digest(path_text):
    path = Path(path_text)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def boolean(text):
    if text not in {"true", "false"}:
        raise SystemExit(f"invalid boolean {text!r}")
    return text == "true"


def number(text):
    return float(text) if text else None


result_path = Path(sys.argv[1])
payload = {
    "schema": "hlm5-g4-hybrid-r2c-result-v1",
    "status": sys.argv[2],
    "job_id": sys.argv[3],
    "attempt_sha256": digest(sys.argv[4]),
    "environment_sha256": digest(sys.argv[5]),
    "source_step": 522000,
    "midpoint_step": 522500,
    "target_step": 523000,
    "midpoint_exact": boolean(sys.argv[11]),
    "final_exact": boolean(sys.argv[12]),
    "components": {
        "midpoint_semantic_exact": boolean(sys.argv[13]),
        "midpoint_ppl_exact": boolean(sys.argv[14]),
        "midpoint_marker_exact": boolean(sys.argv[15]),
        "final_semantic_exact": boolean(sys.argv[16]),
        "final_ppl_exact": boolean(sys.argv[17]),
        "final_marker_exact": boolean(sys.argv[18]),
    },
    "validation_ppl": {
        "continuous_midpoint": number(sys.argv[19]),
        "split_midpoint": number(sys.argv[20]),
        "continuous_final": number(sys.argv[21]),
        "split_final": number(sys.argv[22]),
    },
    "evidence": {
        "midpoint_report_sha256": digest(sys.argv[6]),
        "final_report_sha256": digest(sys.argv[7]),
        "continuous_log_sha256": digest(sys.argv[8]),
        "split_pre_restart_log_sha256": digest(sys.argv[9]),
        "split_post_restart_log_sha256": digest(sys.argv[10]),
    },
}
with result_path.open("x", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
PY
  result_created=1
}

cd "$repo"

[[ -n "${R2C_REGISTERED_LAUNCHER_SHA256:-}" ]] || \
  fail "R2C_REGISTERED_LAUNCHER_SHA256_REQUIRED"
[[ -n "${R2C_REGISTERED_PROTOCOL_SHA256:-}" ]] || \
  fail "R2C_REGISTERED_PROTOCOL_SHA256_REQUIRED"
expect_hash "$R2C_REGISTERED_LAUNCHER_SHA256" "$launcher"
expect_hash "$R2C_REGISTERED_PROTOCOL_SHA256" "$protocol"
expect_hash "$trainer_sha_expected" "$trainer"
expect_hash "$comparator_sha_expected" "$comparator"
expect_hash "$model_sha_expected" "$repo/hlm5_lm.py"
expect_hash "$source_marker_sha_expected" "$source_marker"

read -r source_count source_bytes source_aggregate \
  < <(checkpoint_record "$source_checkpoint" "$world_size")
[[ "$source_bytes" == "$source_bytes_expected" ]] || \
  fail "SOURCE_BYTES_MISMATCH bytes=$source_bytes"
[[ "$source_aggregate" == "$source_aggregate_expected" ]] || \
  fail "SOURCE_AGGREGATE_MISMATCH aggregate=$source_aggregate"
latest_source_marker=$(find "$source_dir" -maxdepth 1 -type f -name 'ckpt_step*.sharded' \
  -printf '%f\n' | sort -V | tail -1)
[[ "$latest_source_marker" == ckpt_step522000.sharded ]] || \
  fail "SOURCE_LATEST_MARKER_MISMATCH marker=$latest_source_marker"
[[ ! -e "$work_root" ]] || fail "R2C_WORK_ROOT_ALREADY_EXISTS path=$work_root"
available_bytes=$(df -B1 --output=avail "$work_parent" | tail -1 | tr -d ' ')
[[ "$available_bytes" -ge "$minimum_free_bytes" ]] || \
  fail "INSUFFICIENT_WORK_STORAGE available=$available_bytes required=$minimum_free_bytes"

module purge
module load profile/deeplrn
module load cineca-ai/4.1.1
export PYTHONUNBUFFERED=1
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29533

# Read-only proof that the state-complete source can be loaded before burning
# the attempt or evaluating either changed trajectory.
srun --kill-on-bad-exit=1 python -B "$trainer" \
  --size huge3b \
  --batch 1 \
  --grad-checkpoint \
  --fsdp \
  --steps "$schedule_horizon" \
  --out-dir "$source_dir" \
  --resume auto \
  --memory-layer \
  --memory-size 256 \
  --stateful-checkpoints \
  --require-complete-state \
  --expected-resume-step "$source_step" \
  --stop-after-resume \
  --no-prune-checkpoints

mkdir "$work_root" "$continuous_dir" "$split_dir"
ln -s "$source_checkpoint" "$continuous_dir/ckpt_step522000"
ln -s "$source_marker" "$continuous_dir/ckpt_step522000.sharded"
ln -s "$source_checkpoint" "$split_dir/ckpt_step522000"
ln -s "$source_marker" "$split_dir/ckpt_step522000.sharded"

launcher_sha=$(sha256sum "$launcher" | cut -d' ' -f1)
protocol_sha=$(sha256sum "$protocol" | cut -d' ' -f1)
trainer_sha=$(sha256sum "$trainer" | cut -d' ' -f1)
comparator_sha=$(sha256sum "$comparator" | cut -d' ' -f1)
node_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd, -)
python -B - "$environment_record" "$node_list" <<'PY'
import json
import os
import platform
import sys
from pathlib import Path

import torch

cuda_backend = getattr(torch.backends, "cuda", None)
matmul = getattr(cuda_backend, "matmul", None)
record = {
    "schema": "hlm5-g4-hybrid-r2c-environment-v1",
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "allow_tf32_matmul": getattr(matmul, "allow_tf32", None),
    "allow_bf16_reduced_precision_reduction": getattr(
        matmul, "allow_bf16_reduced_precision_reduction", None),
    "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    "node_list": sys.argv[2].split(","),
    "world_size": 96,
}
with Path(sys.argv[1]).open("x", encoding="utf-8", newline="\n") as handle:
    json.dump(record, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
PY
environment_sha=$(sha256sum "$environment_record" | cut -d' ' -f1)
attempt_body=$(printf '{"schema":"hlm5-g4-hybrid-r2c-attempt-v1","job_id":"%s","source_step":522000,"midpoint_step":522500,"target_step":523000,"schedule_horizon":610000,"cadence":500,"world_size":96,"node_list":"%s","environment_sha256":"%s","launcher_sha256":"%s","protocol_sha256":"%s","stateful_trainer_sha256":"%s","comparator_sha256":"%s","source_marker_sha256":"%s","source_aggregate_sha256":"%s","source_bytes":%s,"midpoint_policy":"mismatch_is_invalid_unidentified","final_policy":"mismatch_after_exact_midpoint_is_fail_continuity"}' \
  "$SLURM_JOB_ID" "$node_list" "$environment_sha" "$launcher_sha" "$protocol_sha" "$trainer_sha" \
  "$comparator_sha" "$source_marker_sha_expected" "$source_aggregate" "$source_bytes")
create_new_json "$attempt" "$attempt_body"
attempt_created=1

run_segment continuous "$continuous_dir" "$source_step" "$target_step" "$continuous_log"
run_segment split_pre_restart "$split_dir" "$source_step" "$midpoint_step" "$split_pre_log"

continuous_mid_ppl=$(extract_ppl "$continuous_log" $((midpoint_step - 1)))
split_mid_ppl=$(extract_ppl "$split_pre_log" $((midpoint_step - 1)))
midpoint_ppl_exact=false
midpoint_marker_exact=false
if [[ -n "$continuous_mid_ppl" && "$continuous_mid_ppl" == "$split_mid_ppl" ]]; then
  midpoint_ppl_exact=true
fi
if cmp -s "$continuous_dir/ckpt_step${midpoint_step}.sharded" \
  "$split_dir/ckpt_step${midpoint_step}.sharded"; then
  midpoint_marker_exact=true
fi

if run_comparison midpoint \
  "$continuous_dir/ckpt_step${midpoint_step}" \
  "$split_dir/ckpt_step${midpoint_step}" \
  "$mid_report" $((midpoint_step - 1)); then
  midpoint_exact=true
else
  midpoint_exact=false
fi
if [[ "$midpoint_exact" != true || "$midpoint_ppl_exact" != true || \
      "$midpoint_marker_exact" != true ]]; then
  verify_source_unchanged
  write_terminal_result INVALID_REPRODUCIBILITY false false
  echo "R2C_INVALID_REPRODUCIBILITY midpoint trajectories differ before restart semantic=$midpoint_exact ppl=$midpoint_ppl_exact marker=$midpoint_marker_exact" >&2
  exit 2
fi

run_segment split_post_restart "$split_dir" "$midpoint_step" "$target_step" "$split_post_log"

continuous_final_ppl=$(extract_ppl "$continuous_log" $((target_step - 1)))
split_final_ppl=$(extract_ppl "$split_post_log" $((target_step - 1)))
final_ppl_exact=false
final_marker_exact=false
if [[ -n "$continuous_final_ppl" && "$continuous_final_ppl" == "$split_final_ppl" ]]; then
  final_ppl_exact=true
fi
if cmp -s "$continuous_dir/ckpt_step${target_step}.sharded" \
  "$split_dir/ckpt_step${target_step}.sharded"; then
  final_marker_exact=true
fi

if run_comparison final \
  "$continuous_dir/ckpt_step${target_step}" \
  "$split_dir/ckpt_step${target_step}" \
  "$final_report" $((target_step - 1)); then
  final_exact=true
else
  final_exact=false
fi

verify_source_unchanged

if [[ "$final_exact" != true || "$final_ppl_exact" != true || \
      "$final_marker_exact" != true ]]; then
  write_terminal_result FAIL_CONTINUITY true false
  echo "R2C_FAIL_CONTINUITY midpoint exact; post-restart endpoint differs semantic=$final_exact ppl=$final_ppl_exact marker=$final_marker_exact" >&2
  exit 1
fi

write_terminal_result PASS_CONTINUITY true true
echo "R2C_PASS_CONTINUITY midpoint=exact final=exact ppl_mid=$continuous_mid_ppl ppl_final=$continuous_final_ppl"

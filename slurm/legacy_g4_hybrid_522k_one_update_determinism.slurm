#!/bin/bash
#SBATCH --job-name=hlm5-g4-hybrid-r2d-determinism
#SBATCH --account=aifac_l14_039
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=24
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --dependency=singleton
#SBATCH --exclude=lrdn0289,lrdn1459,lrdn2516
#SBATCH --output=/leonardo_work/AIFAC_L14_039/hlm5-g4-hybrid-r2d-%j.out
#SBATCH --error=/leonardo_work/AIFAC_L14_039/hlm5-g4-hybrid-r2d-%j.err

# Two fresh loads, one identical update each, no checkpoint writes.  The result
# localizes the earliest exact divergence that invalidated R2c.

set -Eeuo pipefail

repo=/leonardo/home/userexternal/mdima000/HLM5
source_dir="$repo/runs/g4_hybrid"
source_checkpoint="$source_dir/ckpt_step522000"
source_marker="$source_dir/ckpt_step522000.sharded"
work_parent=/leonardo_work/AIFAC_L14_039
work_root="$work_parent/hlm5-g4-hybrid-r2d"
run_a="$work_root/run-a"
run_b="$work_root/run-b"
attempt="$work_root/attempt.json"
environment_record="$work_root/environment.json"
diagnosis="$work_root/diagnosis.json"
result="$work_root/result.json"
failure="$work_root/failure.json"
run_a_log="$work_root/run-a.log"
run_b_log="$work_root/run-b.log"
launcher="$repo/slurm/legacy_g4_hybrid_522k_one_update_determinism.slurm"
protocol="$repo/docs/legacy-g4-hybrid-r2d-one-update-determinism-contract-2026-08-11.md"
trainer="$repo/research/legacy_g4_recovery/train_hlm5_lm_fineweb_stateful.py"
probe="$repo/research/legacy_g4_recovery/probe_one_update_determinism.py"
comparator="$repo/research/legacy_g4_recovery/compare_one_update_reports.py"
train_bin=/leonardo_scratch/large/userexternal/mdima000/hlm3/data/fineweb/fineweb_22b_train.bin

source_bytes_expected=43294668998
source_aggregate_expected=e32a2b0945e2197cc080d585bd16c7a33a817d63a6e5d96fb436127f8fdecae8
source_marker_sha_expected=8e83af00aed53b74e39aaee0ca090aef81a8232db9d0fd46aad1a16f82fcfe42
trainer_sha_expected=1f10b45c9d35bbaba346e07756c38edb681ea9bb8daee1c4c08ab4d2e58aa0c4
probe_sha_expected=65c18f987b488773584a9e6b4d367d792db417c21062c58a383e5d8e0defc991
comparator_sha_expected=2bb5ac0d04b169bd25ad2c0102d04a52f8fe268f6293fe44936256f0ea2d3654
model_sha_expected=e3080a257276b6d5b235a72cd9c7e18fb5a4d71e2a3fe8e30daebe5f029fd0d9
world_size=96
minimum_free_bytes=5000000000

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
      "{\"schema\":\"hlm5-g4-hybrid-r2d-failure-v1\",\"status\":\"${failure_status}\",\"job_id\":\"${SLURM_JOB_ID}\",\"attempt_sha256\":\"${attempt_sha}\",\"reason\":\"${failure_reason}\"}" \
      > "$failure") || true
  fi
}

fail() {
  failure_status=IMPLEMENTATION_INVALID
  failure_reason="$1"
  record_failure_artifact
  echo "R2D_IMPLEMENTATION_INVALID $*" >&2
  exit 2
}

incomplete() {
  failure_status=INCOMPLETE
  failure_reason="$1"
  record_failure_artifact
  echo "R2D_INCOMPLETE $*" >&2
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

verify_source_unchanged() {
  local count bytes aggregate
  read -r count bytes aggregate < <(checkpoint_record "$source_checkpoint" "$world_size")
  [[ "$bytes" == "$source_bytes" && "$aggregate" == "$source_aggregate" ]] || \
    fail "SOURCE_CHANGED_DURING_R2D"
}

run_probe() {
  local label="$1"
  local report_dir="$2"
  local log_path="$3"
  local rc
  [[ ! -e "$report_dir" && ! -e "$log_path" ]] || \
    fail "PROBE_OUTPUT_ALREADY_EXISTS run=$label"
  (set -o noclobber; : > "$log_path") || fail "RUN_LOG_CREATE_FAILED run=$label"
  set +e
  srun --kill-on-bad-exit=1 python -B "$probe" \
    --source-dir "$source_checkpoint" \
    --report-dir "$report_dir" \
    --run-label "$label" \
    --train-bin "$train_bin" \
    --expected-world "$world_size" \
    --expected-step 522000 > "$log_path" 2>&1
  rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    if grep -Eqi 'DUE TO TIME LIMIT|CANCELLED AT|node_fail|launch failed' "$log_path"; then
      incomplete "PROBE_EXTERNAL_INTERRUPTION run=$label rc=$rc"
    fi
    fail "PROBE_EXECUTION_FAILED run=$label rc=$rc"
  fi
  [[ -s "$report_dir/complete.json" ]] || fail "PROBE_COMPLETION_MISSING run=$label"
  [[ $(find "$report_dir" -maxdepth 1 -type f -name 'rank*.json' | wc -l) -eq 96 ]] || \
    fail "PROBE_RANK_REPORT_COUNT run=$label"
  grep -Fq "ONE_UPDATE_PROBE_COMPLETE run=$label ranks=96" "$log_path" || \
    fail "PROBE_TERMINAL_MARKER_MISSING run=$label"
}

cd "$repo"

[[ -n "${R2D_REGISTERED_LAUNCHER_SHA256:-}" ]] || \
  fail "R2D_REGISTERED_LAUNCHER_SHA256_REQUIRED"
[[ -n "${R2D_REGISTERED_PROTOCOL_SHA256:-}" ]] || \
  fail "R2D_REGISTERED_PROTOCOL_SHA256_REQUIRED"
expect_hash "$R2D_REGISTERED_LAUNCHER_SHA256" "$launcher"
expect_hash "$R2D_REGISTERED_PROTOCOL_SHA256" "$protocol"
expect_hash "$trainer_sha_expected" "$trainer"
expect_hash "$probe_sha_expected" "$probe"
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
[[ ! -e "$work_root" ]] || fail "R2D_WORK_ROOT_ALREADY_EXISTS path=$work_root"
[[ -r "$train_bin" ]] || fail "TRAIN_BIN_UNREADABLE"
available_bytes=$(df -B1 --output=avail "$work_parent" | tail -1 | tr -d ' ')
[[ "$available_bytes" -ge "$minimum_free_bytes" ]] || \
  fail "INSUFFICIENT_WORK_STORAGE available=$available_bytes"

module purge
module load profile/deeplrn
module load cineca-ai/4.1.1
export PYTHONUNBUFFERED=1
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29537

# Read-only source load before the single-use attempt is burned.
srun --kill-on-bad-exit=1 python -B "$trainer" \
  --size huge3b --batch 1 --grad-checkpoint --fsdp \
  --steps 610000 --out-dir "$source_dir" --resume auto \
  --memory-layer --memory-size 256 --stateful-checkpoints \
  --require-complete-state --expected-resume-step 522000 \
  --stop-after-resume --no-prune-checkpoints

mkdir "$work_root"
node_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd, -)
python -B - "$environment_record" "$node_list" <<'PY'
import json
import os
import platform
import sys
from pathlib import Path

import torch

matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
record = {
    "schema": "hlm5-g4-hybrid-r2d-environment-v1",
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

launcher_sha=$(sha256sum "$launcher" | cut -d' ' -f1)
protocol_sha=$(sha256sum "$protocol" | cut -d' ' -f1)
probe_sha=$(sha256sum "$probe" | cut -d' ' -f1)
comparator_sha=$(sha256sum "$comparator" | cut -d' ' -f1)
environment_sha=$(sha256sum "$environment_record" | cut -d' ' -f1)
attempt_body=$(printf '{"schema":"hlm5-g4-hybrid-r2d-attempt-v1","job_id":"%s","source_step":522000,"updates_per_launch":1,"world_size":96,"node_list":"%s","environment_sha256":"%s","launcher_sha256":"%s","protocol_sha256":"%s","probe_sha256":"%s","comparator_sha256":"%s","trainer_sha256":"%s","source_marker_sha256":"%s","source_aggregate_sha256":"%s","source_bytes":%s}' \
  "$SLURM_JOB_ID" "$node_list" "$environment_sha" "$launcher_sha" "$protocol_sha" \
  "$probe_sha" "$comparator_sha" "$trainer_sha_expected" \
  "$source_marker_sha_expected" "$source_aggregate" "$source_bytes")
create_new_json "$attempt" "$attempt_body"
attempt_created=1

run_probe a "$run_a" "$run_a_log"
run_probe b "$run_b" "$run_b_log"

set +e
python -B "$comparator" \
  --run-a "$run_a" --run-b "$run_b" \
  --result "$diagnosis" --expected-world "$world_size"
compare_rc=$?
set -e
[[ -s "$diagnosis" ]] || fail "DIAGNOSIS_MISSING rc=$compare_rc"
verify_source_unchanged

attempt_sha=$(sha256sum "$attempt" | cut -d' ' -f1)
diagnosis_sha=$(sha256sum "$diagnosis" | cut -d' ' -f1)
run_a_log_sha=$(sha256sum "$run_a_log" | cut -d' ' -f1)
run_b_log_sha=$(sha256sum "$run_b_log" | cut -d' ' -f1)
python -B - "$diagnosis" "$result" "$SLURM_JOB_ID" "$attempt_sha" \
  "$diagnosis_sha" "$run_a_log_sha" "$run_b_log_sha" <<'PY'
import json
import sys
from pathlib import Path

diagnosis = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
result = {
    "schema": "hlm5-g4-hybrid-r2d-result-v1",
    "status": diagnosis["status"],
    "classification": diagnosis["classification"],
    "first_divergent_stage": diagnosis["first_divergent_stage"],
    "job_id": sys.argv[3],
    "attempt_sha256": sys.argv[4],
    "diagnosis_sha256": sys.argv[5],
    "run_a_log_sha256": sys.argv[6],
    "run_b_log_sha256": sys.argv[7],
    "source_step": 522000,
    "updates_per_launch": 1,
    "source_unchanged": True,
}
with Path(sys.argv[2]).open("x", encoding="utf-8", newline="\n") as handle:
    json.dump(result, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
PY
result_created=1

status=$(python -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$result")
classification=$(python -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["classification"])' "$result")
if [[ "$status" == IMPLEMENTATION_INVALID ]]; then
  echo "R2D_IMPLEMENTATION_INVALID classification=$classification" >&2
  exit 2
fi
[[ "$compare_rc" -eq 0 && "$status" == VALID_DIAGNOSIS ]] || \
  fail "COMPARATOR_STATUS_RC_MISMATCH status=$status rc=$compare_rc"
echo "R2D_VALID_DIAGNOSIS classification=$classification"

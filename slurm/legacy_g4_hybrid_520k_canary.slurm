#!/bin/bash
#SBATCH --job-name=hlm5-g4-hybrid-520k-canary
#SBATCH --account=aifac_l14_039
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=24
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0289,lrdn1459,lrdn2516
#SBATCH --output=/leonardo_work/AIFAC_L14_039/hlm5-g4-hybrid-520k-canary-%j.out
#SBATCH --error=/leonardo_work/AIFAC_L14_039/hlm5-g4-hybrid-520k-canary-%j.err

# Load-only recovery canary for the legacy 96-way LOCAL_STATE_DICT checkpoint.
# It never enters the trainer and must not change the checkpoint directory.

set -euo pipefail

repo=/leonardo/home/userexternal/mdima000/HLM5
checkpoint="$repo/runs/g4_hybrid/ckpt_step520000"
marker="$repo/runs/g4_hybrid/ckpt_step520000.sharded"

expect_hash() {
  local expected="$1"
  local path="$2"
  local actual
  actual=$(sha256sum "$path" | cut -d' ' -f1)
  if [[ "$actual" != "$expected" ]]; then
    echo "HASH_MISMATCH path=$path expected=$expected actual=$actual" >&2
    exit 2
  fi
}

expect_checkpoint() {
  local count total aggregate latest_marker
  count=$(find "$checkpoint" -maxdepth 1 -type f -name 'shard_rank*.pt' | wc -l)
  [[ "$count" -eq 96 ]] || { echo "SHARD_COUNT_MISMATCH count=$count" >&2; exit 2; }
  for rank in $(seq 0 95); do
    [[ -s "$checkpoint/shard_rank${rank}.pt" ]] || {
      echo "MISSING_OR_EMPTY_SHARD rank=$rank" >&2
      exit 2
    }
  done
  total=$(find "$checkpoint" -maxdepth 1 -type f -name 'shard_rank*.pt' \
    -printf '%s\n' | awk '{sum += $1} END {printf "%.0f\n", sum}')
  [[ "$total" -eq 21670864639 ]] || {
    echo "SHARD_BYTES_MISMATCH bytes=$total" >&2
    exit 2
  }
  aggregate=$(find "$checkpoint" -maxdepth 1 -type f -name 'shard_rank*.pt' \
    -print0 | sort -z -V | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
  [[ "$aggregate" == 634a8f58e26e679323a637b4df2c726b8dd55e1964486009bf192fe8ad53f80b ]] || {
    echo "SHARD_MANIFEST_MISMATCH aggregate=$aggregate" >&2
    exit 2
  }
  latest_marker=$(find "$repo/runs/g4_hybrid" -maxdepth 1 -type f \
    -name 'ckpt_step*.sharded' -printf '%f\n' | sort -V | tail -1)
  [[ "$latest_marker" == ckpt_step520000.sharded ]] || {
    echo "LATEST_MARKER_MISMATCH marker=$latest_marker" >&2
    exit 2
  }
  echo "CHECKPOINT_BOUND count=$count bytes=$total aggregate=$aggregate"
}

cd "$repo"

expect_hash 6f7c425ff46a7edc8cf382d3ca66b4e547dc5b71b7c39a9698fac3d0a41f5014 "$marker"
expect_hash f74bf513719f8c3a110cbf45296889ad5914b3ca1cf7b5ba8538195d123d42f9 \
  "$repo/train_hlm5_lm_fineweb.py"
expect_hash e3080a257276b6d5b235a72cd9c7e18fb5a4d71e2a3fe8e30daebe5f029fd0d9 \
  "$repo/hlm5_lm.py"
expect_hash 3acd55ffc8789bcfafa2e72fc2aad6127e758865043f93767e28e9514952dd8e \
  "$repo/tools/validate_g4_fsdp_resume.py"
expect_checkpoint

module purge
module load profile/deeplrn
module load cineca-ai/4.1.1
export PYTHONUNBUFFERED=1
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29526

srun --kill-on-bad-exit=1 python -B tools/validate_g4_fsdp_resume.py \
  --report-label hybrid_520000_recovery \
  --size huge3b \
  --out-dir runs/g4_hybrid \
  --grad-checkpoint \
  --memory-layer \
  --memory-size 256

# Prove that the read-only validation did not mutate the bound checkpoint.
expect_hash 6f7c425ff46a7edc8cf382d3ca66b4e547dc5b71b7c39a9698fac3d0a41f5014 "$marker"
expect_checkpoint
echo "REPAIR_CANARY_PASS checkpoint=ckpt_step520000 world=96"

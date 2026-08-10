#!/usr/bin/env bash
#SBATCH --job-name=hlm5-e7-3b
#SBATCH --account=aifac_l14_039
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_bprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=e7-3b-%j.out

set -euo pipefail

: "${HLM5_E7_REPO:?Set HLM5_E7_REPO to the committed E7 worktree on Leonardo}"
: "${HLM5_E7_SNAPSHOT:?Set HLM5_E7_SNAPSHOT to the pinned complete snapshot}"

cd "${HLM5_E7_REPO}"
export HLM5_E7_SNAPSHOT
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python scripts/preflight_e7_3b.py
python scripts/run_e7_3b.py
python scripts/verify_e7_3b.py

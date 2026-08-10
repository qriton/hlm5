#!/bin/bash
#SBATCH --job-name=hlm5-e10-3b
#SBATCH --account=aifac_l14_039
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/leonardo_work/AIFAC_L14_039/hlm5-e10-%j.out
#SBATCH --error=/leonardo_work/AIFAC_L14_039/hlm5-e10-%j.err

set -euo pipefail

module purge
module load profile/deeplrn
module load cineca-ai/4.1.1
source /leonardo_work/AIFAC_L14_039/hlm5-e7b-env/bin/activate
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export HLM5_E7_SNAPSHOT=/leonardo_work/AIFAC_L14_039/hlm5-e7b-snapshot-d78a42

repo=/leonardo_work/AIFAC_L14_039/hlm5-e10-f6086d6
cd "$repo"

test "$(git rev-parse HEAD)" = "f6086d655469ebea78e076924786d4978729c036"
test -z "$(git status --porcelain)"
test "$(sha256sum scripts/data/counterfact.json | cut -d' ' -f1)" = "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"

python scripts/preflight_e10_3b.py
python scripts/register_e10_3b.py
python scripts/admit_e10_3b.py
python scripts/replay_e10_3b.py
python scripts/verify_e10_3b.py

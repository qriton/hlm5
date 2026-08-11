#!/bin/bash
#SBATCH --job-name=hlm5-e14-3b
#SBATCH --account=aifac_l14_039
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/leonardo_work/AIFAC_L14_039/hlm5-e14-%j.out
#SBATCH --error=/leonardo_work/AIFAC_L14_039/hlm5-e14-%j.err

set -euo pipefail

module purge
module load profile/deeplrn
module load cineca-ai/4.1.1
source /leonardo_work/AIFAC_L14_039/hlm5-e7b-env/bin/activate
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export HLM5_E7_SNAPSHOT=/leonardo_work/AIFAC_L14_039/hlm5-e7b-snapshot-d78a42

repo=/leonardo_work/AIFAC_L14_039/hlm5-e14
cd "$repo"

test -z "$(git status --porcelain --untracked-files=no)"
test "$(sha256sum scripts/data/counterfact.json | cut -d' ' -f1)" = "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"

python scripts/preflight_e14_3b.py
python scripts/register_e14_3b.py
python scripts/admit_e14_3b.py
python scripts/replay_e14_3b.py
python scripts/verify_e14_3b.py

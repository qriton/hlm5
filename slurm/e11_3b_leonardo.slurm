#!/bin/bash
#SBATCH --job-name=hlm5-e11-3b
#SBATCH --account=aifac_l14_039
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --exclude=lrdn1455
#SBATCH --output=/leonardo_work/AIFAC_L14_039/hlm5-e11-%j.out
#SBATCH --error=/leonardo_work/AIFAC_L14_039/hlm5-e11-%j.err

set -euo pipefail
module purge
module load profile/deeplrn
module load cineca-ai/4.1.1
source /leonardo_work/AIFAC_L14_039/hlm5-e7b-env/bin/activate
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export HLM5_E7_SNAPSHOT=/leonardo_work/AIFAC_L14_039/hlm5-e7b-snapshot-d78a42

repo=/leonardo_work/AIFAC_L14_039/hlm5-e11-7a43980
cd "$repo"
test "$(git rev-parse HEAD)" = "7a43980ac910f4be9e2997a1eaef67063a8bc711"
test -z "$(git status --porcelain)"
test "$(sha256sum scripts/data/counterfact.json | cut -d' ' -f1)" = "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"

python scripts/preflight_e11_3b.py
python scripts/register_e11_3b.py
python scripts/admit_e11_3b.py
python scripts/replay_e11_3b.py
python scripts/verify_e11_3b.py

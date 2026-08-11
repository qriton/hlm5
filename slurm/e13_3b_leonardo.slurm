#!/bin/bash
#SBATCH --job-name=hlm5-e13-3b
#SBATCH --account=aifac_l14_039
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=lrdn1455,lrdn2661,lrdn2974
#SBATCH --output=/leonardo_work/AIFAC_L14_039/hlm5-e13-%j.out
#SBATCH --error=/leonardo_work/AIFAC_L14_039/hlm5-e13-%j.err

set -euo pipefail

module purge
module load profile/deeplrn
module load cineca-ai/4.1.1
source /leonardo_work/AIFAC_L14_039/hlm5-e7b-env/bin/activate
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export HLM5_E7_SNAPSHOT=/leonardo_work/AIFAC_L14_039/hlm5-e7b-snapshot-d78a42

repo=/leonardo_work/AIFAC_L14_039/hlm5-e13-c75a8b8
cd "$repo"

test "$(git rev-parse HEAD)" = "c75a8b87808658d5bde52f54eaa5c7439619ed07"
test -z "$(git status --porcelain --untracked-files=no)"
test "$(sha256sum scripts/data/counterfact.json | cut -d' ' -f1)" = "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"

python scripts/preflight_e13_3b.py
python scripts/register_e13_3b.py
python scripts/admit_e13_3b.py
python scripts/replay_e13_3b.py
python scripts/verify_e13_3b.py

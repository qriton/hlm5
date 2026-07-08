# EasyEdit reference-baseline job (Linux / Leonardo) — fill the Table 1 gap

Goal: produce the reference numbers for **ROME, MEMIT, MEND, SERAC, GRACE** (and
re-confirm our ROME) on **CounterFact** and **zsRE**, in the same metric shape as
the GPT-2 controlled study (`baselines/run_gpt2_table1_v2.py`), so the remaining
cells of the paper's Table 1 can be filled with measured values.

**Why not on the dev box:** EasyEdit's package `__init__` imports its multimodal
editors (blip2 → `timm` → `torchaudio`), and the Windows `libtorchaudio.pyd` is
broken, so `import easyeditor` fails. On Linux the full requirement set installs
cleanly. Run this there (a Leonardo GPU node is ideal).

---

## 1. Environment

```bash
git clone https://github.com/zjunlp/EasyEdit.git
cd EasyEdit
conda create -n easyedit python=3.10 -y && conda activate easyedit
# EasyEdit currently targets torch 2.x / transformers 5.x (see requirements.txt)
pip install -r requirements.txt
# datasets: EasyEdit ships data loaders; fetch CounterFact + zsRE
#   data live under EasyEdit/data/ (see EasyEdit README "Datasets")
```

If a slim install is preferred, ROME/MEMIT/MEND/SERAC need only:
`torch transformers datasets hydra-core omegaconf einops higher sentence-transformers
nltk scikit-learn` — the multimodal extras (`timm opencv-python av qwen_vl_utils
fairscale torchaudio`) are not needed for text editing.

## 2. Models and data

- **Base model:** `gpt2-xl` (1.5B) — the model ROME/MEMIT/SERAC report on, so our
  numbers sit next to the literature. Optionally add `EleutherAI/gpt-j-6B`.
- **Benchmarks:** CounterFact (efficacy / generalization / **neighborhood
  locality** / portability) and zsRE. EasyEdit ships hparams for these under
  `hparams/<METHOD>/gpt2-xl.yaml`.
- **MEND / SERAC** require **training the editor first** on the dataset before
  evaluation (one-time, ~GPU-day each); ROME/MEMIT/GRACE are train-free per edit.

## 3. Runner (per method) — uses EasyEdit's BaseEditor

```python
# run_easyedit_baselines.py  (run inside the EasyEdit repo, easyedit env)
import json, sys
from easyeditor import BaseEditor
from easyeditor import ROMEHyperParams, MEMITHyperParams, MENDHyperParams, SERACHparams, GraceHyperParams

HP = {"ROME": ("ROME", ROMEHyperParams), "MEMIT": ("MEMIT", MEMITHyperParams),
      "MEND": ("MEND", MENDHyperParams), "SERAC": ("SERAC", SERACHparams),
      "GRACE": ("GRACE", GraceHyperParams)}

def run(method, prompts, subjects, targets, rephrases, locality_inputs):
    name, cls = HP[method]
    hp = cls.from_hparams(f"hparams/{name}/gpt2-xl.yaml")
    editor = BaseEditor.from_hparams(hp)
    metrics, _, _ = editor.edit(
        prompts=prompts, subject=subjects, target_new=targets,
        rephrase_prompts=rephrases,            # -> generalization
        locality_inputs=locality_inputs,       # -> neighborhood locality
        sequential_edit=False)
    # EasyEdit returns per-edit rewrite/rephrase/locality accuracy; aggregate:
    agg = aggregate(metrics)   # mean reliability/generalization/locality(/portability)
    json.dump(agg, open(f"results/{method.lower()}_counterfact_gpt2xl.json", "w"), indent=2)
```

Drive it over CounterFact (and zsRE) with EasyEdit's dataset classes
(`from easyeditor import CounterFactDataset, ZsreDataset`), which already provide
`prompt/subject/target_new/rephrase/locality` fields.

## 4. HLM5 on the same base + data (so HLM5 is in the same table)

Attach the HLM5 memory hook to `gpt2-xl` and score with the **same** CounterFact
splits, reusing the metric code from `baselines/run_gpt2_table1_v2.py`
(efficacy = rewrite flip, generalization = rephrase flip, locality = neighborhood
unchanged). This yields the HLM5 row measured identically to the editors. Run both
the deployed (whitened, τ=0.95) and loose-gate operating points.

## 5. Metric harmonization (map to the paper's Table 1)

| EasyEdit term | Paper column | Note |
| --- | --- | --- |
| Reliability / rewrite acc | Efficacy | edit takes effect |
| Generalization (rephrase) | Generalization | CounterFact rephrase split |
| Locality / neighborhood | Locality | **on-support** neighborhood test (the one we lack) |
| Portability | Portability | multi-hop; report if available |
| n/a | Grad steps / cost | 0 for HLM5/GRACE-read; >0 for ROME/MEMIT/MEND/FT |

CounterFact targets are often multi-token; for the HLM5 single-token column, score
**first-token** efficacy and state it, or restrict to the single-token CounterFact
subset for an apples-to-apples row (recommended for the main table, with the
full-target numbers in an appendix).

## 6. Output → paper

Each method writes `results/<method>_<dataset>_<model>.json` with keys
`{efficacy, generalization, locality, portability, grad_steps}`. A small
`compare_table1.py` reads them and emits the LaTeX rows, replacing the
"pending" cells in `Table~\ref{tab:main}` of `docs/hlm5-paper.tex`. Keep the GPT-2
controlled study (Table 1, current) as the small-model panel; add a CounterFact/
zsRE panel from this job.

## 7. Leonardo SLURM template

```bash
#!/bin/bash
#SBATCH --job-name=hlm5-editbaselines
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1 --gres=gpu:1 --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=editbaselines_%j.out
module load profile/deeplrn cuda     # adjust to the current Leonardo modules
source activate easyedit
cd $WORK/EasyEdit
for M in ROME MEMIT GRACE; do python run_easyedit_baselines.py --method $M --data counterfact --model gpt2-xl; done
# MEND/SERAC: train the editor first, then eval
python train_editor.py --method MEND --data counterfact --model gpt2-xl
python run_easyedit_baselines.py --method MEND --data counterfact --model gpt2-xl
python run_hlm5_on_gpt2xl.py --data counterfact --gate deployed
python run_hlm5_on_gpt2xl.py --data counterfact --gate loose
python compare_table1.py results/ > table1_counterfact.tex
```

## 8. Compute estimate

| Item | Cost |
| --- | --- |
| ROME/MEMIT/GRACE × CounterFact (~2k edits), gpt2-xl | a few GPU-hours each |
| MEND / SERAC editor training | ~1 GPU-day each |
| HLM5 attach + eval (no training) | minutes–1 hour |
| **Total** | **~1 GPU-day** for the train-free set; +2 days if MEND/SERAC trained |

## 9. Acceptance check

Before trusting the table, reproduce one published ROME CounterFact reliability
number (≈0.99 efficacy, high generalization, high neighborhood locality on
gpt2-xl) as a harness sanity check. Then the HLM5 row is directly comparable and
the paper's Table 1 "pending" cells become measured.

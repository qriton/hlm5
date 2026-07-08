# scripts/

All experiment scripts, migrated onto the installed `hlm5` package (portable
paths, `main()` guards, `weights_only=True`; see the repo-root migration task).
This includes the "copied producers" table below (externally-sourced artifacts,
copied verbatim from the private working repo) as well as the
on-trunk certificate/gate/editing scripts (`run_1b_*.py`, `run_gpt2_*.py`,
`run_rome_*.py`, `run_editors_counterfact.py`, `run_cert_counterfact_gpt2xl.py`,
`analyze_cert_vs_editors.py`, `read_and_memorize.py`, etc).

**Per-script requirements matrix (script → output artifact → requirements →
finding) lives in the root [`../README.md`](../README.md#scripts--artifact--finding)**
— that table is the single home for the full script inventory, including which
scripts need a GPU, a checkpoint, EasyEdit, or a data fetch. This file keeps the
provenance notes for the copied producers and the not-copied computations below.

## Copied producers

| Script | Produces (`results/`) |
| --- | --- |
| `run_hlm5_capacity_search.py` | `hlm5_capacity_search_report.json`, `capacity_search/random_p*.json` |
| `run_train_key_value.py` | capacity-search trainer invoked by `run_hlm5_capacity_search.py` (writes wherever `--json-out` points, e.g. `capacity_search/random_p*.json`) |
| `run_hlm5_scale_suite.py` | `hlm5_scale_suite_report.json` |
| `run_lm_kb_inject.py` | `lm_kb_inject.json`, `lm_kb_inject_d256.json`, `lm_kb_inject_d256_b70.json` |
| `run_incontext_mqar.py` | `incontext_mqar.json` |
| `run_edit_receipt_demo.py` | `edit_receipts.json` |
| `run_fineweb_kb_inject.py` | `g2c_rare.json` (`--target-mode rare --k 32`); also the source of the G2b/G3b/G3c spread/rare gate JSONs, invoked with different `--target-mode`/`--k`/`--checkpoint` flags |

`verify_artifacts.py` loads every copied JSON in `results/` and asserts the
paper's load-bearing numbers against them; run with `python scripts/verify_artifacts.py`.

### Note: hardcoded private path in `run_fineweb_kb_inject.py`

`run_fineweb_kb_inject.py` line 31 hardcodes
`FINEWEB_VAL = "/leonardo_scratch/large/userexternal/redacted/hlm3/data/fineweb/fineweb_22b_val.bin"`,
a private Leonardo-cluster scratch path. Copied as-is per instructions (path
migration is a separate task) — this constant needs to become a CLI flag or
be pointed at a released copy of the FineWeb validation shard before the
script is independently runnable. `open_mmap()` now raises a clear
`FileNotFoundError` naming this file when `--val-bin`/`--train-bin` point at a
path that doesn't exist on the current machine, instead of failing with a bare
OS error.

### Note: `hlm5_lm`/`hlm5_memory`/`hlm5_edit_audit` imports now ported

`run_edit_receipt_demo.py`, `run_fineweb_kb_inject.py`, `run_incontext_mqar.py`,
and `run_lm_kb_inject.py` previously imported from `hlm5_lm`, `hlm5_memory`, and
(for `run_edit_receipt_demo.py`) `hlm5_edit_audit` — private modules from the
`the private working repo` working repo that were never copied into this bundle. That gap is now
closed: `hlm5_edit_audit.py` was ported into the package as `hlm5/edit_audit.py`
(byte-identical; stdlib + torch only), and the four scripts now import
`hlm5.model.HLM5LM`, `hlm5.memory.EditableHLM5Memory`/`unit`/
`TinyTransformerWithHLM5`, and `hlm5.edit_audit.issue_receipt`/`verify_receipt`/
`trunk_hash` from the released `hlm5` package. `run_train_key_value.py` was
likewise ported: its model class moved into the package as
`hlm5.key_value.HLM5KeyValueModel` (formerly the private `models/` module), and
the script now imports `from hlm5.key_value import HLM5KeyValueModel`. It is
the capacity-search trainer invoked by `run_hlm5_capacity_search.py` (see the
copied-producers table above).

`run_fineweb_kb_inject.py` still needs Leonardo-cluster data to actually run
end-to-end — see the hardcoded scratch path note above — but it now imports
cleanly. Its residual-hook mode also calls `model.residual_state`/
`model.attach_residual_memory`, which are not present on `hlm5.model.HLM5LM`;
that scaffolding gap is unrelated to the import fix and remains open.

`run_headgeom_gpt2xl.py`, `run_ownslot_weight.py`,
`run_multikey_category_test.py`, `verify_artifacts.py`, and every script under
`code/`'s former on-trunk certificate/gate/editing set (now `run_1b_*.py`,
`run_gpt2_*.py`, `run_rome_*.py`, `run_editors_counterfact.py`,
`run_cert_counterfact_gpt2xl.py`, `analyze_cert_vs_editors.py`,
`read_and_memorize.py`) import only the released `hlm5` package and were never
affected.

## Not copied: no-tax gate computation (G2a/G3a)

`g2a_no_tax.json` and `g3a_no_tax.json` (the matched-pair "no-tax" delta
between baseline and hybrid validation PPL) are computed by an inline Python
heredoc inside the SLURM job scripts, not a standalone script:
`leonardo/run_g2_evals.sbatch` (G2a) and `leonardo/run_g3_evals.sbatch` (G3a).
Per the release-prep constraints, SLURM/sbatch files are not copied into this
bundle. The full inline computation (verbatim from `leonardo/run_g3_evals.sbatch`,
G2a is identical up to the file names) is:

```python
import json
base = json.load(open("runs/g3_baseline/train_summary.json"))
hyb = json.load(open("runs/g3_hybrid/train_summary.json"))
b, h = base["best_val_ppl"], hyb["best_val_ppl"]
delta = abs(h - b) / b
print(f"baseline val PPL {b:.2f} | hybrid val PPL {h:.2f} | delta {100*delta:.2f}%")
print(f"G3a_no_tax<=5pct: {'PASS' if delta <= 0.05 else 'FAIL'}")
json.dump({"baseline_val_ppl": b, "hybrid_val_ppl": h, "delta_pct": 100*delta,
           "G3a_pass": delta <= 0.05}, open("reports/g3a_no_tax.json", "w"), indent=2)
```

This reads `best_val_ppl` straight out of `baseline_train_summary.json` /
`hybrid_train_summary.json` (both already in `results/`), so the no-tax
numbers are independently reproducible from artifacts already in this bundle
without needing the sbatch file itself.

## Not found: G2c training/eval harness beyond `run_fineweb_kb_inject.py`

The G2/G3 eval sbatch scripts (`leonardo/run_g2_evals.sbatch`,
`leonardo/run_g3_evals.sbatch`) that orchestrate calls to
`run_fineweb_kb_inject.py` with the specific `--target-mode`/`--k`/`--checkpoint`
flags used for each gate are themselves SLURM job files and were not copied
for the same reason as above.

# scripts/

All experiment scripts use the installed `hlm5` package, portable paths,
`main()` guards, and `weights_only=True` where the released checkpoint format
supports it. This includes the producer table below as well as the on-trunk
certificate/gate/editing scripts (`run_1b_*.py`, `run_gpt2_*.py`,
`run_rome_*.py`, `run_editors_counterfact.py`, `run_cert_counterfact_gpt2xl.py`,
`analyze_cert_vs_editors.py`, `read_and_memorize.py`, etc).

Certificate/dosing scripts use the 2026-07-13 hardened convention from
`hlm5.certify`: float64 certificate arithmetic, `EPS = 0.0`, and exact slope
signs. Negative slopes are always upper-bound constraints; do not reintroduce a
near-zero slope dead-zone in copied experiment code.

**Per-script requirements matrix (script → output artifact → requirements →
finding) lives in the root [`../README.md`](../README.md#scripts--artifact--finding)**
— that table is the single home for the full script inventory, including which
scripts need a GPU, a checkpoint, EasyEdit, or a data fetch. This file keeps
reproducibility notes for producers that need external data or cluster-only
inputs.

## Producers

| Script | Produces (`results/`) |
| --- | --- |
| `run_hlm5_capacity_search.py` | `hlm5_capacity_search_report.json`, `capacity_search/random_p*.json` |
| `run_train_key_value.py` | capacity-search trainer invoked by `run_hlm5_capacity_search.py` (writes wherever `--json-out` points, e.g. `capacity_search/random_p*.json`) |
| `run_hlm5_scale_suite.py` | `hlm5_scale_suite_report.json` |
| `run_lm_kb_inject.py` | `lm_kb_inject.json`, `lm_kb_inject_d256.json`, `lm_kb_inject_d256_b70.json` |
| `run_incontext_mqar.py` | `incontext_mqar.json` |
| `run_edit_receipt_demo.py` | `edit_receipts.json` |
| `run_fineweb_kb_inject.py` | `g2c_rare.json` (`--target-mode rare --k 32`); also the source of the G2b/G3b/G3c spread/rare gate JSONs, invoked with different `--target-mode`/`--k`/`--checkpoint` flags |

`verify_artifacts.py` loads every JSON in `results/` and asserts the
paper's load-bearing numbers against them; run with `python scripts/verify_artifacts.py`.

### Note: FineWeb data for `run_fineweb_kb_inject.py`

`run_fineweb_kb_inject.py` requires a local FineWeb uint16 shard with the HLM3
header. Pass it explicitly with `--val-bin`; rare-target mode also needs
`--train-bin` or a train shard inferable from the validation path. The script
raises a clear `FileNotFoundError` when the supplied shard path is missing.

### Note: `hlm5_lm`/`hlm5_memory`/`hlm5_edit_audit` imports now ported

`run_edit_receipt_demo.py`, `run_fineweb_kb_inject.py`, `run_incontext_mqar.py`,
and `run_lm_kb_inject.py` import released package modules:
`hlm5.model.HLM5LM`, `hlm5.memory.EditableHLM5Memory`/`unit`/
`TinyTransformerWithHLM5`, and `hlm5.edit_audit.issue_receipt`/`verify_receipt`/
`trunk_hash`. `run_train_key_value.py` imports
`hlm5.key_value.HLM5KeyValueModel`; it is the capacity-search trainer invoked by
`run_hlm5_capacity_search.py` (see the producers table above).

`run_fineweb_kb_inject.py` still needs external FineWeb shards to run end-to-end,
but it imports cleanly. Its residual-hook mode also calls `model.residual_state`/
`model.attach_residual_memory`, which are not present on `hlm5.model.HLM5LM`;
that scaffolding gap is unrelated to the import fix and remains open.

`run_headgeom_gpt2xl.py`, `run_ownslot_weight.py`,
`run_multikey_category_test.py`, `verify_artifacts.py`, and every script under
`code/`'s former on-trunk certificate/gate/editing set (now `run_1b_*.py`,
`run_gpt2_*.py`, `run_rome_*.py`, `run_editors_counterfact.py`,
`run_cert_counterfact_gpt2xl.py`, `analyze_cert_vs_editors.py`,
`read_and_memorize.py`) import only the released `hlm5` package and were never
affected.

## No-tax gate computation (G2a/G3a)

`g2a_no_tax.json` and `g3a_no_tax.json` (the matched-pair "no-tax" delta
between baseline and hybrid validation PPL) are computed directly from the
training-summary JSONs. The computation is:

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

## G2/G3 Injection Gates

The released script for spread/rare injection is `run_fineweb_kb_inject.py`.
Reproducing the exact gate artifacts requires the released checkpoint plus local
FineWeb shards supplied via `--val-bin` and, for rare mode, `--train-bin`.

# HLM5 — Certified Knowledge Editing

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21258599.svg)](https://doi.org/10.5281/zenodo.21258599)

**Certified Knowledge Editing: Closed-Form Reachability Guarantees for Additive
Edits to a Frozen Language Model.**
Marius Dima (Qriton). Paper: [`paper/hlm5-paper.pdf`](paper/hlm5-paper.pdf) · arXiv
link forthcoming.

HLM5 attaches a slot memory to a *frozen* decoder-only transformer and asks the
reverse of the usual editing question: instead of installing a fact and checking
afterward whether it took and stayed local, it computes — in closed form, before
the edit — a certificate that accepts or refuses each single-token edit and, if it
accepts, doses it at the certified strength. Because the gate and the read see only
the unperturbed hidden state, the post-edit logits are affine in the edit strength,
so single-token reachability is one-dimensional linear feasibility solvable in one
pass over the head matrix. Every edit, forget, and rollback happens outside the
trunk weights and emits a verifiable receipt.

This repository is the self-contained bundle: the `hlm5` package, every experiment
script, every result JSON the paper cites, the paper sources, and loader
instructions for the released 1B checkpoint.

## Quickstart

```bash
git clone https://github.com/qriton/hlm5.git
cd hlm5
pip install -e .                 # add [experiments] for the GPT-2 / figure scripts
```

Download the frozen 1B trunk (~4.2 GB) from Hugging Face into `models/`:

```bash
huggingface-cli download qriton/hlm5-1b-trunk \
  hlm5_lm_baseline_fineweb_g3_final.pt \
  --local-dir models
```

(The 65,536-token BPE tokenizer already ships in this repository at
`models/tokenizers/fineweb-65536-compat.json` — no download needed.)

Then run the read-and-memorize demo — it reads a briefing, admits each candidate
fact through the certificate, memorizes the admitted ones at their certified
strength, and verifies recall, locality, and rollback (CPU, ~45 s):

```bash
python scripts/read_and_memorize.py
```

See [`models/README.md`](models/README.md) for checkpoint hashes and loading, and
the loader in [`hlm5/io.py`](hlm5/io.py) (`load_trunk`, `load_tokenizer`).

## Headline results

Every row cites the JSON under [`results/`](results/) it is read from. The deployed
pipeline is the certificate-governed 17-fact evaluation in
`hlm5_1b_faithful_certdosed.json` (frozen 1B trunk, whitened degree-5 gate,
threshold 0.95); 95% bootstrap CIs are over facts.

| Result | Value | Artifact |
| --- | --- | --- |
| Certificate-dosed efficacy | **0.765** (13/17), 95% CI [0.529, 0.941] | `hlm5_1b_faithful_certdosed.json` (`cert_naive` arm) |
| + residual-synthesis rescue | **1.000** (17/17) | `hlm5_1b_faithful_certdosed.json` (`cert_synth` arm) |
| Global-boost dosing ablation | 0.529 (9/17) | `hlm5_1b_faithful_certdosed.json` (`global_repro` arm) |
| Locality (neutral prompts) | **1.000** (8/8, bit-identical logits) | `hlm5_1b_faithful_certdosed.json` |
| Paraphrase transfer | 0.020 (exact-key by design) | `hlm5_1b_faithful_certdosed.json` |
| Operating envelope, single key | 78.4% of 1,200 targets reachable | `cert_envelope_synth.json` |
| Operating envelope, 60 keys | 79.4% ± 2.2% reachable | `cert_envelope_multikey.json` |
| Readout-level synthesis rescue | 40/40 sampled unreachable targets flip (median β = 20) | `synth_verify.json`, `cert_envelope_synth.json` |
| No-tax (matched baseline vs. hybrid) | \|ΔPPL\| = 0.018% at 136M, 0.116% at 1B | `g2a_no_tax.json`, `g3a_no_tax.json`, `notax_params.json` |
| Gate selectivity | perfectly separable (exact-key 1.0, all off-target 0.0); own-slot weight 0.9993 | `gate_roc.json`, `ownslot_weight_certdosed.json` |
| GPT-2 controlled study (incl. ROME) | HLM5 vs. logit-bias McNemar p = 6e-5; ROME re-impl 0.815/0.854, EasyEdit ref 0.972/0.771 | `gpt2_table1_v2.json`, `rome_gpt2.json`, `rome_easyedit_gpt2.json` |
| Certificate predicts editors (CounterFact, exploratory) | ROME paraphrase ρ = −0.16, p = 0.005; FT ρ = −0.13, p = 0.023; neighborhood damage ROME/GRACE p < 0.01 | `cert_vs_editors_analysis.json` |

The reachability frontier is the honest core: a *global* edit strength overshoots
the certified interval and silently fails reachable edits (0.529); the certified
per-fact β★ flips exactly the certificate-reachable set (0.765); residual-synthesis
rescue (full memory path) flips the rest (1.000) — all at locality 1.000, zero gradients, and
paraphrase transfer 0.020. A gate-matched static logit bias matches the dosed
pipeline on every measured axis; the value is the a-priori admission test,
certified dose, synthesis rescue, and audit trail, not raw editing power.

## Repository layout

```
hlm5-release/
  hlm5/        installed package: model.py, memory.py, certify.py,
               edit_audit.py, key_value.py, io.py (loaders)
  scripts/     every experiment script + producers (see scripts/README.md)
  results/     every result JSON the paper cites + RESULTS.md (honest log)
  paper/       hlm5-paper.tex/.pdf, twocol; docx via paper/build.py; figures/
  models/      1B checkpoints (gitignored; download from HF) + tokenizer
  tests/       test_certify.py, test_smoke.py (CPU)
  LICENSE  CITATION.cff  pyproject.toml  RELEASING.md
```

## Scripts → artifact → finding

Every quantitative claim in the paper is produced by one of these scripts and
lands in `results/`. Requirements legend: **CPU** runs without a GPU; **GPU**
wants CUDA; **1B** needs the released baseline checkpoint; **hybrid** needs the
co-trained checkpoint (publication pending, see [`RELEASING.md`](RELEASING.md));
**136M** needs the unreleased 136M gate checkpoints; **GPT-2**/**GPT-2-XL**
auto-download from Hugging Face; **EasyEdit** needs an external clone;
**CounterFact** needs `fetch_counterfact.py` first; **Leonardo** needs private
cluster data.

### On-trunk certificate, gate, and dosing (frozen 1B)

| Script | Output (`results/`) | Requirements | Finding |
| --- | --- | --- | --- |
| `run_1b_certificate.py` | `cert_envelope_synth.json` | 1B, GPU | Operating envelope: 78.4% of 1,200 targets reachable; residual synthesis rescues 40/40 sampled unreachable |
| `run_1b_envelope_multikey.py` | `cert_envelope_multikey.json` | 1B, GPU | Envelope is stable across keys: 79.4% ± 2.2% reachable over 60 keys |
| `run_1b_betastar.py` | `betastar.json` | 1B, GPU | β★ selection lifts worst-case margin 1.08 → 18.5 (17×) |
| `run_1b_synth_verify.py` | `synth_verify.json` | 1B, GPU | End-to-end rescue: 0/40 flip with `unit(W_y)`, 40/40 flip with synthesized r★ (median β = 20) |
| `run_1b_headgeom_metrics.py` | `headgeom_metrics.json` | 1B, GPU | Reachability is head-geometry (norm-dependent), not a memory property |
| `run_1b_gate_roc.py` | `gate_roc.json` | 1B, GPU | Gate perfectly separable: exact-key 1.0, every paraphrase/typo/other-relation 0.0 |
| `run_ownslot_weight.py` | `ownslot_weight_certdosed.json` | 1B, GPU | Own-slot softmax weight 0.9993 at the deployed 17-fact setting (gate fires 17/17) |
| `run_1b_faithful_certdosed.py` | `hlm5_1b_faithful_certdosed.json` | 1B, GPU | **Deployed pipeline:** dosing ablation 0.529 → certified 0.765 → +synthesis 1.000, locality 1.000 |
| `run_1b_faithful.py` | `hlm5_1b_faithful.json` | 1B, GPU | Source of the 0.529 global-boost row (dosing ablation; superseded by certdosed) |
| `run_1b_gate_sweep.py` | `hlm5_1b_gate_sweep.json` | 1B, GPU | Generalization-vs-locality trade-off curve as the gate threshold varies |
| `run_1b_table_row.py` | `hlm5_1b_table_row.json` | 1B, GPU | On-trunk HLM5 vs. logit-bias vs. retrieval (efficacy 0.765) |
| `run_multikey_category_test.py` | `multikey_category_test.json` | 1B, GPU | Reachable fraction does not differ across novel/real/generic keys (KW p = 0.86) |
| `run_notax_params.py` | `notax_params.json` | 1B, hybrid, 136M | Parameter counts for the matched-pair no-tax table (baseline 1B ≈ 1.045B params) |
| `run_seq_edit_stream.py` | `seq_edit_stream.json` | 1B, GPU | Latency flat to 1,024 edits (~2.3 ms/query); efficacy confounded — see `RESULTS.md` |
| `read_and_memorize.py` | `read_and_memorize_receipts.json` | 1B, CPU | End-to-end governed read→admit→memorize→verify demo with receipts |

### GPT-2 controlled study and locate-and-edit baselines

| Script | Output (`results/`) | Requirements | Finding |
| --- | --- | --- | --- |
| `run_gpt2_table1_v2.py` | `gpt2_table1_v2.json` | GPT-2, GPU | Hardened study: HLM5 vs. logit-bias McNemar p = 6e-5; vs. FT-full/FT-L/RAG |
| `run_gpt2_table1.py` | `gpt2_table1.json` | GPT-2, GPU | First-pass GPT-2 study — **superseded** by `run_gpt2_table1_v2.py` |
| `run_rome_gpt2.py` | `rome_gpt2.json` | GPT-2, GPU | Our ROME re-implementation: 1.0 / 0.815 / 0.854 |
| `run_rome_easyedit.py` | `rome_easyedit_gpt2.json` | GPT-2, EasyEdit, GPU | Reference EasyEdit ROME: 1.0 / 0.972 / 0.771 (external clone, not bundled) |
| `run_headgeom_gpt2xl.py` | `headgeom_gpt2xl.json` | GPT-2-XL, GPU | Head-geometry census: 99.82% of tokens are their own nearest head neighbor |

### Certificate-predicts-editors on CounterFact (exploratory)

| Script | Output (`results/`) | Requirements | Finding |
| --- | --- | --- | --- |
| `fetch_counterfact.py` | `data/counterfact.json` (local) | CounterFact | Downloads the CounterFact records the trio below consume |
| `run_cert_counterfact_gpt2xl.py` | `cert_counterfact_gpt2xl*.json*` | GPT-2-XL, CounterFact, GPU | Phase A: certificate features per CounterFact record |
| `run_editors_counterfact.py` | `editors_counterfact_gpt2xl.jsonl` | GPT-2-XL, EasyEdit, CounterFact, GPU | Phase B: run FT/ROME/GRACE editors on the same records |
| `analyze_cert_vs_editors.py` | `cert_vs_editors_analysis.json` | CPU | Phase C: pre-edit difficulty predicts editor paraphrase/neighborhood outcomes (exploratory) |

### Producers (externally-sourced artifacts, copied verbatim)

These were copied from the private working repo; see [`scripts/README.md`](scripts/README.md)
for provenance and the hardcoded-path caveat on `run_fineweb_kb_inject.py`.

| Script | Output (`results/`) | Requirements | Finding |
| --- | --- | --- | --- |
| `run_hlm5_capacity_search.py` | `hlm5_capacity_search_report.json`, `capacity_search/` | CPU/GPU | Slot-memory capacity search |
| `run_train_key_value.py` | `capacity_search/random_p*.json` | CPU/GPU | Capacity-search trainer invoked by the search above |
| `run_hlm5_scale_suite.py` | `hlm5_scale_suite_report.json` | CPU/GPU | Scale suite of the key-value memory |
| `run_lm_kb_inject.py` | `lm_kb_inject*.json` | CPU/GPU | HLM5 as an editable, auditable knowledge base in a trained LM |
| `run_incontext_mqar.py` | `incontext_mqar.json` | CPU/GPU | In-context associative recall (MQAR) |
| `run_edit_receipt_demo.py` | `edit_receipts.json` | CPU/GPU | Replay-verifiable edit receipts |
| `run_fineweb_kb_inject.py` | `g2c_rare.json` (+ G2/G3 gate JSONs) | Leonardo, GPU | FineWeb-trunk fact injection; source of the 136M/1B gate JSONs |
| `run_hybrid_energy_ops.py` | `hybrid_energy_ops.json` | hybrid, GPU | Energy-language operators on the co-trained memory |

### Reproduce and figures

| Script | Output | Requirements | Finding |
| --- | --- | --- | --- |
| `verify_artifacts.py` | (assertions) | CPU | Loads every cited JSON and asserts the paper's load-bearing numbers |
| `make_figures.py` | `paper/figures/*.pdf` | CPU | Regenerates the data figures from `results/` |
| `make_schematics.py` | `paper/figures/fig-arch-forward.pdf`, `fig1-pipeline-a-b.pdf` | CPU | Regenerates the two hand-made schematic figures |

## Determinism

Every script seeds `torch` (and `random`) to **0** in its `main()`. The
certificate machinery — envelope, β★, head geometry, gate ROC, own-slot weight,
parameter counts — is **closed-form deterministic** given the frozen trunk:
`run_1b_certificate.py`, `run_1b_envelope_multikey.py`, `run_1b_betastar.py`,
`run_1b_headgeom_metrics.py`, `run_1b_gate_roc.py`, `run_ownslot_weight.py`,
`run_notax_params.py`, and `verify_artifacts.py` recompute bit-for-bit. The
efficacy/generalization/locality numbers are **single-seed** point estimates; the
fact-level uncertainty is reported as **95% bootstrap CIs inside the artifacts**
(e.g. the `[low, high]` triples in `hlm5_1b_faithful_certdosed.json` and
`gpt2_table1_v2.json`), not as multi-seed spread.

## Honest status (read before citing)

Adapted from [`results/RESULTS.md`](results/RESULTS.md); no inflation.

- **Verified-real:** the certificate (envelope, β★, head reachability — it
  correctly predicts the rare failures), residual-synthesis rescue (40/40, small
  β), the perfectly-selective gate, exact-key spread injection (flip 1.000,
  locality 1.000, zero gradients), the GPT-2 controlled study, and reference ROME.
- **Narrow:** deployed efficacy on arbitrary 1B targets sits at the reachability
  frontier (certified 0.765, global-boost ablation 0.529, +synthesis 1.000);
  paraphrase transfer is ~0 (exact-key by design); the naive 1B additive read ties
  a gate-matched logit bias — the GPT-2 generalization advantage does *not*
  replicate on the 1B trunk.
- **Exploratory:** the certificate-predicts-editors correlations on CounterFact
  (paraphrase: required strength L; neighborhood: certified-dose margin —
  ROME/FT/GRACE) — signed and significant but exploratory.
- **Planned, not measured:** the robust certificate's drift calibration (ε_h,
  ε_r), 3B scale, and the full public-benchmark sweep (CounterFact/zsRE/MQuAKE ×
  MEMIT/MEND/SERAC). The paper's Limitations section states each as a named
  falsification experiment (E1–E8). Full head-to-head evaluation against MEMIT,
  MEND, and SERAC on standard benchmarks remains open.

## Reproduce the paper

```bash
python scripts/verify_artifacts.py     # assert every load-bearing number
python scripts/make_figures.py         # data figures  -> paper/figures/
python scripts/make_schematics.py      # schematic figures -> paper/figures/
tectonic paper/hlm5-paper.tex          # compile the PDF
```

`verify_artifacts.py` reads only JSONs already in `results/` and needs no
checkpoint. The on-trunk scripts need the 1B checkpoint in `models/`.

## Citing

```bibtex
@article{dima2026certified,
  title   = {Certified Knowledge Editing: Closed-Form Reachability Guarantees
             for Additive Edits to a Frozen Language Model},
  author  = {Dima, Marius},
  year    = {2026},
  doi     = {10.5281/zenodo.21258599},
  note    = {arXiv link forthcoming}
}
```

See also [`CITATION.cff`](CITATION.cff). The arXiv identifier will be filled in on
submission (see [`RELEASING.md`](RELEASING.md)).

## License

Business Source License 1.1 (BSL 1.1), © 2026 Qriton Technologies S.R.L. Use for
academic research, education, personal projects, internal evaluation, and
publishing research results is expressly permitted; commercial use is not.
The license converts to Apache-2.0 on **2030-01-01**. See [`LICENSE`](LICENSE);
for commercial use contact license@qriton.com.

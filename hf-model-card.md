---
license: other
license_name: bsl-1.1
license_link: LICENSE
language:
  - en
tags:
  - knowledge-editing
  - model-editing
  - certified-editing
  - hopfield
---

# HLM5 — 1B frozen trunk for certified knowledge editing

HLM5 is a frozen decoder-only language model paired with a certified, auditable,
exact-key edit layer. It is the trunk for **"Certified Knowledge Editing:
Closed-Form Reachability Guarantees for Additive Edits to a Frozen Language
Model"** (Marius Dima, Qriton). A slot memory attaches to this frozen trunk; a
hidden-state query selects slots through a hard cosine gate and adds the value to
the pre-head representation. Because the gate and read see only the unperturbed
hidden state, the post-edit logits are affine in the edit strength, so
single-token reachability is one-dimensional linear feasibility — solved in closed
form as a certificate that **accepts or refuses each edit before injection** and
doses the accepted ones at a certified strength.

- **Code, scripts, and result artifacts:** https://github.com/qriton/hlm5
- **Paper:** in the repository (`paper/`); arXiv link forthcoming.

## Model description

- **Architecture:** decoder-only transformer, dim 1792, 24 layers, 14 heads,
  context length 1024.
- **Vocabulary:** 65,536-token byte-pair tokenizer (`fineweb-65536-compat.json`).
- **Training data:** FineWeb (~20B tokens).
- **Validation PPL:** 18.01 (baseline trunk).
- **Parameters:** ~1.045B.

This repository hosts `hlm5_lm_baseline_fineweb_g3_final.pt` (the frozen baseline
trunk, step 150,999) and the tokenizer. The matched co-trained-memory "hybrid"
checkpoint is published alongside it once the release runbook's upload step is done.

## Results

Evaluated as the deployed certificate-governed pipeline (whitened degree-5 gate,
threshold 0.95) on a 17-fact set; every number traces to a JSON in the GitHub
repository under `results/`.

- **Certificate-dosed efficacy 0.765** (13/17), rising to **1.000** (17/17) with
  residual-synthesis rescue through the full memory path, at **locality 1.000** (bit-identical logits on
  neutral prompts) and paraphrase transfer 0.020 (exact-key by design). The
  **0.529** figure some earlier material led with is the *global-boost dosing
  ablation* — a single strength that overshoots the certified interval — not the
  method.
- **Operating envelope:** 78.4% of 1,200 single-token targets reachable at a
  novel-entity key; 79.4% ± 2.2% across 60 keys.
- **Gate:** perfectly separable (exact-key 1.0, all paraphrase/typo/other-relation
  0.0); realized own-slot weight 0.9993.
- **No-tax:** matched baseline-vs-hybrid validation-PPL delta 0.116% at 1B
  (0.018% at 136M).

## Intended use

Research on certified, auditable knowledge editing: reachability certificates,
certificate-governed dosing, exact-key slot memories, and edit/forget/rollback
receipts on a frozen trunk. Suitable for reproducing the paper and building on the
certificate machinery.

## Out of scope

This is a **research base model, not an instruction-tuned or chat model**. It is
not aligned, not safety-tuned, and not intended for deployment as an assistant or
for generating end-user-facing text. The edit layer is exact-key by design: it
does **not** generalize edits to paraphrases.

## Citation

```bibtex
@article{dima2026certified,
  title   = {Certified Knowledge Editing: Closed-Form Reachability Guarantees
             for Additive Edits to a Frozen Language Model},
  author  = {Dima, Marius},
  year    = {2026},
  note    = {arXiv link forthcoming}
}
```

## License

Business Source License 1.1 (BSL 1.1), © 2026 Qriton Technologies S.R.L.
Academic research, education, personal, and non-commercial evaluation use is
permitted; commercial use is not. Converts to Apache-2.0 on 2030-01-01. See the
`LICENSE` file in the GitHub repository; commercial inquiries: license@qriton.com.

# Releasing HLM5

Maintainer runbook for the public release. Run the steps in order; each is a gate
for the next.

## 1. Push the code to GitHub

Create an **empty** repository `qriton/hlm5` (no README, no license, no
`.gitignore` — this repo already carries all three), then:

```bash
git remote add origin https://github.com/qriton/hlm5.git
git push -u origin main
```

The 1B checkpoints are gitignored and must **not** be pushed to GitHub; they live
on Hugging Face (steps 3–4).

## 2. Verify the paper's self-citation URLs resolve

The paper cites two Qriton project URLs. Confirm both resolve **before** arXiv
submission — a dead URL in the references is a preventable reviewer flag:

- `https://github.com/qriton/energy-lang` — `qriton2026energylang` bibitem.
- `https://hlm.qriton.com` — `qriton2026hlm` bibitem.

If either is dead, fix the bibitem in `paper/hlm5-paper.tex`, regenerate the
two-column variant and PDFs with `python paper/build.py`, and commit before
pushing.

## 3. Upload the hybrid checkpoint — DONE (2026-07-08)

Both trunks are live at https://huggingface.co/qriton/hlm5-1b-trunk (baseline
and hybrid at the repository root, so `--local-dir models` places them exactly
where `hlm5.io` looks). `models/README.md` carries both download commands.

## 4. Replace the Hugging Face model card

The live card currently leads with the superseded 0.529 number. Replace its
contents with `hf-model-card.md` from this repo (it leads with the certificate-dosed
0.765/1.000 result and frames 0.529 as the dosing ablation). The YAML frontmatter
is already in HF's expected form (`license: other`, `license_name: bsl-1.1`,
`license_link: LICENSE`).

While in the HF repo, also: (a) upload this repo's `LICENSE` file (the card's
`license_link` points at it, and the HF repo has none today); (b) delete or
update the stale `hlm5_lm.py` / `hlm5_memory.py` copies there — the canonical
model code is now the `hlm5` package in this repository.

## 5. Submit to arXiv

Build the arXiv package and submit it:

```bash
python paper/build.py          # produces paper/arxiv/hlm5-paper-arxiv.zip
```

The zip contains exactly one main `.tex` plus the referenced figure PDFs (arXiv
AutoTeX requirement). Submit with categories **cs.CL** (primary) + **cs.LG**,
under the arXiv non-exclusive license. `build.py` needs `tectonic` on PATH
(https://tectonic-typesetting.github.io) and, for the docx, `pip install
pypandoc-binary` (or pass `--skip-docx`).

## 6. Backfill the arXiv identifier

Once arXiv assigns an ID, grep for the placeholder and fill it in everywhere:

```bash
grep -rn "forthcoming" README.md hf-model-card.md CITATION.cff
```

Update the BibTeX `note` in `README.md` and `hf-model-card.md`, the
`preferred-citation` notes field in `CITATION.cff`, and the live HF card.

## 7. Tag the release

```bash
git tag -a v1.0.0 -m "v1.0.0 — first public release

Certified Knowledge Editing: closed-form reachability guarantees for
additive edits to a frozen language model (paper + code + artifacts).

- hlm5 package (model, memory, certify, edit_audit, key_value, io)
- every experiment script and every result JSON the paper cites;
  scripts/verify_artifacts.py re-asserts all load-bearing numbers on CPU
- paper sources + built PDFs (1col/2col) + build pipeline (paper/build.py)
- 1B frozen trunk on Hugging Face (qriton/hlm5-1b-trunk); tokenizer in-repo
- 17 CPU tests, CI (ruff + pytest), BSL-1.1 (Apache-2.0 on 2030-01-01)"
git push origin v1.0.0
```

Create the GitHub `v1.0.0` release from the tag.

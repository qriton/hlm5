# Certificate predicts editors: confirmatory study (2026-10-02)

Registration: `docs/certificate-predicts-editors-confirmatory-2026-10-01.md` (hashes in `REGISTRATION.sha256`;
`integrity.py` confirms the registered scripts are byte-identical after the run). gpt2-xl, CounterFact case_ids
5000-5599 (600 records never used before), single edit from a fresh model, weights restored exactly after
every edit (max deviation 0.0). 600 of 600 records for BASE, ROME, MEMIT and FT; no errors in the final results
(see `DEVIATIONS.md` for the out-of-memory interruption and resume).

## Registered tests (two-sided Spearman, Holm-Bonferroni over the five, family alpha 0.05)

| test | predictor | outcome | rho [95% bootstrap] | p | Holm p | verdict |
|---|---|---|---|---|---|---|
| C1 | L | ROME paraphrase success | -0.142 [-0.217, -0.061] | 4.7e-4 | 0.0014 | confirmed |
| C2 | L | MEMIT paraphrase success | -0.181 [-0.259, -0.102] | 7.8e-6 | 3.3e-5 | confirmed |
| C3 | margin_after_heur | FT paraphrase success | -0.183 [-0.251, -0.109] | 6.6e-6 | 3.3e-5 | confirmed |
| C4 | L | ROME neighbour log-odds lost vs BASE | +0.119 [0.038, 0.199] | 0.0035 | 0.007 | confirmed |
| C5 | L | MEMIT neighbour log-odds lost vs BASE | +0.115 [0.035, 0.193] | 0.005 | 0.007 | confirmed |

All five confirm, with the registered signs. L is the certificate's required strength: the minimum dose of the
unit target direction that makes the new answer win at the last position, computed from the frozen head and the
hidden state alone, before any edit.

## How large (registered diagnostics, and a descriptive quartile split not in the registration)

- The effects are small: |rho| between 0.11 and 0.18.
- Registered practical-value diagnostic: AUC of L for flagging records with zero paraphrase success is 0.586
  (ROME, 68 of 600) and 0.594 (MEMIT, 275 of 600). Above chance, not a usable screen by itself.
- Descriptive, easiest quartile of L (<= 3.47, n = 151) against hardest (>= 5.83, n = 150):

| editor | paraphrase success, easy vs hard | neighbour log-odds lost (nats), easy vs hard | neighbours whose argmax flipped, easy vs hard |
|---|---|---|---|
| ROME | 0.801 vs 0.680 | 1.06 vs 1.56 | 0.148 vs 0.149 |
| MEMIT | 0.530 vs 0.347 | 0.06 vs 0.17 | 0.008 vs 0.008 |
| FT | 0.169 vs 0.070 | 0.49 vs 0.86 | 0.072 vs 0.077 |

- The damage result is on the registered measure, the log-odds margin of neighbouring facts. The rate at which
  neighbours actually flip does not differ between easy and hard edits.
- Registered confound diagnostic: Spearman(L, BASE neighbourhood score) = +0.183 (p = 6.7e-6): the unedited
  model already knows the neighbours of high-L records better, which is why damage is measured against BASE.

## Aggregates on this window

| editor | efficacy | paraphrase | neighbourhood |
|---|---|---|---|
| BASE (no edit) | 0.003 | 0.006 | 0.788 |
| ROME | 0.998 | 0.747 | 0.641 |
| MEMIT | 0.772 | 0.403 | 0.782 |
| FT | 0.978 | 0.096 | 0.720 |

## What may now be claimed, and what may not

- May: on gpt2-xl and CounterFact, the certificate's required strength L, computed before editing, predicts
  where ROME, MEMIT and FT edits generalise to paraphrases less and shift neighbouring facts' margins more.
  Confirmed on untouched records, corrected for five tests.
- May not: that it predicts rewrite failure (H1 of the exploratory study failed and was not retested); that it
  predicts neighbours flipping; that it is a practical pre-edit screen (AUC 0.59); anything about other models.
- Withdrawn: the earlier "neighbourhood damage ROME/GRACE p < 0.01" line (the confound above).

## Layout, and how to re-check

Paths inside this folder mirror the working tree the study ran in, so the registration's own references resolve.

| path | what |
|---|---|
| `docs/certificate-predicts-editors-confirmatory-2026-10-01.md` | the registration, written before any confirmatory record was computed |
| `REGISTRATION.sha256` | SHA-256 of the registration and the three scripts, taken at registration |
| `baselines/confirm/cert_window.py`, `editors_window.py` | the producers: copies of `scripts/run_cert_counterfact_gpt2xl.py` and `scripts/run_editors_counterfact.py` changed only in record window and output path. They ran on the authors' machine (local EasyEdit clone, absolute paths) and are kept verbatim as the record |
| `baselines/confirm/analyze_confirm.py` | the fixed analysis; needs only SciPy and the two result files |
| `baselines/confirm/results/` | per-record certificate features and editor outcomes, the analysis output, and the audit of the 27 out-of-memory rows |
| `baselines/confirm/logs/` | the run logs, including every out-of-memory event |
| `DEVIATIONS.md` | what departed from the registered procedure, and why it does not touch the tests |
| `integrity.py` | record counts, errors, restoration, and that the registered files still match their hashes |
| `split_oom.py`, `rebuild_oom_audit.py` | the helpers named in `DEVIATIONS.md`, verbatim |

```bash
python research/certificate_confirmatory/integrity.py
python research/certificate_confirmatory/baselines/confirm/analyze_confirm.py   # rewrites results/confirm_analysis.json
```

Re-running the analysis on the shipped records reproduces every correlation, interval and verdict exactly;
p-values agree to floating-point rounding across SciPy versions (checked with 1.14 and 1.18).

The exploratory study this follows (records 0-299; `results/cert_vs_editors_analysis.json`) had its own plan, an
internal document not in this repository; its criteria and outcome are summarised at the top of the registration.

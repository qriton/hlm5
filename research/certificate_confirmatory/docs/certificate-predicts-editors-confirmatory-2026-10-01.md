# Certificate predicts editors: confirmatory registration (2026-10-01)

Registered before any confirmatory record is computed. Follows the exploratory study
`docs/certificate-predicts-editability-plan-2026-07-02.md` (Phase A/B on CounterFact
records 0-999 / 0-299, gpt2-xl). Nothing in this file may change after the first
confirmatory record is written; the commit hash of this file is the registration.

## What the exploratory study found, re-read on 2026-10-01

- **H1 (primary) failed.** Best AUC for rewrite failure 0.629 (FT, margin_after_heur) and
  0.609 (ROME, L), below the registered 0.65. This stands; it is not retested.
- **Neighbourhood "damage" was confounded.** The analysis correlated certificate features
  with the edited model's raw neighbourhood score, which mostly reflects how well the
  *unedited* model knows the neighbours. Re-analysis of the same 300 records:
  - GRACE changes no neighbour (mean damage 0.000); its Spearman(L, neighbourhood) is
    +0.226, identical to BASE's +0.226. The published "GRACE p < 0.01" is the confound.
  - ROME damage measured against BASE: Spearman(L, Δscore) = +0.065 (p = 0.26);
    Spearman(L, Δlog-odds) = +0.171 (p = 0.003). MEMIT Δlog-odds: +0.252 (p = 1e-5).
- **Paraphrase generalisation held** (exploratory): Spearman(L, paraphrase) ROME -0.162
  (p = 0.005), MEMIT -0.280 (p = 8e-7, computed 2026-10-01, not previously reported);
  Spearman(margin_after_heur, paraphrase) FT -0.155 (p = 0.007).

## Window (untouched)

CounterFact (`baselines/data/counterfact.json`, the same file), records with
`case_id >= 5000`, in file order, keeping the first **600** whose `target_new` and
`target_true` first BPE tokens (with leading space) are well-defined (the exploratory
filter). A scan of every results/report JSON on this machine on 2026-10-01 found no file
touching case_ids 3000-21918.

## Procedure (identical to the exploratory run except the window)

- Certificate: `baselines/confirm/cert_window.py`, a copy of
  `run_cert_counterfact_gpt2xl.py` changed only in window, N = 600 and output paths.
- Editors: `baselines/confirm/editors_window.py`, a copy of
  `run_editors_counterfact.py` changed only in window and output path; same EasyEdit
  clone, hparams, single-edit-from-fresh-model protocol and restoration check.
  Editors: **BASE, ROME, MEMIT, FT**. GRACE is excluded: its paraphrase rate (0.005) and
  damage (0.000) have no variance to predict.
- Analysis: `baselines/confirm/analyze_confirm.py`, fixed below.

## Confirmatory tests (two-sided Spearman; Holm-Bonferroni over C1-C5, family alpha 0.05)

| test | predictor | outcome | expected sign |
|---|---|---|---|
| C1 | L | ROME paraphrase_argmax | negative |
| C2 | L | MEMIT paraphrase_argmax | negative |
| C3 | margin_after_heur | FT paraphrase_argmax | negative |
| C4 | L | ROME damage = BASE.neighborhood_logp_diff_mean - ROME.neighborhood_logp_diff_mean | positive |
| C5 | L | MEMIT damage, same definition | positive |

A test **confirms** when its Holm-adjusted p < 0.05 **and** its sign is the expected one.
Records with an editor error are dropped for that editor only, and the count is reported.

## Reported regardless of outcome

- Diagnostic: Spearman(L, BASE.neighborhood_score), the confound.
- Practical value: AUC of L for flagging the records where ROME's or MEMIT's paraphrase
  success is zero; effect sizes with 95% bootstrap intervals (2,000 resamples).

## What changes depending on the outcome

- Confirmed tests replace the README's "exploratory" line, with confirmatory numbers.
- Tests that fail are removed from the README as claims.
- The GRACE neighbourhood claim is withdrawn either way (shown above to be the confound).
- No threshold, window, editor or feature is changed after the first record is written.

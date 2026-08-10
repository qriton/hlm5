# SCA-1 Preregistration — Source-Calibrated HLM5 Admission Transfer

**Date:** 2026-08-10

**Status at registration:** design and TOPv2 population fixed before any TOPv2
utterance is tokenized by, encoded by, or scored with HLM5. Dataset text has
been parsed only for deterministic normalization, duplicate/overlap removal,
intent extraction, hashing, and population census.

**Scope:** HLM5 model research only. No KB, Flow, Ingesto, training, factual
editing, or product integration.

## 1. Question

NKT-1 showed that blind ZCA supplied most of the development gain in frozen
HLM5 natural-key routing, but its fixed production-style threshold admitted
only 1 of 494 target rows. Source-label Fisher did not clear the matched blind
or shuffled-label attribution bars.

SCA-1 asks one narrower question:

> Can blind covariance conditioning, with an admission threshold calibrated
> only on source-domain episodes, transfer to low-support routing in untouched
> target domains while preserving useful coverage and controlling false routes?

The candidate is conventional blind ZCA. No label-aware covariance, recurrent
settle, energy optimization, or target-domain metric fit is present.

## 2. Untouched population and provenance

Dataset: TOPv2 1.1, released with Chen et al., *Low-Resource Domain Adaptation
for Compositional Task-Oriented Semantic Parsing* (EMNLP 2020).

- official archive:
  `https://dl.fbaipublicfiles.com/topv2/TOPv2_Dataset.zip`
- archive SHA-256:
  `e73a13eb32f67f69c630abde549fa153f288564e2703b71492f04cb59b2cae13`
- license: CC BY-SA 4.0
- official paper:
  `https://aclanthology.org/2020.emnlp-main.413/`
- official target-domain convention: `reminder` and `weather`, the two domains
  for which the release supplies low-resource splits
- source domains: `alarm`, `event`, `messaging`, `music`, `navigation`, `timer`

The checked-in manifest pins every official TSV, README, and license hash. Its
file SHA-256 before HLM5 outcome access is
`e2575806939206cca4c2ffe506d00f9894d3c9b2c70463082a1a5a7dae243159`;
its canonical scientific SHA-256 is
`90293546a57991bf046df32a87e20798b8da089bbe5c0fe88d07b98fcea8682c`.

An outcome-blind tracked-estate search found no prior SLURP, TOPv2, MTOP,
HINT3, or HWU64 experiment in HLM2, HLM5, or HLM-Demos. SLURP was rejected as
the follow-up population because MASSIVE is explicitly its multilingual
localization. “Untouched” here means no prior HLM experiment or outcome access;
it does not prove that public TOPv2 text was absent from FineWeb pretraining.

The official test TSVs may be downloaded, parsed, deduplicated, and hashed for
census. Their utterances must not be tokenized, encoded, or scored before a
passing development verdict and a separate immutable test registration.

## 3. Deterministic cleanup

For every utterance, normalize with Unicode NFKC, case folding, trim, and
whitespace collapse. Extract the root intent from the leading `IN:` node and
name a class `domain::ROOT_INTENT`.

Within each partition, group globally across all eight domains by normalized
utterance:

1. if a group has conflicting class labels, drop the whole group;
2. otherwise keep the lexicographically smallest registered row ID;
3. exclude evaluation rows whose normalized text occurs anywhere in raw train;
4. exclude test rows whose normalized text occurs anywhere in raw train or raw
   evaluation.

The fixed cleanup census is:

| Partition | Raw | Clean | Conflicting rows dropped | Duplicate rows dropped | Earlier-split overlaps dropped |
|---|---:|---:|---:|---:|---:|
| train | 124,597 | 106,234 | 327 | 18,036 | 0 |
| evaluation | 17,160 | 13,578 | 2 | 360 | 3,220 |
| test | 38,785 | 30,176 | 15 | 1,028 | 7,566 |

No post-outcome substitution is permitted.

## 4. Fixed source and target populations

An intent is eligible when cleaned train contains at least three rows and
cleaned evaluation contains at least 20 rows.

This yields exactly nine eligible target intents:

1. `reminder::CREATE_REMINDER`
2. `reminder::DELETE_REMINDER`
3. `reminder::GET_REMINDER`
4. `reminder::GET_REMINDER_DATE_TIME`
5. `reminder::UPDATE_REMINDER`
6. `reminder::UPDATE_REMINDER_DATE_TIME`
7. `reminder::UPDATE_REMINDER_TODO`
8. `weather::GET_WEATHER`
9. `weather::UNSUPPORTED_WEATHER`

For each eligible target intent, select three train supports and 20 evaluation
queries by the registered salted SHA-256 order. The target address bank has 27
separate slots; the development population is class-balanced at 180 rows.

Blind covariance fitting uses exactly 2,000 cleaned source-train rows per
source domain, chosen by a separate registered salted SHA-256 order: 12,000
rows total. It receives no reminder or weather row and no evaluation or test
row.

Thirty-seven source intents are eligible. Sort them by the registered salted
intent hash and assign three disjoint groups of nine:

- calibration-supported: three train supports plus 20 evaluation positives
  per intent (27 slots, 180 positives);
- calibration-negative: 20 evaluation rows per intent (180 negatives), used
  only to set thresholds; and
- off-support audit: 20 evaluation rows per intent (180 negatives), never used
  to set thresholds.

Ten remaining eligible source intents are unused. Exact intent memberships,
support descriptors, population counts, and population hashes are fixed in
`data/topv2_development_manifest.json`.

The prospective test population is 20 cleaned test rows for each of the same
nine target intents (180 rows), fixed by a separate salt. It remains unencoded.

## 5. Frozen HLM5 representation

- checkpoint:
  `D:/HLM2/artifacts/models/demo/hlm5-136m-fineweb-g2-2026-06-10/model.pt`
- checkpoint SHA-256:
  `2addfa88808be6847e29f70c9182093c15ff420c168f9dd8b41b72c4757a8eda`
- tokenizer SHA-256:
  `15993635191a1c5f1a5dc7aeaacbdf9a44a45d90abef954fc77b686f4fbbe588`
- architecture: HLM5LM, 12 layers, width 768, 12 heads, context 1,024,
  vocabulary 65,536, `memory_layer=False`
- representation: final-layer hidden state at the final non-special utterance
  token, using the already published NKT-1 tokenizer and extraction code
- hidden extraction: CUDA float32, evaluation mode, deterministic algorithms,
  TF32 disabled, `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- scientific geometry: CPU float64

The checkpoint loads strictly. No parameter, prompt, pooling rule, layer,
tokenizer, or representation is trainable or selectable.

## 6. Frozen key operators

Let the 12,000 source-fit hiddens be rows of `H`, their mean be `mu`, and

`C = (H - mu)^T (H - mu) / 12000`.

With fixed shrinkage `alpha = 0.05`, define

`C_alpha = (1 - alpha) C + alpha trace(C)/768 I`.

Every transformed row is L2-normalized. The five arms are:

1. `raw`: `h`;
2. `centered`: `h - mu`;
3. `diagonal_std`: `(h - mu) diag(C_alpha)^(-1/2)`;
4. `random_basis_zca`: `(h - mu) Q diag(lambda)^(-1/2) Q^T`; and
5. `blind_zca`: `(h - mu) C_alpha^(-1/2)`.

Here `lambda` are the ascending eigenvalues of `C_alpha`. For the random-basis
control, draw a 768 by 768 CPU-float64 standard-normal matrix with PyTorch seed
`20260811`, take a complete QR decomposition, and flip each `Q` column so the
corresponding diagonal of `R` is nonnegative. Thus random-basis ZCA has the
same singular values as blind ZCA but unrelated directions. Registration pins
the PyTorch version and resulting tensor hashes.

Blind ZCA uses no intent label. `diagonal_std` tests whether per-coordinate
scale is enough; `random_basis_zca` tests whether the measured covariance
directions, rather than the transform spectrum alone, are load-bearing.

## 7. Exact HLM5 reader

For normalized query `q` and each normalized support key `k_r`, compute

`score_r(q) = max(q dot k_r, 0)^5`.

Choose the maximum-score slot, breaking ties by lowest slot index. The
predicted class is that slot's intent. Preserve all three supports per intent;
do not average prototypes.

The implementation must match:

1. an independent NumPy scorer;
2. positive-cosine top-slot routing; and
3. the checked-in float32 `EditableHLM5Memory.score()` implementation for
   predictions and abstentions.

Degree five is top-slot-equivalent to positive cosine. This reader is a frozen
cache-equivalent address rule, not a recurrent attractor.

## 8. Source-only calibration

For each arm separately:

1. transform its 27 calibration supports, 180 calibration positives, and 180
   calibration negatives with the single source-fitted operator;
2. route every calibration-negative row against the calibration support bank;
3. set `tau_arm` to the NumPy `0.95` quantile of negative maximum scores with
   `method="higher"`; and
4. admit a row iff `maximum_score > tau_arm` (strict inequality).

This construction admits at most 5% of the registered calibration negatives,
including under score ties. Calibration-positive outcomes are reported but do
not select or change the threshold.

Apply the unchanged `tau_arm` to the target bank and to the disjoint
off-support-audit rows. There is no target-domain threshold, covariance, label,
or outcome fit.

## 9. Metrics and estimators

For each arm report:

- target macro top-1 accuracy over nine intents;
- target macro coverage (mean per-intent admitted fraction);
- target macro correct-admission rate (mean per-intent fraction both correct
  and admitted), the primary metric;
- aggregate selective accuracy among admitted target rows;
- maximum predicted-intent share over all target rows;
- calibration-negative empirical false-admission rate;
- calibration-positive coverage, accuracy, and correct-admission rate;
- disjoint off-support-audit false-admission rate;
- score quantiles, true-versus-best-false cosine margins, and support-Gram
  geometry; and
- row-level prediction, score, threshold, admission, correctness, and hashes.

For candidate-minus-control comparisons, compute the arithmetic mean of the
nine paired per-intent correct-admission differences. Bootstrap the nine intent
indices with replacement for 10,000 resamples using NumPy seed `20260811` and
report the linear-interpolation 2.5th and 97.5th percentiles. Use the same
estimator for the observed point and every resample. These intervals describe
stability across the nine fixed development intents; they are not an
independent confidence guarantee.

## 10. Validity gates

All are binding:

1. dataset, license, checkpoint, tokenizer, Git, implementation, dependency,
   test, preregistration, manifest, preflight, command, and registration hashes
   match;
2. all cleanup counts, intent groups, population hashes, 12,000 source-fit
   rows, 27 supports per bank, and 180 rows per evaluation population match;
3. source fitting contains no target-domain, evaluation, or test row;
4. calibration-supported, calibration-negative, and off-support-audit intent
   groups are pairwise disjoint;
5. no test utterance is tokenized, encoded, or scored;
6. the model loads strictly in evaluation mode and every hidden and scientific
   value is finite;
7. inverse-root identity error is at most `1e-10`;
8. random-basis and blind-ZCA singular values match within `1e-10` relative
   and absolute error, while their basis hashes differ;
9. calibration uses only negative maximum scores, NumPy quantile `higher`, and
   strict `>` admission; its empirical negative rate is at most 0.05;
10. degree-five, positive-cosine, independent NumPy, and checked-in float32
    reader predictions/abstentions agree; and
11. a hidden-cache replay reproduces the scientific object and rows exactly.

Any validity failure yields `HARNESS_INVALID`, which takes precedence over
interruption or scientific bars.

## 11. Development verdict

`PASS_TO_TEST` requires every validity gate and every bar:

1. blind-ZCA target macro correct-admission rate `>= 0.20`;
2. blind ZCA minus raw correct-admission macro `>= +0.030`;
3. its paired-intent interval lower endpoint `> 0`;
4. blind ZCA minus diagonal standardization `>= +0.015`;
5. its paired-intent interval lower endpoint `> 0`;
6. blind ZCA minus random-basis ZCA `>= +0.015`;
7. its paired-intent interval lower endpoint `> 0`;
8. blind-ZCA target macro coverage `>= 0.30`;
9. blind-ZCA aggregate selective accuracy `>= 0.65`;
10. blind-ZCA off-support-audit false-admission rate `<= 0.05`;
11. blind-ZCA maximum predicted-intent share `<= 0.25`; and
12. blind-ZCA target macro top-1 accuracy is no more than `0.010` below raw.

Otherwise the verdict is `STOP_BEFORE_TEST`. No aggregate improvement can
override a failed binding bar.

## 12. Test boundary and stop rules

A passing development result permits one separate, immutable test
registration using the same target supports, source-fitted transforms,
calibration groups, thresholds, arm definitions, and bars on the fixed 180-row
test target population. Development cannot select a new threshold, shrinkage,
support count, source sample, target intent, operator, prompt, layer, pooling
rule, metric, or bar.

If development fails, preserve the artifacts and keep test unencoded. Do not
tune TOPv2. A future attempt needs a new mechanism and a new untouched
population, not a variation of this split.

## 13. Exact claim boundary

A pass would establish only that blind source-domain covariance conditioning
and source-only score calibration transfer to this fixed, balanced,
three-shot TOPv2 target-domain routing assay in frozen HLM5-136M hidden space,
and beat raw, diagonal-scale, and spectrum-matched random-basis controls.

It would not establish an HLM-specific retrieval advantage, recurrence,
attractor dynamics, semantic explanation, production routing, factual
editing, compliance, safety, or scaling. Because degree five is top-slot
cache-equivalent, the surviving component would remain a conventional static
key-conditioning primitive.

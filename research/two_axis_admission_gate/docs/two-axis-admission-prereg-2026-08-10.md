# SCA-2 Preregistration — Two-Axis HLM5 Admission Transfer

**Date:** 2026-08-10

**Status at registration:** design and HWU64 population fixed before any HWU64
utterance is tokenized by, encoded by, or scored with HLM5. Dataset text has
been parsed only for deterministic normalization, fold reconstruction,
duplicate/overlap removal, hashing, label census, and population assignment.

**Scope:** HLM5 model research only. No KB, Flow, Ingesto, training, factual
editing, or product integration.

## 1. Question

SCA-1 established that blind ZCA plus a source-only absolute-score threshold
transferred to untouched TOPv2 domains on 11 of 12 registered bars. It failed
the one remaining bar: target selective accuracy was `0.5714`, below `0.65`,
despite target macro correct admission `0.2222`, macro coverage `0.3889`, and
off-support false admission `0.0222`.

The failure isolates a missing variable. Maximum support score measures
proximity to the addressable set; it does not measure whether the winning
class is separated from the next class. SCA-2 asks:

> Does adding a source-calibrated winner-versus-runner-up class margin to the
> already-working blind-ZCA proximity gate repair selective accuracy on an
> untouched HLM5 population while retaining useful correct admissions?

No target threshold, target covariance, target label, target outcome, degree,
support count, prompt, pooling rule, or layer is fitted.

## 2. Untouched population and provenance

Dataset: the official NLU-Evaluation-Data release, commonly called HWU64,
released with Liu et al., *Benchmarking Natural Language Understanding
Services for Building Conversational Agents* (IWSDS 2019).

- official repository:
  `https://github.com/xliuhw/NLU-Evaluation-Data`
- pinned dataset commit:
  `f6071b496b17d71e6eb43f543af0707f4ff30557`
- official processing scripts:
  `https://github.com/xliuhw/NLU-Evaluation-Scripts`
- pinned scripts commit:
  `356711b59f347532d0290f070ff9aad5af7ed02e`
- paper: `https://arxiv.org/abs/1903.05566`
- license: CC BY 4.0

The authors' archived cross-validation release contains 64 benchmark intents.
The annotated source has four additional tiny auxiliary labels, but those are
absent from every official benchmark fold and are not eligible for SCA-2.

An outcome-blind tracked-and-archived-estate search found no prior HWU64 or
NLU-Evaluation-Data experiment in HLM2, HLM5, or HLM-Demos. “Untouched” means
no prior HLM experiment or outcome access; it does not prove that this public
2019 corpus was absent from FineWeb pretraining.

The tracked manifest pins `LICENSE`, `README.md`, and all ten official Rasa
test-fold JSON files. Its file SHA-256 is
`ae3af9eadb6511501a1c8695376b04c6176e3be8e465a68e07b0fd18477ae576`;
its canonical scientific SHA-256 is
`43ca391013300dd88a76c7e4a17b33503cb6682e295d5eac4071a50387590fd0`.

## 3. Fixed partition and cleanup

Use official fold 1 as development, official fold 2 as sealed test, and the
union of official folds 3 through 10 as training. This choice was fixed before
HLM5 outcome access.

Normalize text with Unicode NFKC, case folding, trimming, and whitespace
collapse. Within a partition, group globally by normalized text:

1. drop the whole group if labels conflict;
2. otherwise retain the lexicographically smallest registered row ID; and
3. remove from training every text present in raw development or raw test;
   remove from test every text present in raw development.

The fixed census is:

| Partition | Raw | Clean | Conflicting rows | Duplicate rows | Earlier/held-out overlaps |
|---|---:|---:|---:|---:|---:|
| training, folds 3–10 | 8,884 | 8,881 | 0 | 2 | 1 |
| development, fold 1 | 1,076 | 1,076 | 0 | 0 | 0 |
| sealed test, fold 2 | 1,076 | 1,076 | 0 | 0 | 0 |

The three duplicated utterances have the same label. No post-outcome row or
fold substitution is permitted.

## 4. Fixed intent populations

An intent is eligible when clean training has at least three rows and both
development and test have at least eight rows. Exactly 61 of 64 intents are
eligible. The three ineligible intents are `iot_hue_lighton`, `iot_wemo_on`,
and `music_settings`.

Sort eligible intents by salted SHA-256 using
`sca2-hwu64-intent-groups-v1`, then assign five consecutive groups of 12:

| Population | Registered intents |
|---|---|
| calibration-supported | `general_affirm`, `datetime_convert`, `general_joke`, `qa_definition`, `iot_wemo_off`, `general_commandstop`, `takeaway_order`, `music_likeness`, `transport_taxi`, `recommendation_locations`, `general_repeat`, `email_addcontact` |
| calibration-negative | `recommendation_movies`, `qa_stock`, `qa_factoid`, `recommendation_events`, `general_dontcare`, `transport_traffic`, `iot_hue_lightup`, `transport_query`, `play_music`, `alarm_remove`, `audio_volume_up`, `weather_query` |
| audit-supported | `general_quirky`, `calendar_set`, `iot_hue_lightchange`, `calendar_query`, `alarm_set`, `social_post`, `email_querycontact`, `play_audiobook`, `iot_hue_lightdim`, `general_explain`, `lists_createoradd`, `music_query` |
| audit-negative | `qa_currency`, `alarm_query`, `calendar_remove`, `iot_coffee`, `qa_maths`, `play_podcasts`, `general_negate`, `transport_ticket`, `play_game`, `audio_volume_down`, `email_sendemail`, `play_radio` |
| target-supported | `audio_volume_mute`, `iot_hue_lightoff`, `lists_remove`, `datetime_query`, `general_praise`, `social_query`, `lists_query`, `cooking_recipe`, `email_query`, `takeaway_query`, `news_query`, `general_confirm` |

`iot_cleaning` is the one unused eligible intent.

For each supported group, select three separate training supports per intent.
For every positive or negative evaluation population, select eight rows per
intent by the registered salted row order. Every support bank therefore has
36 slots and every evaluation population has 96 rows.

Blind covariance fitting receives all 7,047 clean training rows whose intent
is not in the target-supported group. It receives no target-intent,
development-fold, or test-fold row. Labels determine population exclusion but
are not inputs to the covariance objective.

The sealed test populations are 96 target-positive fold-2 rows and 96
audit-negative fold-2 rows. They may be parsed and hashed for census but must
not be tokenized, encoded, or scored before a passing development verdict and
a separate immutable test registration.

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
  token, using the already-published NKT-1 extraction code
- hidden extraction: CUDA float32, evaluation mode, deterministic algorithms,
  TF32 disabled, `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- scientific geometry: CPU float64

The checkpoint loads strictly. Nothing in the representation is selectable.

## 6. Frozen key operators

Fit the same five source-only operators as SCA-1 to the 7,047 source rows:

1. `raw`;
2. `centered`;
3. `diagonal_std`;
4. `random_basis_zca`; and
5. `blind_zca`, the candidate geometry.

Let source hiddens be rows of `H`, `mu` their mean, and

`C = (H - mu)^T (H - mu) / 7047`.

With `alpha = 0.05`, use

`C_alpha = 0.95 C + 0.05 trace(C)/768 I`.

Every transformed row is L2-normalized. Blind ZCA applies
`C_alpha^(-1/2)`. Diagonal standardization retains only its diagonal.
Random-basis ZCA uses the same eigenvalue spectrum with a CPU-float64 complete
QR basis from PyTorch seed `20260811`, with QR signs canonicalized by a
nonnegative diagonal of `R`. The registered implementation and preflight pin
the exact basis and operator hashes.

No intent label enters the transform calculation. These controls distinguish
aligned covariance directions from centering, coordinate scale, and transform
spectrum alone.

## 7. Exact reader and the two axes

For normalized query `q` and each normalized support key `k_r`, compute

`slot_score_r(q) = max(q dot k_r, 0)^5`.

Choose the maximum-score slot, breaking ties by lowest slot index. Preserve all
three supports per intent; do not average prototypes. The predicted class is
the winning slot's intent.

For each class `c`, define its class score as the maximum of its three slot
scores. Let `s1` and `s2` be the largest and second-largest class scores, with
class ties inherited from lowest support-slot index. The two admission axes are:

- absolute proximity: `a = s1`;
- relative class ambiguity: `m = s1 - s2`.

The candidate admits iff both `a > tau_abs` and `m > tau_margin`. Strict
inequalities are binding. A zero-score tie has margin zero and is rejected.

The implementation must match an independent NumPy scorer, positive-cosine
top-slot routing, and the checked-in float32
`EditableHLM5Memory.score()` predictions and absolute abstentions. Degree five
is top-slot-equivalent to positive cosine; this remains a cache-equivalent
static reader, not a recurrent attractor.

## 8. Source-only calibration

Calibrate each geometry arm separately against its 36-slot
calibration-supported bank.

Absolute threshold:

1. route the 96 calibration-negative rows;
2. set `tau_abs` to the NumPy `0.95` quantile of their maximum scores with
   `method="higher"`; and
3. use strict `a > tau_abs`.

This admits at most 5% of the registered calibration negatives, including
under ties.

Margin threshold:

1. route the 96 calibration-positive rows and identify incorrect top-1 rows;
2. require at least eight such errors, otherwise stop before source audit;
3. set `tau_margin` to the NumPy `0.90` quantile of the incorrect rows'
   class margins with `method="higher"`; and
4. use strict `m > tau_margin`.

Thus no more than 10% of the observed calibration errors survive the margin
axis. Correct calibration-positive rows do not select the threshold. Apply
both thresholds unchanged to source audit and, only if source audit passes, to
the target population. There is no threshold sweep.

## 9. Direct controls

For every geometry arm report:

- `absolute_only`: admit by `a > tau_abs`;
- `dual`: admit by both registered axes.

The direct causal comparison is blind-ZCA dual versus blind-ZCA
absolute-only: same hiddens, transform, support bank, scores, prediction, and
absolute threshold; only the class-margin condition differs.

Also construct a non-deployable, outcome-blind-at-ranking diagnostic on each
audit or target population. Let `N` be the number admitted by the candidate
dual gate. Among rows passing the candidate absolute gate, admit exactly the
`N` largest absolute scores, breaking ties by row ID. This
`matched_proximity_top_k` control has exactly the candidate's aggregate
admission count and uses no correctness label. It tests whether relative
margin selects correctness better than merely tightening proximity to the
same budget.

Raw, centered, diagonal, and spectrum-matched random-basis dual gates are
reported as geometry controls. They are not separately tuned and are not
primary SCA-2 promotion bars.

## 10. Metrics and estimators

For every supported population and gate report:

- macro top-1 accuracy over 12 intents;
- macro coverage;
- macro correct-admission rate;
- aggregate selective accuracy among admitted rows;
- admitted and correct-admitted counts;
- maximum predicted-intent share;
- per-intent versions of the above; and
- absolute-score, class-margin, and true-versus-best-false score summaries.

For negative populations report false-admission rate and counts. Preserve a
row record containing hashes, truth/prediction where defined, both scores,
both thresholds, admission, correctness, geometry arm, and gate.

Descriptive paired-intent bootstrap intervals use 10,000 resamples, NumPy seed
`20260812`, the same estimator for the point and each resample, and linear
2.5th/97.5th percentiles. They describe stability across the 12 fixed intents;
they are not independent 95% coverage guarantees and are not binding bars.

## 11. Source-audit gate before target encoding

Apply the calibrated thresholds unchanged to the disjoint audit-supported
bank, 96 audit positives, and 96 audit negatives. The target support and
development utterances must not be tokenized or encoded until all source-audit
bars pass.

For blind ZCA, all source-audit bars are binding:

1. dual macro coverage `>= 0.20`;
2. dual macro correct-admission rate `>= 0.15`;
3. dual selective accuracy `>= 0.65`;
4. dual minus absolute-only selective accuracy `>= +0.050`;
5. dual correct-admission divided by absolute-only correct-admission `>= 0.60`;
6. dual audit-negative false-admission rate `<= 0.05`.

Zero absolute-only correct admissions fails the retention bar. Failure yields
`STOP_BEFORE_TARGET`; target development and test remain unencoded.

## 12. Development verdict

If source audit passes, apply the unchanged source operator and thresholds to
the target bank, target development positives, and audit-negative off-support
rows. `PASS_TO_TEST` requires every validity gate and every target bar:

1. candidate dual macro coverage `>= 0.20`;
2. candidate dual macro correct-admission rate `>= 0.15`;
3. candidate dual selective accuracy `>= 0.65`;
4. candidate dual minus candidate absolute-only selective accuracy
   `>= +0.075`;
5. candidate dual minus matched-proximity selective accuracy `>= +0.050`;
6. candidate dual correct-admission divided by candidate absolute-only
   correct-admission `>= 0.60`;
7. target off-support false-admission rate `<= 0.05`;
8. maximum predicted-intent share `<= 0.25`; and
9. blind-ZCA macro top-1 accuracy is no more than `0.010` below raw.

Otherwise the verdict is `STOP_BEFORE_TEST`. Equal-threshold or aggregate
improvement cannot override a failed bar.

## 13. Validity gates

All are binding:

1. dataset, license, checkpoint, tokenizer, Git, implementation, dependency,
   test, preregistration, manifest, preflight, command, and registration
   hashes match;
2. cleanup counts, all 64 labels, 61 eligible labels, five disjoint 12-intent
   groups, row hashes, 7,047 source-fit rows, 36 supports per bank, and 96 rows
   per evaluation population match;
3. source fitting contains no target intent or held-out fold;
4. no fold-2 utterance is tokenized, encoded, or scored;
5. the model loads strictly in evaluation mode and all hidden/scientific values
   are finite;
6. inverse-root identity error is at most `1e-10`;
7. random-basis and blind-ZCA singular values match within `1e-10` relative
   and absolute error while basis hashes differ;
8. absolute and margin thresholds use the registered populations, quantiles,
   `higher` method, and strict inequalities, and meet their construction rates;
9. class scores use the maximum of three separate support slots, never a
   prototype or same-class runner-up;
10. degree-five, positive-cosine, independent NumPy, and checked-in float32
    reader predictions/absolute abstentions agree;
11. matched-proximity admits exactly the candidate admission count and uses no
    correctness label;
12. source audit is computed before any target encoding, and failed source
    audit leaves no target hidden cache; and
13. a hidden-cache replay reproduces the scientific object and rows exactly.

Known validity failure yields `HARNESS_INVALID` even if interruption follows.
External interruption before a known invalidity yields `INCONCLUSIVE`.

## 14. Test boundary and stop rules

A passing development result permits one separate immutable test registration
using the same source-fit transform, support banks, calibration groups,
thresholds, arms, rules, and bars on the fixed fold-2 populations.

Development cannot select a new margin quantile, score quantile, threshold,
shrinkage, degree, support count, source sample, intent, operator, prompt,
layer, pooling rule, metric, or bar. If development fails, preserve the
artifacts and keep fold 2 unencoded. A future attempt needs a new mechanism and
a new untouched population, not a variation of HWU64.

## 15. Exact claim boundary

A pass would establish only that a source-error-calibrated class-margin gate,
added to source-calibrated blind-ZCA proximity, transfers across these fixed
HWU64 intent populations in frozen HLM5-136M final-hidden space, improves
selective accuracy against the same absolute gate and an equal-budget
proximity ranking, and retains useful correct admissions.

It would not establish an HLM-specific retrieval advantage, recurrent
attractor dynamics, semantic explanation, production routing, factual
editing, compliance, safety, or scaling. The surviving component would remain
a conventional static selective-routing primitive exposed and tested inside
HLM5.


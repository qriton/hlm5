# I1 Preregistration — Class-Balanced Intent Basin Gate

**Date:** 2026-08-10

**Status:** pre-outcome design; no CLINC encoder outcome may be computed until
the development registration receipt exists

**Scope:** HLM model research only; no KB, Flow, Ingesto, training, deployment,
compliance, or scaling claim

## 1. Why this experiment exists

The registered ParaRel semantic-attractor assay supplied a clean negative
result. Its degree-five settle monotonically lowered the displayed energy and
matched an independent implementation exactly, yet changed 843 of 2,320
correct static routes into errors. The global key-density scalar was valid but
not aligned with address identity.

This experiment changes the scientific question rather than tuning that spent
assay. It uses a new natural-ambiguity task, equal support per intent, explicit
common-mode removal, a training-derived trust region, and a held-out
development/test boundary. It asks whether repeated state dynamics add anything
beyond the strongest static reader built from the same frozen representation.

## 2. Proposition

Let `p[c,r]` be four equal-weight, centered unit prototypes for each of 150
CLINC intents. For a centered unit query `q`, define the degree-five memory
energy

```text
E(q) = -1 / (P * (d + 1)) * sum_i relu(p_i · q)^(d + 1)
d = 5, P = 600.
```

The negative Euclidean gradient field is

```text
h(q) = 1 / P * sum_i relu(p_i · q)^d p_i,
```

and the negative Riemannian gradient on the unit sphere is

```text
g(q) = h(q) - q (q · h(q)).
```

The candidate performs deterministic constrained descent inside a spherical
cap around the input. The cap radius is the median training-only angular
distance from a centered training embedding to its centered class centroid.
The proposition is:

> On natural CLINC150 validation utterances, eight class-balanced constrained
> settle steps improve macro intent accuracy by at least 2 percentage points
> over the strongest registered static control and by at least 0.5 percentage
> points over the identical one-step update, with both paired intent-cluster
> interval lower endpoints above zero, while preserving supported-vs-OOS AUROC
> within 0.01 and avoiding class collapse.

This is a development gate. The official test split remains sealed unless all
development validity and scientific bars pass.

## 3. Fixed external inputs

### Dataset

- official repository: https://github.com/clinc/oos-eval
- commit: `828f8093932c8fe6ca7936c3d2e52903b1c523de`
- source file: `data/data_full.json`
- source SHA-256:
  `1e53b1322dac050a0a601e4ec5157d510633202de752a747dfd9d4f7cf9c1d5c`
- license: Creative Commons Attribution 3.0 Unported
- paper: Larson et al., *An Evaluation Dataset for Intent Classification and
  Out-of-Scope Prediction*, EMNLP-IJCNLP 2019,
  https://aclanthology.org/D19-1131/

The official full split contains 150 intents with 100 train, 20 validation, and
30 test utterances per intent. It also contains 100 validation and 1,000 test
out-of-scope utterances. A pre-outcome tracked-file search found no prior
CLINC150, `oos-eval`, or Banking77 experiment in the HLM2/HLM5 estate.

The outcome-blind manifest census found five exact train/evaluation text
overlaps, including three validation rows and two test rows. Some carry
conflicting labels. Before any encoding, the protocol therefore excludes every
validation or test row whose exact text appears in training, preserves the
training row, and records each exclusion. The fixed evaluated populations are
2,997 validation and 4,498 test in-scope rows. OOS counts are unchanged. This is
a leakage rule derived from raw text identity, not a model outcome.

### Encoder

- `sentence-transformers/all-MiniLM-L6-v2`
- revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- CPU evaluation, normalized encoder output
- all scientific arithmetic after encoding: `torch.float64`

No encoder tuning, prompt prefix, projection training, outcome-fitted
whitening, threshold fit, or label-name encoding is allowed.

## 4. Population and prototype construction

Intent labels are sorted lexicographically and assigned indices `0..149`.
Within each split, source order is preserved.

For every intent, its 100 training utterances are divided into four contiguous
blocks of 25. For each block:

1. encode all 25 utterances;
2. take their arithmetic mean;
3. normalize the mean to unit length.

This produces exactly four raw prototypes per intent and 600 overall. Let `mu`
be the arithmetic mean of the 600 raw prototypes. Define the frozen transform

```text
T(x) = unit(x - mu).
```

Apply `T` to every raw prototype and every query. The centered class centroid is
the unit-normalized mean of its four centered prototypes. Raw centroids are
computed from the four raw prototypes before applying `T`.

The trust radius is

```text
r_train = median arccos(clamp(T(x_train) · centroid[label], -1, 1))
```

over all 15,000 training utterances. This formula is frozen before encoding;
the median is the float64 linear-interpolation 0.5 quantile. The resulting
scalar is derived only from the training split and is emitted in the result.

Development rows are the 2,997 non-overlapping in-scope validation utterances
and 100 OOS validation utterances. Test rows are the 4,498 non-overlapping
in-scope test utterances and 1,000 OOS test utterances. Manifest construction
may read texts, labels,
counts, and hashes but may not load the encoder or compute any routing score.

## 5. Candidate dynamics

For each query, initialize `q0 = T(encoder(text))`. The feasible set is

```text
C(q0) = {q : ||q|| = 1 and arccos(q0 · q) <= r_train}.
```

At each of eight steps:

1. compute the negative Riemannian gradient `g(q)`;
2. if `||g|| <= 1e-12`, copy the state unchanged;
3. otherwise use the unit tangent direction `u = g / ||g||`;
4. propose a spherical exponential-map step with initial angular size
   `r_train / 2`;
5. if the proposal leaves `C(q0)`, project it along the `q0`-to-proposal
   geodesic to the cap boundary;
6. compute the exact sphere log-map displacement `Delta = Log_q(proposal)`;
   its angle is evaluated stably as `atan2(||tangent||, dot)` rather than
   `acos(dot)`, which can round a nonzero small displacement to zero;
7. accept only if `<grad_S E(q), Delta> < 0` and

```text
E(proposal) <= E(q) + 1e-4 * <grad_S E(q), Delta> + 1e-12;
```

8. otherwise halve the angular step, at most 24 times.

If cap projection yields no movement while the direction points outside the
feasible set, the row is copied unchanged and recorded as constrained
stationary.  For the implementation-repaired registration, no movement is the
stable chord predicate `||proposal - q||_2 <= 1e-12`.  On the registered
`[0, pi/2)` cap this is monotone in angular displacement and avoids the
`O(sqrt(machine epsilon))` error of `acos(dot)` near one.  Exhausting the line
search for any other reason is `HARNESS_INVALID`.

The one-step ablation uses the identical implementation and constants with one
outer step. The identity arm uses zero steps and must be bit-exact to `q0`.

## 6. Readouts and controls

For centered prototypes, the degree-five class score is

```text
a_c(q) = (1 / 4) * sum_r relu(p[c,r] · q)^5.
```

The candidate and one-step predictions are `argmax_c a_c(q)`. Registered
static controls are:

1. `centered_degree5`: `argmax_c a_c(q0)`;
2. `centered_centroid`: maximum centered class-centroid cosine at `q0`;
3. `raw_centroid`: maximum raw class-centroid cosine against the raw encoder
   output.

The development-selected static control is the control with highest validation
macro accuracy. Ties use lexicographic control name. Selection is part of the
registered algorithm, not a post-result choice.

Additional controls:

4. independently coded matched constrained settle, required to reproduce final
   states within `1e-10` and every prediction exactly;
5. zero-step identity, required to be bit-exact and reproduce
   `centered_degree5`;
6. a fixed CPU permutation of the 150 output labels using seed `20260810`; its
   candidate macro accuracy must remain below 2%;
7. the one-step arm, which determines whether repeated dynamics are
   load-bearing.

The selected static confidence is its maximum class score. Candidate confidence
is `max_c a_c(q8)`. These scores are used without threshold fitting for
supported-vs-OOS AUROC.

## 7. Metrics and estimators

Primary metrics:

- macro accuracy across the 150 intent-level accuracies;
- paired candidate-minus-selected-static intent differences;
- paired candidate-minus-one-step intent differences;
- two deterministic 10,000-resample intent-cluster intervals, seed `20260810`;
- supported-vs-OOS AUROC using average ranks for ties;
- maximum fraction of in-scope predictions assigned to one intent.

The point statistic and every bootstrap resample use the float64 arithmetic
mean of the 150 paired intent differences. Percentile endpoints use NumPy linear
interpolation.

Diagnostics include every static score, per-intent accuracy, cap occupancy,
accepted angular steps, gradient norms, energy trajectory, directional Armijo
terms, key coherence, singular spectrum, and exact input/trajectory/result
digests.

## 8. Development verdict

Validity gates, all required:

1. source, manifests, model snapshot, environment, Git commit, protocol,
   implementation, tests, command, and registration hashes match;
2. population is exactly 150 labels, 15,000 train, 2,997 non-overlapping
   validation, and 100 validation OOS rows, with four 25-row prototypes per
   label and the three registered validation exclusions;
3. all tensors, scores, energies, and diagnostics are finite;
4. states are unit-normalized within `1e-10` and remain inside the registered
   cap within `1e-10`;
5. every accepted transition satisfies the displayed descent inequality;
6. identity, static equivalence, independent matched settle, and shuffled-label
   controls pass;
7. `0 < r_train < pi / 2`;
8. no test text is encoded or scored during preflight, registration, or the
   development run.

Any failed validity gate yields `HARNESS_INVALID`.

Development `PASS_TO_TEST`, all required:

1. candidate minus selected-static macro accuracy is at least `+0.020`;
2. its intent-cluster interval lower endpoint is strictly above `0.000`;
3. candidate minus one-step macro accuracy is at least `+0.005`;
4. its intent-cluster interval lower endpoint is strictly above `0.000`;
5. candidate AUROC is at least selected-static AUROC minus `0.010`;
6. no intent receives more than 3% of in-scope predictions.

Otherwise the development verdict is `STOP_BEFORE_TEST`. Infrastructure
interruption before all rows exist is `INCONCLUSIVE` only if no validity failure
is already known.

## 9. Test boundary

`--register-test` must refuse unless the immutable development result says
`PASS_TO_TEST` and its raw/result digests recompute. The test receipt binds the
selected static control, training-derived radius, development result digest,
unchanged implementation hashes, test manifest, and exact test command.

The test run uses the same six scientific bars and no changed parameter. Test
`PASS` licenses only the statement that this fixed, class-balanced constrained
settle improved frozen-encoder CLINC intent routing on the registered split. A
test failure closes the candidate. Neither outcome authorizes training or
scaling.

## 10. Stop rules

- `STOP_BEFORE_TEST`: do not encode the official test split and do not tune the
  degree, prototype count, centering, trust radius, steps, line search, or
  encoder on validation outcomes.
- `HARNESS_INVALID`: repair only the implementation-contract violation while
  preserving scientific constants and archive the invalid attempt.
- `PASS_TO_TEST`: register the unchanged test run before any test embedding is
  computed.
- Test `FAIL`: close this objective and population; no test-driven repair.

## 11. Explicit non-claims

This assay cannot establish semantic understanding, factual correctness,
language-model improvement, EU AI Act compliance, production readiness,
universal explainability, robustness outside CLINC150, or an HLM-specific
implementation advantage. The independent matched optimizer is mandatory, so
any positive result remains reproducible as conventional constrained
mean-shift-style state optimization.

## 12. Implementation-only amendment after invalid attempt 01

The first registered development execution was `HARNESS_INVALID`: one
cap-boundary row exceeded the independent-state tolerance because
`acos(dot)` reported an apparent `1.49e-08` movement for a projected
roundoff-scale chord.  The complete attempt and its original receipts are
preserved under `results/invalid-attempt-01/`.

Before attempt 02, both implementations changed only their no-movement
predicate to the chord definition in Section 5, and a deterministic
384-dimensional regression was added.  Degree, encoder, prototypes, centering,
trust-radius derivation, step schedule, Armijo rule, controls, scientific bars,
data, and seeds are unchanged.  A retry requires a new clean preflight and a
new pre-outcome development registration.

Attempt 02 remained `HARNESS_INVALID` on the same row.  Stepwise diagnostics
showed that both implementations agreed through step 7 within `5.6e-17` and
generated the same step-8 projected state within `2e-16`.  One dot product
rounded to exactly one, however, so `acos(dot)` produced a zero log-map and the
independent line search took a different branch.  Before attempt 03, both
separately structured log-map calculations changed to the stable, mathematically
equivalent `atan2` form above, with a synthetic regression in which `dot == 1`
but the true chord and angle are nonzero.  The saved spent-row tensors confirm
that both branches then accept the same proposal within machine precision.
Attempt 02 is preserved under `results/invalid-attempt-02/`; a third attempt
again requires a new clean preflight and pre-outcome registration.

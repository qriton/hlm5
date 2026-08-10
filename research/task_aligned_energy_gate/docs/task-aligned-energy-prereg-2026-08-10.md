# B77-E1 Preregistration — Task-Aligned Energy Decomposition

**Date:** 2026-08-10

**Status:** pre-outcome design; no Banking77 encoder outcome may be computed
before the development registration receipt exists

**Scope:** HLM model research only; no KB, Flow, Ingesto, deployment,
compliance, language-model scaling, or polynomial-Hopfield advantage claim

## 1. Question and decomposition

Two valid negative assays established that a fixed degree-five prototype
density can descend exactly while damaging semantic routing. This experiment
does not tune those spent populations. It moves to untouched BANKING77 and
separates the surviving alignment hypothesis from the unsettled recurrence
hypothesis:

1. **Alignment:** a label-fitted Fisher/LDA metric improves the strongest
   unaligned centroid reader.
2. **Dynamics:** exact descent on the resulting class-mixture energy improves
   that identical aligned reader at zero steps and improves a one-step dose.

The first claim may pass even if the second fails. Test access is claim-scoped.

## 2. Fixed source and population

- repository: https://github.com/PolyAI-LDN/task-specific-datasets
- commit: `57ec275d8078af65b7731c2a98be812d844a6d6b`
- license: CC-BY-4.0
- train CSV SHA-256:
  `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b`
- test CSV SHA-256:
  `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d`
- 77 intents; 10,003 official training rows; 3,080 official test rows.

An outcome-blind census found four within-train duplicate groups after Unicode
NFKC normalization, case folding, trimming, and whitespace collapse. There are
no label conflicts. Keep the lowest source index and drop the four later rows,
leaving 9,999 unique training texts.

Within each intent, sort rows by SHA-256 of
`split_salt + NUL + intent + NUL + normalized_text`, then source index. Assign
the first 10 to calibration, the next 10 to development, and all remaining
rows to fit. Counts are 770 calibration, 770 development, and 8,459 fit. Every
intent contributes exactly 10 calibration and 10 development rows; the
smallest fit class has 15 rows.

The official test has no exact train overlap. Seven rows match a training text
after the registered normalization and carry the same label. Exclude those
seven from the 3,073-row primary test population before encoding. The full
3,080-row official score is a transparent secondary metric only if test access
is later authorized.

## 3. Frozen representation

- encoder: `sentence-transformers/all-MiniLM-L6-v2`
- revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- normalized encoder output; CPU
- all scientific arithmetic after encoding: `torch.float64`
- no encoder tuning, prompt, label-name encoding, learned neural projection,
  or development/test-fitted transform.

## 4. Task-aligned metric fit

For fit embeddings `x_i` with labels `y_i`, compute the global mean `mu`, class
means `mu_c`, pooled within-class covariance
`S_w = (1/n) sum_i (x_i-mu_{y_i})(x_i-mu_{y_i})^T`, and equal-class between
covariance `S_b = (1/77) sum_c (mu_c-mu)(mu_c-mu)^T`. For each registered shrinkage

```text
alpha in {0.05, 0.25, 0.50, 0.75, 1.00}
S_alpha = (1-alpha) S_w + alpha tr(S_w)/D I.
```

Let `B = S_alpha^(-1/2)`. Eigendecompose `B S_b B` and retain the 76 largest
eigenvectors. With deterministic eigenvector sign canonicalization, define the
projection `A = U_76^T B`, state `z = A(x-mu)`, and projected fit centroid
`m_c = mean(z_i | y_i=c)`.

The aligned static score is the equal-prior LDA/Gaussian score

```text
s_c(z) = m_c dot z - 0.5 ||m_c||^2.
```

Select `alpha` solely by calibration macro accuracy of this zero-step score;
ties choose the numerically smallest alpha. This grid and rule are frozen
before any real embedding. Development never selects a model or parameter.

Unaligned controls use the same fit rows and frozen embeddings:

- raw unit class-centroid cosine;
- global-mean-centered unit class-centroid cosine.

Select the unaligned comparator solely by calibration macro accuracy; ties use
lexicographic control name.

## 5. Displayed energy and dynamics

For the selected aligned model, the class-balanced mixture energy is

```text
E(z) = 0.5 ||z||^2 - log sum_c exp(s_c(z)).
```

Its exact negative gradient is

```text
g(z) = sum_c softmax(s(z))_c m_c - z.
```

This is a supervised equal-covariance Gaussian-mixture / modern-Hopfield-style
energy. It is not the failed degree-five density and it is not claimed to be
different from conventional softmax attention or mean shift.

Derive the Euclidean trust radius from fit only: for every correctly classified
fit row, compute its smallest positive distance to a competing LDA decision
hyperplane, `(s_y-s_j)/||m_y-m_j||`; the radius is the float64 linear median of
those distances. It must be finite and positive.

Starting at `z0`, each step proposes movement along the unit negative-gradient
direction with initial length `radius/2`, projects outside proposals onto the
closed Euclidean ball `||z-z0|| <= radius`, and accepts only when the direction
is descending and

```text
E(proposal) <= E(z) + 1e-4 grad(E,z) dot (proposal-z) + 1e-12.
```

Backtrack by halves at most 24 times. A zero gradient is stationary. At the cap
boundary, a no-movement projected proposal whose direction points outward is
constrained stationary. Any other exhaustion is `HARNESS_INVALID`.

The candidate uses eight steps. The dose control uses one identical step. The
identity uses zero steps and must be bit-exact. A separately structured matched
optimizer must reproduce final states within `1e-10` and predictions exactly.

## 6. Additional controls

- A fixed seed-`20260810` permutation of fit target labels preserves label
  counts. Refit the selected-alpha LDA model and require its development static
  macro accuracy below 3%.
- Zero-step aligned scores must exactly reproduce direct LDA scores.
- No candidate parameter is absent from its zero-step reader; recurrence adds
  computation but no trainable capacity.
- Raw and centered centroids remain conventional matched-data controls.

## 7. Metrics

Primary accuracy is the arithmetic mean of 77 intent accuracies. Paired
differences use intent-level values and a deterministic 10,000-resample paired
intent bootstrap, seed `20260810`; point and resamples use the float64 mean and
NumPy linear percentile interpolation. Also report row accuracy, per-intent
values, changed-row transition counts, maximum predicted-intent share, energy
trajectory, cap occupancy, gradient norms, halvings, fitted spectra, selected
shrinkage, and complete hashes.

## 8. Validity and development verdicts

All validity gates are binding:

1. source, manifests, encoder snapshot, environment, Git head,
   implementation, tests, command, and registration hashes match;
2. exact 77-label fit/calibration/development populations and registered
   duplicate exclusions;
3. calibration selection follows the frozen grid and tie rules;
4. every value is finite; fitted covariance is positive definite;
5. every accepted transition satisfies the displayed energy and Armijo
   inequalities and remains inside the cap;
6. identity, static equivalence, independent optimizer, and shuffled-label
   controls pass;
7. trust radius is finite and positive;
8. no official test text is encoded or scored.

Any failure is `HARNESS_INVALID`; an infrastructure interruption is
`INCONCLUSIVE` only when no invalidity is known.

**Alignment PASS**, all required:

1. aligned static minus selected unaligned static macro `>= +0.020`;
2. paired intent interval lower endpoint `> 0.000`;
3. aligned static maximum predicted-intent share `<= 0.050`.

**Dynamics PASS**, all required:

1. candidate minus aligned static macro `>= +0.010`;
2. paired intent interval lower endpoint `> 0.000`;
3. candidate minus one-step macro `>= +0.005`;
4. paired intent interval lower endpoint `> 0.000`;
5. candidate maximum predicted-intent share `<= 0.050`.

Verdicts are `PASS_BOTH_TO_TEST`, `PASS_ALIGNMENT_TO_TEST`, or
`STOP_BEFORE_TEST`. Dynamics alone cannot proceed if alignment fails because
the stated mechanism is the composed aligned system.

## 9. Test boundary and stop rules

Test registration verifies the immutable development result and opens only the
claim class that passed. Alignment-only access evaluates aligned static versus
the frozen selected unaligned control and does not run dynamics. Both-claim
access evaluates unchanged alignment and dynamics bars. No test-driven repair
is allowed.

- Alignment-only test PASS supports only train-fitted Fisher/LDA
  preconditioning on this Banking77 split.
- Both-claim test PASS additionally supports the fixed eight-step class-mixture
  settle over its matched zero/one-step readers.
- Any failed test bar closes that claim.
- No result authorizes LM training or scaling.

## 10. Explicit non-claims

This assay cannot establish semantic understanding, factual correctness,
universal explainability, EU AI Act compliance, production readiness, safety,
robustness outside Banking77, or an HLM-specific advantage. A dynamics pass
would be reproducible as conventional gradient descent on the displayed
Gaussian-mixture scalar.

## 11. Implementation-only amendment after invalid attempt 01

The first registered execution was `HARNESS_INVALID`. Every validity gate
except independent state equality passed. The two optimizers produced identical
predictions and final cap norms agreeing at approximately `1e-15`, but their
cap-surface coordinates differed by `5.102135665069341e-07`. The independent
path had manually rewritten several stable numerical primitives, including
softmax, row norms, displayed energy, and ball projection. Repeated gradient
evaluation and cap projection accumulated their rounding differences.

The repair gives both paths the same canonical CPU float64 primitives for
softmax, row norm, evaluation of the displayed scalar, and projection to the
registered ball; optimizer control flow and Armijo orchestration remain
separately structured. Dataset, split, encoder, covariance grid and selection
rule, energy, temperature, radius, steps, bars, controls, and seeds are
unchanged. The exact original receipts, rows, result, and verdict are preserved
under `results/invalid-attempt-01/`. A retry requires a new clean preflight and
pre-outcome registration.

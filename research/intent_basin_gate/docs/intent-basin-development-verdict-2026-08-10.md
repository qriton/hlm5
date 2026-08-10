# I1 Development Verdict — Class-Balanced Intent Basin Gate

**Date:** 2026-08-10

**Experiment:** `hlm5-i1-class-balanced-intent-basin-v1`

**Formal verdict:** `STOP_BEFORE_TEST`

**Validity:** clean; all eleven registered validity gates passed

**Scope:** HLM model research only; no KB, Flow, Ingesto, training, scaling,
deployment, compliance, or universal-explainability claim

## 1. Conclusion

The fixed eight-step degree-five settle does not improve natural intent
routing after equalizing class support, removing the prototype common mode, and
constraining movement to a training-derived spherical trust region.  The
selected static raw-centroid reader reached macro accuracy
`0.9178947368421052`; one settle step reached `0.9065614035087719`; eight steps
reached `0.7661754385964913`.

The candidate-minus-static difference was `-0.15171929824561403`, with the
registered intent-cluster interval
`[-0.19901798245614036, -0.10833333333333334]`.  The candidate-minus-one-step
difference was `-0.14038596491228073`, with interval
`[-0.18542105263157893, -0.09840350877192983]`.  Supported/OOS AUROC fell from
`0.9819085752419086` to `0.9110744077410744`, a difference of
`-0.07083416750083416`.

This is a valid negative result, not a certificate or implementation failure.
The independent settle matched final states within
`4.1494585545365226e-15`, every registered validity gate passed, and candidate
predictions did not collapse into one intent.  The official CLINC150 test split
therefore remains sealed and must not be registered or encoded.

## 2. Registered execution

The valid run used the exact pre-outcome commit and registered command:

```text
git commit: 3dd844fb7b2804bf249e257b856b9c57632038f0
command:    python -B run_intent_basin_gate.py --run-development
device:     CPU
dtype:      torch.float64 after encoding
```

The development population contained 150 intents, 15,000 balanced training
utterances, 2,997 non-overlapping validation utterances, and 100 validation OOS
utterances.  Three exact train/validation text overlaps were excluded before
encoding.  Each intent had four fixed 25-example prototypes.

Inputs came from official CLINC150 source commit
`828f8093932c8fe6ca7936c3d2e52903b1c523de` and were encoded with frozen
`sentence-transformers/all-MiniLM-L6-v2` revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.  The trust radius derived only
from training data was `0.809161756386441` radians.

## 3. Binding scientific bars

| Registered bar | Required | Measured | Passed |
|---|---:|---:|:---:|
| Candidate minus selected static macro | `>= +0.020` | `-0.15171929824561403` | No |
| Static comparison interval lower endpoint | `> 0.000` | `-0.19901798245614036` | No |
| Candidate minus one-step macro | `>= +0.005` | `-0.14038596491228073` | No |
| One-step comparison interval lower endpoint | `> 0.000` | `-0.18542105263157893` | No |
| Candidate minus static supported/OOS AUROC | `>= -0.010` | `-0.07083416750083416` | No |
| Maximum predicted-intent share | `<= 0.030` | `0.02569235902569236` | Yes |

Only the anti-collapse bar passed.  The static controls were selected exactly
as preregistered: raw centroid `0.9178947368421052`, centered degree-five
`0.9132280701754386`, and centered centroid `0.9095438596491227`.

## 4. Validity gates

All registered validity gates passed:

| Gate | Result |
|---|:---:|
| Registration, hashes, environment, Git head, and command verified | PASS |
| Development population exact | PASS |
| All tensors, scores, and diagnostics finite and unit-normalized | PASS |
| Energy inequality satisfied within `1e-12` | PASS |
| Armijo inequality satisfied within `1e-12` | PASS |
| Every state remained inside the training-derived cap | PASS |
| Zero-step identity bit-exact | PASS |
| Independent matched settle | PASS |
| Shuffled-label null below `0.020` | PASS |
| Trust radius in `(0, pi/2)` | PASS |
| No test text encoded | PASS |

The maximum independent-state discrepancy was
`4.1494585545365226e-15`.  The largest reported energy increase was
`9.993620514459067e-13`, the largest Armijo residual was
`9.994500886623126e-13`, maximum cap error was
`3.3306690738754696e-16`, and maximum unit-norm error was
`6.661338147750939e-16`.  Shuffled-label macro accuracy was `0.011`.

The environment emitted a SciPy/scikit-learn warning because local NumPy was
`2.4.6` while those optional packages declared a lower supported range.  No
SciPy/scikit-learn operation entered the registered estimators.  Frozen encoder
hashes, deterministic row/result digests, finite checks, and independent
controls all passed; the warning remains replication environment debt rather
than a validity failure.

## 5. What the raw rows show

The candidate and selected static reader disagreed on 603 of 2,997 in-scope
utterances.  Of those changes:

- 501 changed a correct static prediction into an error;
- 46 corrected a static error;
- 2,250 were correct under both readers;
- 200 were wrong under both readers.

Across 150 intents, the candidate improved 15, tied 66, and worsened 69.  The
largest losses included `improve_credit_score` (`1.00` to `0.00`), `mpg`
(`1.00` to `0.05`), and four intents losing at least 0.90 accuracy.  There were
real local gains, including `shopping_list` (`0.45` to `0.95`), but they were
outweighed broadly and decisively.

The eight-step candidate also disagreed with the one-step arm on 551 rows.  It
turned 461 one-step correct predictions into errors while rescuing 40 one-step
errors.  This step-dose ordering is consistent with repeated movement
compounding objective/task misalignment rather than repairing ambiguous
queries.

All 3,097 primary and independent predictions agreed.  No row entered the
constrained-stationary state within eight registered steps, so the negative
result concerns the fixed trained-depth schedule, not convergence to a proven
fixed point.

## 6. Interpretation

This experiment deliberately removed two ambiguities from the earlier ParaRel
failure.  The static baseline had genuine headroom rather than perfect
accuracy, and every intent contributed equal memory support.  Common-mode
centering and a training-only trust region also prevented unconstrained drift.
The repeated settle still lost 15.17 macro-accuracy points.

The supported conclusion is narrow:

> Class balancing, centering, and conservative movement bounds do not make this
> fixed higher-order density energy align with CLINC150 intent correctness.
> Valid energy descent remains evidence about the registered scalar, not about
> the semantic validity of the resulting decision.

A plausible mechanism is that the degree-five prototype field pulls queries
toward locally dense directions that need not preserve discriminative class
boundaries.  The worsening from static to one step to eight steps supports that
reading, but it does not independently identify the exact geometric cause for
every intent.

## 7. Harness correction history

Two pre-outcome lineages produced `HARNESS_INVALID` rather than scientific
results and are preserved under `results/invalid-attempt-01/` and
`results/invalid-attempt-02/`.

Both invalidities came from one cap-boundary row.  The two implementations
agreed through step 7 within `5.6e-17` and generated the same step-8 projected
proposal within approximately `2e-16`, but one near-one dot product rounded to
exactly `1.0`.  `acos(dot)` therefore returned a zero log-map in one path and a
nonzero value in the other, changing the line-search branch.  The final repair
used the stable, mathematically equivalent
`atan2(||tangent||, dot)` sphere-log angle and added a high-dimensional
regression where `dot == 1` while the true chord is nonzero.

No scientific constant, model, dataset, seed, control, bar, or registered
command changed across the invalid and valid attempts.  This correction is a
durable numerical lesson: near-coincident spherical geometry must not use
`acos(dot)` as its small-angle oracle.

## 8. What survives and what stops

What survives:

- frozen MiniLM plus raw-centroid intent routing on this development set;
- the static class-balanced prototype readers as conventional baselines;
- the stable constrained-geometry implementation and independently matched
  optimizer as research instrumentation;
- energy/replay evidence when described only as evidence of the bound
  computation;
- the discipline of task metrics, dose controls, nulls, and a sealed test set.

What stops:

- this degree-five eight-step intent settle on the spent CLINC150 development
  population;
- official test registration or encoding;
- post-outcome tuning of degree, steps, radius, prototype blocks, centering,
  encoder, or line-search constants on these validation outcomes;
- scaling or product claims from this objective;
- any claim that monotone energy movement explains or validates the intent
  decision.

A future dynamics experiment requires a new untouched population and an energy
explicitly trained or constructed against the task margin.  The static reader,
one-step dose, independently matched optimizer, and sealed holdout remain
mandatory controls.

## 9. Evidence and digests

| Artifact | SHA-256 or registered digest |
|---|---|
| Derived CLINC development manifest | `8873d168557c0ff274c4a99b41d7731fe23f1d8d77cbde5358643c4fa1d0bbae` |
| Sealed CLINC test manifest | `17fd571881c7dbefbb55175330b4eee489557a1b7d8bef8addecf5f177c62b3f` |
| Preflight receipt file | `e737e0bdf8580e8b92e76384e32c725672571a6d250db509bca24593e7ed6b0f` |
| Registration file | `e8d37caa0cedb0c7855cbb782b167da4c4348ef6a917d263563621116d352b44` |
| Registration scientific digest | `ee18ba9b78875c25486430c78330e7f90943e82bb9565182e6f6e837a2e56aff` |
| Raw rows file | `2aaa74345358ac342a163b423e29ffb086c160b5947d443556b027b348191e60` |
| Result scientific digest | `ecfb567c7575db736c429486e3c69ac35d555fd3436a17abababe3bc72473b08` |
| Result JSON file | `0b2708d5eb737a2d1c33aba21e36289ee5ca5499003e84f88df6b0833d381c37` |

The raw row file contains exactly 3,097 JSON lines: 2,997 in-scope validation
rows and 100 validation OOS rows.  Independent recomputation reproduced every
reported binding metric exactly.  The embedded raw-row hash, registration hash,
registration scientific digest, and result digest all recompute.

## 10. Claim boundary

This result does not establish semantic understanding, factual correctness,
language-model improvement, production readiness, EU AI Act compliance,
universal explainability, robustness beyond CLINC150 development, or an
HLM-specific advantage.  It is one valid negative development result for one
frozen energy, optimizer, encoder, population, and set of registered controls.

# SCA-1 Development Verdict — Source-Calibrated HLM5 Admission Transfer

**Date:** 2026-08-10

**Formal verdict:** `STOP_BEFORE_TEST`

**Harness status:** `VALID` — 37/37 binding validity checks passed
**Test boundary:** preserved; no TOPv2 test utterance was tokenized, encoded, or scored

## Decision

Blind source-domain ZCA is a real, useful HLM5 hidden-space primitive, but the
registered single-score admission rule is not yet reliable enough to advance
to the sealed test population.

The candidate passed 11 of 12 scientific bars. It more than doubled target
macro correct-admission relative to raw keys, beat diagonal scaling and the
spectrum-matched random-basis control with positive paired-intent intervals,
increased coverage, avoided prediction collapse, preserved top-1 accuracy,
and controlled the disjoint off-support false-admission rate. It failed the
aggregate selective-accuracy bar:

`0.5714 < 0.6500`.

That single failure is binding. No aggregate gain overrides it.

## Frozen result

| Arm | Macro top-1 | Macro coverage | Macro correct-admission | Selective accuracy | Off-support FAR |
|---|---:|---:|---:|---:|---:|
| raw | 0.3444 | 0.1778 | 0.1000 | 0.5625 | 0.0222 |
| centered | 0.3333 | 0.2278 | 0.1111 | 0.4878 | 0.0278 |
| diagonal standardization | 0.3833 | 0.1889 | 0.1333 | 0.7059 | 0.0222 |
| spectrum-matched random basis | 0.3556 | 0.1778 | 0.1056 | 0.5938 | 0.0278 |
| **blind ZCA** | **0.4000** | **0.3889** | **0.2222** | **0.5714** | **0.0222** |

Blind ZCA admitted 70 of 180 target rows. Forty were correct and 30 were
wrong. Its maximum predicted-intent share was 0.1833, below the registered
0.25 collapse bar.

Candidate-minus-control paired-intent differences in macro
correct-admission were:

| Comparator | Point difference | Descriptive 95% bootstrap interval | Intents improved / tied / worsened |
|---|---:|---:|---:|
| raw | +0.1222 | [0.0611, 0.1889] | 7 / 2 / 0 |
| diagonal standardization | +0.0889 | [0.0444, 0.1389] | 7 / 2 / 0 |
| spectrum-matched random basis | +0.1167 | [0.0556, 0.1833] | 7 / 2 / 0 |

The intervals describe stability over the nine fixed development intents;
they are not independent population-level confidence intervals.

## What worked

1. **Covariance directions were load-bearing.** Blind ZCA beat both diagonal
   scale correction and a transform with the same singular spectrum in a
   random basis. The effect was not merely centering, per-coordinate variance,
   or generic anisotropic gain.
2. **Source-only open-set calibration transferred.** The blind-ZCA threshold
   admitted 8/180 registered calibration negatives (0.0444) and only 4/180
   disjoint off-support audit rows (0.0222), without target-domain threshold
   fitting.
3. **Frozen HLM5 hiddens carried useful three-shot target structure.** Blind
   ZCA reached 0.4000 target top-1 and 0.2222 correct-admission from three
   separate supports per intent. This is a frozen-representation result; no
   model parameter was changed.
4. **The measurement was reproducible.** The in-process cache replay and a
   second external invocation reproduced scientific SHA-256
   `53861b1e0a4760791a981d420041144f029987b6ec2555ff228897fe5d1f0892`
   and the row artifact exactly.

## What did not work

The absolute maximum support score is a good open-set signal but an
insufficient correctness signal among nearby supported intents. The source
calibration positives were 0.8333 selective-accurate, while the untouched
target rows were only 0.5714. Most wrong admissions were confusions among the
closely related reminder intents; the threshold could tell that a query was
near the address bank, but not that its winning address was sufficiently
separated from the runner-up.

This narrows the failure. The useful operator survived; the one-dimensional
admission statistic did not transfer its conditional correctness.

## Consequence and next bounded question

Preserve blind ZCA and source-only open-set calibration as candidate
components. Set aside the claim that a single maximum-score threshold yields
valid cross-domain admission.

Do not tune TOPv2 and do not encode its test population. A future assay must
use a new untouched population and preregister a genuinely new admission
mechanism that separates two questions:

1. absolute support proximity for open-set rejection; and
2. winner-versus-runner-up separation for conditional routing risk.

A source-calibrated two-axis gate (maximum score plus a class-ambiguity margin
or an equivalent risk-control statistic) is the constructive next hypothesis.
It must retain raw, diagonal, and spectrum-matched random-basis controls and
must be registered before any new-population model outcome.

## Provenance

- preregistration commit: `143ef8b`
- frozen implementation commit: `28f3be3`
- preflight SHA-256:
  `29979ba9f15dadff58e29235d2dee7d42c9bc85602ed6030abfb9ae2f1552629`
- development registration SHA-256:
  `eaab1e56a3e2bf1a12de0009c5e975882d6126c5cd934f436cf78144f191e72d`
- result SHA-256:
  `798ee38ee2768c4f1bc2413d01b50825c6aec5753cf1892af3f359fe82af641f`
- row JSONL SHA-256:
  `2fd5c54fa5f45faae2ae40c9f672988fc11349339038f0fd4ff4f653e617f276`
- hidden tensor SHA-256 recorded by the result:
  `66f527434e6ee4f310c4c8e492f880d04b3adfc253ce098ee72128049343dc93`
- local cache-file SHA-256 (generated, not committed):
  `6419f078aca9c9509c100d2854c7408ec870cfd8d6ffa6996bde818fdd45016c`

The cache-equivalent degree-five reader establishes no recurrence, attractor,
energy-descent, HLM-specific retrieval, product, compliance, or scaling claim.

# B77-E1 Test Verdict — Task-Aligned Preconditioning

**Date:** 2026-08-10

**Experiment:** `hlm5-b77-task-aligned-energy-v1`

**Authorized claim scope:** `alignment_only`

**Formal verdict:** `PASS`

**Validity:** clean; all registered test validity and alignment bars passed

## 1. Conclusion

The development-positive alignment result replicated on the sealed official
BANKING77 test population. On the 3,073-row primary split, the frozen raw
centroid reader reached macro accuracy `0.847467882994199`; the train-fitted
Fisher/LDA reader reached `0.8995162732004839`. The gain was
`0.052048390206284934`, with registered paired intent-cluster interval
`[0.030619818777713515, 0.07658468724258193]`.

All three test bars passed. This validates the alignment/preconditioning fix
class on a new untouched population. It does not validate recurrence: the
development settle had changed zero predictions and failed its gate, so the
test receipt prohibited one-step and eight-step execution. The test raw rows
contain no dynamics fields.

## 2. Test population and frozen model

- official source test rows: `3,080`, balanced at 40 per intent;
- primary rows scored: `3,073`;
- normalized train-overlap rows excluded from primary scoring: `7`;
- selected shrinkage, frozen from calibration: `0.05`;
- selected comparator, frozen from calibration: `raw_centroid`;
- encoder/model revision and all implementation hashes matched registration;
- fitted model digests and fit-derived trust radius reproduced exactly.

All 3,080 official test texts were encoded only after the alignment-only test
receipt authorized test access. The seven registered overlap rows were absent
from the primary raw rows and metrics; no full-official secondary score was
emitted.

## 3. Binding test result

| Arm | Primary macro accuracy |
|---|---:|
| Raw centroid | `0.847467882994199` |
| Centered centroid | `0.8458445063708223` |
| Fisher/LDA aligned static | `0.8995162732004839` |

| Bar | Required | Measured | Passed |
|---|---:|---:|:---:|
| Aligned minus selected unaligned macro | `>= +0.020` | `+0.052048390206284934` | Yes |
| Intent-cluster interval lower endpoint | `> 0.000` | `+0.030619818777713515` | Yes |
| Maximum aligned predicted-intent share | `<= 0.050` | `0.017246989912137977` | Yes |

The aligned reader improved 48 intents, tied 18, and worsened 11. Relative to
raw centroid it corrected 221 previously wrong rows while breaking 61
previously correct rows, for a net gain of 160 correct decisions. The largest
intent-level gains were `supported_cards_and_currencies` (`0.35` to `0.90`) and
`topping_up_by_card` (`0.40` to `0.90`). These descriptive rows were not used
for model or parameter selection.

## 4. Replication across development and test

| Population | Raw centroid | Aligned static | Gain | Interval lower |
|---|---:|---:|---:|---:|
| Development, 770 | `0.8441558441558441` | `0.8883116883116886` | `+0.044155844155844164` | `+0.014285714285714292` |
| Test primary, 3,073 | `0.847467882994199` | `0.8995162732004839` | `+0.052048390206284934` | `+0.030619818777713515` |

The effect grew slightly on test and retained a positive intent-cluster lower
endpoint. The result therefore does not depend on the 770-row development
partition or on a single aggregate.

## 5. What this teaches HLM5

The positive mechanism is not “more energy” and not repeated settling. It is
the task-aligned coordinate system:

```text
frozen MiniLM state
    -> train-only within-class whitening
    -> between-class discriminant subspace
    -> equal-prior LDA readout
```

That is a concrete instance of the wider HLM finding that the composed,
task-restricted Gram matters more than raw gain. The same fitted centroids also
defined a valid scalar energy, but descending it moved almost every development
state to its trust boundary without changing a single decision. Keep the
preconditioner; remove the redundant recurrence for this use case.

## 6. Evidence and independent recomputation

| Artifact | SHA-256 or scientific digest |
|---|---|
| Test registration | `8ad33fa41c58bb9c5f98e20e54d005ddbe4b19658aa70f5bda7cd9c625415a5c` |
| Test registration scientific digest | `dead0f06d64509d22badccaed1fa85acd86fe9b75a72b2694ff1dd808624b987` |
| Raw primary test rows | `a131e090199a7fa780de432824751bba61fba0a12e64d45c2ace0cd052df7f12` |
| Test result file | `b7466db3d5608dcd6595ca93a46a51dfd4d2e0b28b995d8234fb9896eff30e9d` |
| Test result digest | `063a6fa77a51bcfac7091e07852942a55bee73943fdf79625c1022275e259c59` |
| Official feature digest | `34d9559d0f2c8998e7c2a274050a6c6bfae3d01efcf3452eba7b379b2cc2fa39` |

Independent raw-row recomputation reproduced the three static macros, paired
interval, raw-row hash, and result digest exactly. It also verified that the
3,073 raw row IDs equal the registered primary manifest, none of the seven
excluded IDs was scored, and no candidate or one-step prediction is present.

## 7. Claim boundary

The supported claim is narrow:

> On this fixed frozen-MiniLM BANKING77 protocol, label-fitted Fisher/LDA
> preconditioning improved macro intent accuracy over the calibration-selected
> unaligned centroid reader by 5.20 percentage points on the sealed primary
> test population.

This is conventional supervised metric learning, not evidence of a proprietary
HLM advantage. It does not establish semantic understanding, universal
explainability, factual correctness, robustness outside BANKING77, compliance,
production readiness, or a reason to train or scale an HLM language model.

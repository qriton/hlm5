# B77-E1 Development Verdict — Task-Aligned Energy Decomposition

**Date:** 2026-08-10

**Experiment:** `hlm5-b77-task-aligned-energy-v1`

**Formal verdict:** `PASS_ALIGNMENT_TO_TEST`

**Validity:** clean; all eleven registered validity gates passed

**Scope:** HLM model research only; no KB, Flow, Ingesto, deployment,
compliance, language-model scaling, or HLM-specific advantage claim

## 1. Conclusion

The experiment cleanly separates the useful mechanism from the inert one.
Train-fitted Fisher/LDA preconditioning improved the calibration-selected raw
centroid reader from macro accuracy `0.8441558441558441` to
`0.8883116883116886`, a gain of `0.044155844155844164`. The registered paired
intent-cluster interval was
`[0.014285714285714292, 0.07792207792207792]`. All three alignment bars passed.

Energy recurrence added nothing. Zero steps, one step, and eight exact descent
steps all produced macro accuracy `0.8883116883116886` and exactly the same 770
predictions. Both registered dynamics differences and intervals were exactly
zero, so four of five dynamics bars failed. The official test was therefore
authorized for alignment only; the recurrent arm was stopped.

This is the result the wider HLM record predicted: task-aligned
preconditioning works; more movement on an already aligned channel is not
automatically useful.

## 2. Frozen development procedure

- official BANKING77 source commit:
  `57ec275d8078af65b7731c2a98be812d844a6d6b`;
- frozen encoder: `sentence-transformers/all-MiniLM-L6-v2` revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
- fit/calibration/development counts: `8459 / 770 / 770`;
- exactly 10 calibration and 10 development rows per intent;
- selected covariance shrinkage: `0.05` from the registered five-value grid;
- selected unaligned comparator: `raw_centroid`;
- all post-encoder arithmetic: CPU `torch.float64`;
- trust radius from fit only: `4.720266704071971`.

The registered runner encoded fit and calibration first, selected shrinkage and
the unaligned comparator, and only then encoded development. No official test
text was encoded during development.

## 3. Development arms

| Arm | Macro accuracy | Maximum intent share |
|---|---:|---:|
| Raw centroid | `0.8441558441558441` | see raw result |
| Centered centroid | `0.8350649350649351` | see raw result |
| Aligned static | `0.8883116883116886` | passed `<= 0.050` |
| One exact descent step | `0.8883116883116886` | see raw result |
| Eight exact descent steps | `0.8883116883116886` | passed `<= 0.050` |
| Shuffled-label aligned static | `0.02467532467532468` | required `< 0.030` |

Alignment improved 26 intents, tied 38, and worsened 13. The eight-step arm
changed zero predictions relative to either zero or one step: 684 rows were
correct under both and 86 were wrong under both.

## 4. Binding bars

### Alignment

| Bar | Required | Measured | Passed |
|---|---:|---:|:---:|
| Aligned minus selected unaligned macro | `>= +0.020` | `+0.044155844155844164` | Yes |
| Intent-cluster interval lower endpoint | `> 0.000` | `+0.014285714285714292` | Yes |
| Maximum aligned predicted-intent share | `<= 0.050` | passed | Yes |

### Dynamics

| Bar | Required | Measured | Passed |
|---|---:|---:|:---:|
| Candidate minus aligned static macro | `>= +0.010` | `0.000` | No |
| Static interval lower endpoint | `> 0.000` | `0.000` | No |
| Candidate minus one-step macro | `>= +0.005` | `0.000` | No |
| One-step interval lower endpoint | `> 0.000` | `0.000` | No |
| Maximum candidate predicted-intent share | `<= 0.050` | passed | Yes |

## 5. Why the dynamics result is informative

The settle was operational, not dead code:

- `99.0909090909091%` of rows finished on the input-anchored cap;
- the trajectory recorded `3597` constrained-stationary step events;
- maximum energy increase was `7.247535904753022e-13`, within the registered
  `1e-12` tolerance;
- maximum Armijo residual was `7.249270628228999e-13`;
- maximum cap error was `1.7763568394002505e-15`;
- the primary and independently orchestrated optimizers were bit-identical and
  had zero prediction mismatches.

The state moved and the displayed scalar descended, but the LDA decision did
not change. On this population, recurrence consumed computation without adding
decision value. Valid energy descent remains evidence about the scalar, not a
reason to retain a recurrent mechanism.

## 6. Invalid-attempt history

The first execution was `HARNESS_INVALID`, not a scientific verdict. A manually
expanded numerical path in the independent optimizer accumulated a
`5.102135665069341e-07` cap-surface discrepancy, although all predictions and
cap norms agreed. The exact receipts and result are preserved under
`results/invalid-attempt-01/`.

The implementation-only repair shared canonical CPU float64 primitives for
softmax, row norm, displayed-energy evaluation, and ball projection while
retaining separately structured optimizer/Armijo control flow. No scientific
constant, source, split, model, seed, selector, bar, or command changed. A new
preflight and registration preceded the valid execution.

## 7. Evidence

| Artifact | SHA-256 or scientific digest |
|---|---|
| Preflight receipt | `9a11c42296c84f29b70b331c2b5d47bff79729e64d7a091a25bba3040c77e1b0` |
| Development registration | `eabe54ed2f473ab91307f8a9f8ccc89f08e84e3b2258e476af0f0a1564cd8a4d` |
| Registration scientific digest | `898f019f571ca1c3e8a97eb0bf8d54a7ee2b2bec35ca68740cf4d4f81ad9ade6` |
| Raw development rows | `310bd7546285ffc569e55f43e6c50ad37273bbe7b238e6a5224993694902735a` |
| Development result file | `5fe98d843c75fcd41aba0c8e9f2462313a1b1fca6bea8712d2b0fedb326cebd2` |
| Development result digest | `4caeb80eb13ab5cf9136f8a6f6a14aa551e81e917f07f4683190f14e30884b27` |

Independent raw-row recomputation reproduced all six arm macros, the paired
alignment interval, the raw-row hash, and the result digest exactly.

## 8. Claim boundary

What survives is a conventional, closed-form supervised preconditioner over a
frozen representation. The energy/mean-shift recurrence stops on development
and was not authorized for test. This result says nothing about language-model
perplexity, factual editing, universal explainability, compliance, production
readiness, or superiority over attention/softmax; the displayed mixture energy
is itself a conventional Gaussian-mixture / modern-Hopfield-style object.

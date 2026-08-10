# S1 Verdict — Certified Semantic Attractor Routing

**Date:** 2026-08-10

**Experiment:** `hlm5-s1-certified-semantic-attractor-v1`

**Formal verdict:** `FAIL`

**Validity:** clean; all nine registered validity gates passed

**Scope:** HLM model research only; no KB, Flow, Ingesto, training, scaling,
deployment, compliance, or universal-explainability claim

## 1. Conclusion

The fixed degree-five polynomial settle does not add useful semantic-address
recovery over the registered static cache on ParaRel. Static cosine routing was
already perfect on the untouched population: macro accuracy was `1.000000`.
Eight certified attractor steps reduced macro accuracy to
`0.6194226551956815`, a difference of `-0.3805773448043185` with a registered
relation-cluster interval of
`[-0.498070860745614, -0.2600297844973009]`.

The result is not a failed certificate or implementation mismatch. Every
accepted transition satisfied the displayed energy inequality, the independent
matched implementation agreed exactly, and all validity controls passed. The
settle reliably optimized its stated energy, but that energy was not aligned
with correct held-out address routing.

This closes this fixed certified-settle direction on the spent ParaRel
population. The preregistered stop rule forbids tuning degree, step count,
Armijo constants, centering, whitening, or encoder choice on these outcomes.

## 2. Registered execution

The run used the exact pre-outcome commit and registered command:

```text
git commit: 089876491a48d3a872a9240661e1b96a7ca131f7
command:    python -B run_semantic_attractor_gate.py --run-registered
device:     CPU
dtype:      torch.float64 after encoding
```

The population contained 38 relations, 304 stored canonical addresses, 2,320
supported held-out paraphrases, and 2,320 matched off-support queries. Inputs
were generated from ParaRel commit
`cb5554678457beb5ac163d888f1ce8cf174b3f0b` and encoded with frozen
`sentence-transformers/all-MiniLM-L6-v2` revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

## 3. Binding scientific bars

| Registered bar | Required | Measured | Passed |
|---|---:|---:|:---:|
| Candidate minus static macro accuracy | `>= +0.050` | `-0.3805773448043185` | No |
| Relation-cluster interval lower endpoint | `> 0.000` | `-0.498070860745614` | No |
| Candidate AUROC minus static AUROC | `>= -0.010` | `-0.33558263971462554` | No |
| Maximum predicted-relation share | `<= 0.150` | `0.08448275862068966` | Yes |

Supported/off-support AUROC fell from `0.8927191401605232` for static routing to
`0.5571365004458977` after settling. The candidate did not collapse globally
into one relation, but it substantially weakened both exact routing and
off-support separation.

## 4. Validity gates

All registered validity gates passed:

| Gate | Result |
|---|:---:|
| Registration and command verified | PASS |
| Population exact | PASS |
| Finite and unit-normalized states | PASS |
| Energy non-increasing | PASS |
| Armijo bound satisfied | PASS |
| Zero-step identity bit-exact | PASS |
| Static cosine and degree-five static reader equivalent | PASS |
| Independent matched settle | PASS |
| Shuffled-target null below 5% | PASS |

The maximum independent-state discrepancy was exactly `0.0`. The largest
reported energy increase and Armijo residual were both
`7.771561172376096e-16`, within the registered `1e-12` tolerance. Maximum final
unit-norm error was `4.440892098500626e-16`. Shuffled-target macro accuracy was
`0.013157894736842105`.

The encoder loaded and completed on the pinned CPU environment. The environment
did emit a SciPy/scikit-learn compatibility warning because local NumPy was
`2.4.6` while those optional packages declared a lower supported range. No
SciPy/scikit-learn computation entered the registered metric path, and the
frozen encoder, deterministic replay digests, finite checks, and independent
controls all passed. This remains environment debt for any replication, not a
validity failure in this run.

## 5. What the raw rows show

The static reader selected the correct stored address on all `2,320` supported
queries. The candidate changed `843` predictions (`36.336206896551726%`), and
all `843` changes were errors:

- `430` moved to another stored subject within the correct relation;
- `413` moved to a stored address in a different relation.

Despite those errors, median confidence changed by
`+0.034982984189643795`, while median final-minus-initial energy was
`-0.10703279749415148`. The dynamics therefore became more confident and more
energy-consistent while becoming less task-correct.

The largest wrong destination clusters were `P937` (192 errors), `P103` (189),
`P264` (109), and `P530` (56). This concentration remained below the global
15% collapse bar. The worst source relation was `P37`, whose accuracy moved
from `1.00` to `0.00`; six more source relations fell from `1.00` to `0.125`.

## 6. Interpretation

The strongest supported conclusion is narrow:

> A valid monotone energy certificate proves that the registered optimizer
> follows the registered scalar. It does not prove that the scalar represents
> semantic correctness, task reasoning, or a faithful explanation of the
> output.

The candidate's exact agreement with an independently coded matched settle
also rules out an implementation-exclusivity claim. The harmful movement is a
property of the displayed objective and frozen address geometry under this
protocol, not evidence that one code path malfunctioned.

The key matrix had numerical rank `304`, coherence
`0.9107021846830912`, and nonzero-spectrum condition number
`194.18675833726323`. A plausible mechanism is that the higher-order sum rewards
movement toward dense, mutually coherent key regions rather than preserving the
identity of the nearest intended slot. That mechanism is an interpretation,
not a separately identified causal result.

The perfect static baseline also means this population had no positive-accuracy
headroom for the registered `+0.050` proposition. That does not invalidate the
untouched test; it reveals that the frozen semantic encoder plus static cache
already solved this particular address assay. It prevents any claim that the
same settle is universally harmful outside this assay.

## 7. What survives and what stops

What survives:

- frozen semantic addresses plus static cosine retrieval on this ParaRel assay;
- the governed external-memory lifecycle and its independently testable
  provenance, rollback, and policy controls;
- the energy/replay machinery as an honest certificate of the computation it
  actually binds;
- the discipline of pairing every certificate with a held-out task metric and
  a matched conventional control.

What stops:

- this degree-five, eight-step certified semantic settle on ParaRel;
- any claim that energy descent by itself explains or validates a semantic
  decision;
- post-outcome parameter, geometry, or encoder tuning on the spent population;
- any scaling continuation from this result.

A future dynamics experiment is justified only by a new, preregistered task in
which the static semantic address is measurably corrupted or ambiguous and the
energy is designed against that task loss. It must use untouched data and retain
the static cache, zero-step identity, shuffled-target, and independently matched
optimizer controls.

## 8. Evidence and digests

| Artifact | SHA-256 or registered digest |
|---|---|
| Derived ParaRel manifest | `84ef9b5ddbb3e12febd41b71096d65c0c581fd2df7ccc7f4681c7d6d299307af` |
| Preflight receipt file | `6aa5bda9373aa2f62e8f14ffa53be04fdf4d4ed075a0b0e2f4b5f61ce0f231f7` |
| Registration file | `0c9f91bd792505f5490e263ca484c41f3a652d37a28c71f81d240beabaa55df9` |
| Registration scientific digest | `d955a0f4ff559058cae69f995d9dbf73e2d2b8aedadda5aa8f787e5563d3b4b7` |
| Raw rows file | `f61af51d9520fb1b0972e7b4cc656383c93f897d2c9dc92e5db4c0650e124651` |
| Result scientific digest | `1601f51da556a078f3709946499444d96ad7a51e67b924e16c109265517c6579` |
| Result JSON file | `5854ba939b2e4356288c0fcd931fbffaf9cec83c82217ab893f2462abec6ff0e` |

The raw row file contains exactly `4,640` JSON lines: `2,320` supported and
`2,320` off-support rows. The result's embedded raw-row hash and scientific
digest recompute exactly from the committed evidence.

## 9. Claim boundary

This result does not establish semantic understanding, factual correctness,
language-model improvement, production readiness, EU AI Act compliance,
universal explainability, robustness beyond ParaRel, or an HLM-specific
advantage. It is one valid negative result for one frozen energy, state update,
encoder, population, and set of registered controls.

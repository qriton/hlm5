# HLM5 Multi-Support Kernel-Equivalence — Verdict

**Date:** 2026-08-10  
**Status:** `NARROW_MULTI_SUPPORT_CACHE_EQUIVALENT`  
**Scope:** deterministic CPU assay of the frozen editable-memory operator; no
checkpoint, language-model forward, training, remote compute, or scale

## Decision

HLM5 has a real but narrow multi-support function: two equally or nearly
equally addressed values reconstruct a target that hard top-1 and either value
alone cannot produce. The function is not robust to modest support imbalance,
is weaker than the simpler cosine-soft cache on the locked skew ladder, and is
bit-exactly reproduced by an independently coded degree-5 kernel cache.

Keep this as a bounded operator/reference result. Do not tune degree,
temperature, threshold, or mixture balance; do not use it to resume HLM5
pretraining or claim an HLM-specific retrieval advantage.

## Registered result

Across five deterministic permutation seeds, 16 share pairs, and five fixed mixture ratios, every create,
reactivate, and pair-edit query had exactly two supports. The aggregate task
contained 400 queries per phase.

The seeds permute an orthogonal synthetic construction to catch indexing or
pairing errors; they are not independent samples from a natural-data
population and license no confidence interval or generalization claim.

| Mix ratio | HLM5 degree-5 | Matched degree-5 cache | Cosine-soft cache | Hard top-1 |
|---:|---:|---:|---:|---:|
| 1.00 | 100% | 100% | 100% | 0% |
| 0.95 | 100% | 100% | 100% | 0% |
| 0.90 | 0% | 0% | 100% | 0% |
| 0.80 | 0% | 0% | 0% | 0% |
| 0.70 | 0% | 0% | 0% | 0% |
| **Overall** | **40%** | **40%** | **60%** | **0%** |

The same 40% degree-5 profile reproduced exactly after reversible
deactivate/reactivate and after rotating both stored shares through the pair-edit
lifecycle. Removing one live share reduced HLM5 accuracy to 0%; the
shuffled-value null was 0/400; and all 80 off-support queries returned bit-exact
zero in every reader. Those controls establish that the balanced result really
uses both stored values.

All validity gates passed:

- support cardinalities were exactly 2/1/0 in the registered phases;
- every key and value permutation was bijective and all expected rows existed;
- every reported output, weight, and gain was finite;
- off-support was bit-exact zero; and
- HLM5 and the independent matched degree-5 cache were bit-exact for scores,
  support masks, weights, gains, retrieved values, outputs, and predictions in
  every phase.

## Mathematical interpretation

This reader is an explicit kernel smoother:

```text
s_i(q) = alpha_i * relu(cos(q, k_i))^5
R(q)   = sum_i softmax(s_i(q)/T over support) * v_i.
```

A cache that stores the same keys, values, and strengths and evaluates this
equation is the same operator for one support, two supports, or many. The
implementation-level bit equality confirms the structural proposition; a task
cannot distinguish these two readers without changing the mechanism or the
matched-control definition.

The skew result also exposes the degree-five price. At ratio 0.90 both shares
remain inside the registered support, but degree-five weighting is already too
concentrated for the sum decoder; the less concentrated cosine-soft cache still
succeeds. Multi-support aggregation is therefore not a hidden advantage of the
degree-five kernel in this measured regime.

## What remains genuinely useful

- The conservative-attention/dissipative-memory split remains a low-tax HLM5
  architecture pattern.
- The exact canonical-address lifecycle remains useful for deterministic,
  governed create/edit/deactivate/reactivate/delete operations.
- The reader can combine multiple balanced synthetic supports, and that behavior survives
  reversible lifecycle operations and pair edits.
- An explicit cache is the simpler truthful implementation of the measured
  primitive; cosine-soft aggregation is more robust on this fixed skew ladder.

## Stop rule

Close the degree-five multi-support advantage lane. Do not run a temperature,
degree, threshold, decoder-margin, or query-balance sweep on these tasks. Any
future HLM-specific model claim must introduce a mechanism not reducible to an
explicit kernel cache—such as independently validated state dynamics—and beat
that cache under necessity, readability, stability, and matched-cost controls.

## Provenance

- Preregistration:
  `docs/multi-support-kernel-equivalence-prereg-2026-08-10.md`
  (`sha256 8079c107124d9b58dee501b8d6a6175a94eb94908aa0023224fe7c50d94d7b10`)
- Initial registration:
  `baselines/results/multi_support_kernel_equivalence_registration.json`
  (`sha256 acebfbd8d457eedb7892b6540a5352fba7d17863a80a53ece1231ca4eadeb9e7`)
- Execution registration:
  `baselines/results/multi_support_kernel_equivalence_execution_registration.json`
  (`sha256 07d438d54714c1e3ecb8237f1d2e6b98654f830f441e708631786622f812b054`)
- Runner:
  `demos/multi_support_kernel_equivalence_reference.py`
  (`sha256 f39d7952695a0e40557811a7730bf35017c607094843abe9686a3ac77f7b9b71`)
- Tests:
  `tests/test_multi_support_kernel_equivalence_reference.py`
  (`sha256 c0636998ce9ae9f0cc634bf50c3d3fe7e3d1b7065914e2dea5f72ae944056214`)
- Result:
  `baselines/results/multi_support_kernel_equivalence_reference.json`
  (`raw sha256 f7928cd07b879b290a0bb9fbe623fe32c7b9816b487afcf7d0650f0cb856b79b`;
  scientific sha256
  `07a410e232745067feaee9862804e6c173c3a50b0d00de9aa079a2044a1829c6`)

No artifact in this package authorizes training, G4 resumption, a K ladder,
semantic-routing claims, or KB/Flow/Ingesto integration.

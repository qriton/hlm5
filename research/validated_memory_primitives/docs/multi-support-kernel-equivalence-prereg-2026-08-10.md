# HLM5 Multi-Support Kernel-Equivalence Preregistration

**Date:** 2026-08-10  
**Scope:** one deterministic CPU assay of the frozen editable-memory operator;
no checkpoint, language-model forward, training, gradient, remote compute,
threshold/temperature/degree sweep, or scale authorization

## Question

The exact-address lifecycle established useful create/edit/deactivate/restore/
reactivate/delete behavior, but every accepted query had one live support and
the degree-5 reader was exactly matched by hard top-1 cache retrieval. This
follow-up asks the smallest question that can make multi-support meaningful:

> Can two stored values jointly reconstruct a target that neither value can
> decode alone, does that behavior survive controlled query imbalance and
> lifecycle operations, and is it anything more than explicit kernel-cache
> aggregation?

The matched control is not only the prior hard top-1 cosine cache. The assay
must also independently implement the exact positive degree-5 kernel, support
mask, temperature, gain, and value sum used by `EditableHLM5Memory`. A win over
top-1 that remains exactly reproducible by this cache is multi-support
functionality, not an HLM-specific retrieval advantage.

## Structural equivalence proposition

For active slots `A`, define

```text
s_i(q) = alpha_i * relu(cos(q, k_i))^d
S(q)   = {i in A : s_i(q) >= theta}
w_i(q) = softmax(s_i(q) / T over S(q))
R(q)   = sum_i w_i(q) v_i
g(q)   = max_i s_i(q)
```

with `R(q)=0` and `g(q)=0` when `S(q)` is empty. The support-masked HLM5 reader
returns `R(q)` and the registered integration used by the address-conditioning
work applies `g(q)R(q)` to a zero baseline state. An explicit cache holding the
same `(k_i, v_i, alpha_i)` and evaluating the same equations is therefore
identical for every support cardinality. This follows directly from the frozen
implementation; the assay validates the implementation-level equality and
tests whether a simpler cosine-soft cache is already sufficient on the task.

## Pinned lineage

| Artifact | SHA-256 |
|---|---|
| `hlm5_memory.py` | `0210733ab1160ac073d44e5a9d7967bb3b985fc323a07efde925694ae56c346a` |
| `demos/exact_address_lifecycle_reference.py` | `4bad382d97e74b88401d5b478191926f3027007a4a8ec99cf3337f51c9669e8a` |
| `tests/test_exact_address_lifecycle_reference.py` | `a46472c7be72ed315d3f5c6b0c1924c04378ec8fe0c5553c2fffebb3567184fa` |
| `baselines/results/exact_address_lifecycle_reference.json` | `f61ceee8f9ec7233d0cf1ec0e256c034c7a5ab075059846abe01048e54ba5711` |

The pinned reference must retain status `CACHE_EQUIVALENT_REFERENCE` and
scientific SHA-256
`a69593454521f693c21ba7195d33e2e59dc0aafc8242cce0e886aba2554de907`.

## Two-stage registration

An initial receipt hashes this document before the new runner, tests, result,
or verdict exist. After implementation and synthetic CPU contract tests, a
second receipt locks the runner and test hashes, Python/PyTorch versions, exact
command, constants, and pinned lineage before the scored assay runs. The
runner must verify both receipts and every pinned hash before task evaluation.

## Locked construction

Use CPU float32 only with deterministic seeds:

```text
seeds       = [20260820, 20260821, 20260822, 20260823, 20260824]
dim         = 64
pairs       = 16
active slots= 32
capacity    = 40
degree      = 5
temperature = 0.10
tau_cos     = 0.55
theta       = tau_cos^degree
mix ratios  = [1.00, 0.95, 0.90, 0.80, 0.70]
```

For each seed, generate two CPU `torch.randperm(64)` vectors: the key
permutation uses that seed and the value permutation uses `seed + 100000`.
Keys are the first 32 permuted standard basis vectors, grouped into 16 ordered
pairs. Values are the first 32 value-permuted basis vectors with the same pair
grouping. Inject every `(key, value)` with strength and value scale one.

For pair `j` and ratio `r`, the query is

```text
q_j(r) = unit(k_2j + r * k_(2j+1)).
```

All five ratios must have exactly two supports at `tau_cos=0.55`; no other key
may enter support. The primary set contains `5 seeds * 16 pairs * 5 ratios =
400` queries.

The fixed decoder contains 48 rows:

1. target rows `t_j = unit(v_2j + v_(2j+1))`, `j=0..15`; and
2. all 32 individual value rows as distractor classes.

The correct label for every create query from pair `j` is target row `j`.
Either value alone scores its individual distractor above `t_j`; an adequately
balanced two-value aggregate scores `t_j` above both distractors. Thus top-1 or
one-live-share retrieval cannot pass by construction.

## Registered readers and controls

Evaluate the same keys, values, active mask, and alphas with:

1. `hlm5_degree5`: `EditableHLM5Memory` in `support_masked` mode;
2. `matched_degree5_cache`: an independent implementation of the displayed
   degree-5 equations, without calling `EditableHLM5Memory.forward`, `score`,
   or `_selected_scores`;
3. `cosine_soft_cache`: support on cosine `>= tau_cos`, softmax of cosine over
   support at the same temperature, and the degree-5 maximum score as the
   integration gain;
4. `hard_top1_cache`: the prior single-slot cosine cache with the same gate and
   degree-5 integration gain; and
5. `shuffled_value_null`: the HLM5 reader with each pair's two values rotated
   to the next pair while labels remain unrotated.

The independent degree-5 cache is the binding matched control. The cosine-soft
and top-1 arms locate whether any observed behavior needs degree-five weighting
or merely value aggregation.

## Lifecycle phases

For every seed:

1. `create`: evaluate all 80 ratio/pair queries.
2. `deactivate_one_share`: deactivate slot `2j+1` for every pair and evaluate
   the same queries; every support size must become one.
3. `reactivate`: restore every deactivated snapshot and repeat `create`.
4. `pair_edit`: rotate both values of each pair to the next pair using
   `edit_value`; the new correct target for query pair `j` is `(j+1) mod 16`.
5. `off_support`: query the 16 unused key-basis directions from positions
   32:48 of the key permutation; support must be zero and output bit-exact zero.

The shuffled null is evaluated on the create query set. No deletion/serialization
claim is added because the pinned exact-address reference already measures
those operations.

## Required reporting

For each reader, seed, phase, and ratio report query count, correct count,
accuracy, support-size histogram, output signature, and prediction signature.
For the HLM and matched degree-5 cache additionally report exact equality for
scores, support mask, weights, gain, retrieved value, output, and prediction.
Report maximum absolute output differences for the cosine-soft control.

## Validity gates

The assay is valid only if all conditions hold:

1. both receipts, all pinned hashes, exact command, runtime, constants, tool,
   and test hashes match;
2. all key and value permutations are bijections and are emitted with hashes;
3. every create/reactivate/edit/shuffled query has support size exactly two,
   every deactivate query exactly one, and every off-support query zero;
4. all expected query counts, seeds, pairs, ratios, readers, and phases exist;
5. every output, weight, gain, and decoder score is finite;
6. the HLM and independent matched degree-5 cache are bit-exact on every
   required tensor and prediction; and
7. the off-support output is bit-exact zero for every reader.

A validity miss is `INVALID`, never a scientific failure.

## Binding interpretation bars

After validity, classify in this order:

### `ROBUST_MULTI_SUPPORT_CACHE_EQUIVALENT`

All must hold:

- HLM create accuracy `>=0.80` overall and `>=0.75` at every ratio;
- HLM balanced-ratio (`r=1.00`) accuracy `>=0.95`;
- reactivate and pair-edit meet the same accuracy bars;
- hard top-1, deactivate-one-share, and shuffled-value accuracy are each
  `<=0.10` overall;
- off-support is exactly zero; and
- matched degree-5 cache equivalence passes.

### `NARROW_MULTI_SUPPORT_CACHE_EQUIVALENT`

Use only when validity, equivalence, nulls, off-support, and balanced-ratio
accuracy `>=0.95` pass, but any robust skew/lifecycle accuracy bar misses.

### `NO_MULTI_SUPPORT_FUNCTION`

Use when the valid HLM balanced-ratio accuracy is below `0.95` or a load-bearing
top-1/single-share/shuffled control exceeds `0.10`.

Also report, without changing the status, whether degree five beats the
cosine-soft cache by at least `0.10` overall accuracy. Even such a difference
would be a kernel-choice result, not an HLM-specific advantage, because the
matched degree-5 cache remains equivalent by construction.

## Stop rule and claim boundary

No outcome authorizes pretraining, fine-tuning, G4 resumption, a larger K
ladder, semantic-routing claims, or KB/Flow/Ingesto integration. A robust pass
may be packaged as a cache-equivalent multi-support reference. A narrow/no-
function result closes this degree-5 multi-support repair direction. Any future
HLM-specific advantage claim must name a mechanism not reducible to this
explicit kernel-cache equation and beat its matched implementation.

## Intended artifacts

- Initial receipt:
  `baselines/results/multi_support_kernel_equivalence_registration.json`
- Execution receipt:
  `baselines/results/multi_support_kernel_equivalence_execution_registration.json`
- Runner:
  `demos/multi_support_kernel_equivalence_reference.py`
- Tests:
  `tests/test_multi_support_kernel_equivalence_reference.py`
- Result:
  `baselines/results/multi_support_kernel_equivalence_reference.json`
- Verdict:
  `docs/multi-support-kernel-equivalence-verdict-2026-08-10.md`

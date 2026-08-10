# S1 Preregistration — Certified Semantic Attractor Routing

**Date:** 2026-08-10
**Status:** pre-outcome design; not registered until the receipt described in
Section 10 exists
**Scope:** HLM model research only; no KB, Flow, Ingesto, training, or deployment
claim

## 1. Why this experiment exists

The validated HLM5 external reader computes a positive degree-five kernel over
explicit keys. An independent matched cache reproduces it bit-for-bit. Running
that same reader on another semantic benchmark cannot establish an HLM-specific
retrieval advantage; it can only measure the address encoder.

The remaining bounded HLM question is whether actual attractor dynamics add
useful error correction. Dense associative memory was introduced as recovery
from incomplete or corrupted cues using a higher-order energy, while modern
Hopfield retrieval is equivalent to attention in its one-step form:

- Krotov and Hopfield, *Dense Associative Memory for Pattern Recognition*
  (2016): https://arxiv.org/abs/1606.01164
- Ramsauer et al., *Hopfield Networks is All You Need* (2020):
  https://arxiv.org/abs/2008.02217

This assay tests repeated polynomial energy descent explicitly. It does not use
the asymmetric learned HLM2/HLM3 update whose recorded scalar failed to be a
Lyapunov function.

## 2. Proposition

Let `K` be unit-normalized canonical semantic address vectors and `q0` a
unit-normalized held-out paraphrase vector for one stored subject–relation pair.
On the unit sphere define

```text
E(q) = -(1 / (d + 1)) * sum_i relu(k_i · q)^(d + 1),   d = 5.
```

The candidate performs Riemannian gradient descent on this displayed energy,
using a deterministic Armijo backtracking line search. The proposition is:

> On untouched ParaRel paraphrases, the certified settle improves macro
> subject–relation routing accuracy by at least 5 percentage points over static
> cosine retrieval, with a relation-cluster bootstrap lower bound above zero,
> without reducing supported-vs-off-support score AUROC by more than 0.01 or
> collapsing predictions into a dominant relation.

This is a state-dynamics proposition, not a claim that the implementation cannot
be reproduced by a conventional cache.

## 3. Fixed external inputs

### Dataset

- ParaRel repository: https://github.com/yanaiela/pararel
- commit: `cb5554678457beb5ac163d888f1ce8cf174b3f0b`
- license: MIT
- source files:
  `data/pattern_data/graphs_json/*.jsonl` and
  `data/trex_lms_vocab/*.jsonl`
- primary description: Elazar et al., *Measuring and Improving Consistency in
  Pretrained Language Models*, TACL 2021,
  https://aclanthology.org/2021.tacl-1.60/

An outcome-blind census found 39 repository relations and 329 patterns. The
sole one-pattern relation, `P1001`, is ineligible because it has no held-out
paraphrase. The locked population is therefore the paper-aligned 38 relations
and 328 patterns. Every eligible relation has at least 53 fact rows.

### Encoder

- `sentence-transformers/all-MiniLM-L6-v2`
- Hugging Face revision:
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- CPU inference, evaluation mode, normalized embeddings
- the registration records package versions and the digest of every local model
  file used by the loader

No encoder fine-tuning, projection fitting, whitening, centering, relation
classifier, entity extractor, or outcome-dependent threshold is allowed.

## 4. Deterministic population construction

For each eligible relation, in ascending relation-ID order:

1. Preserve pattern-file order. Pattern zero is the canonical store template;
   every later pattern is a held-out query template.
2. Sort fact rows by `(uuid, sub_label, obj_label)` and take the first eight
   rows with unique `sub_label`.
3. Render an address by replacing `[X]` with `sub_label` and `[Y]` with the
   literal word `something`.
4. Store one canonical address per selected subject–relation pair. The object
   label is retained only for provenance; it is not present in the address.
5. Render every held-out template for the same subject. Its correct slot is the
   canonical address for that subject–relation pair.

This yields exactly 304 stored addresses. The expected supported-query count is
computed by the manifest builder and then frozen; a mismatch at registered
execution is `HARNESS_INVALID`.

For each supported query, construct one off-support query by replacing its
subject with the selected subject from the next eligible relation, cycling
forward until that subject–relation pair is absent from the store. The pattern,
punctuation, and relation stay unchanged. Off-support queries have no correct
slot and are used only for score-separation diagnostics.

The derived manifest stores every relation ID, source-file SHA-256, selected
UUID, canonical pattern, held-out pattern, rendered query, target slot, and
off-support construction. Manifest generation must not load the encoder or
compute routing outcomes.

## 5. Candidate dynamics

All scientific arithmetic after encoding is CPU `torch.float64`.

For unit state `q` and unit key rows `K`, define

```text
h(q)       = sum_i relu(k_i · q)^d k_i
direction  = h(q) - q * (q · h(q))
proposal   = unit(q + eta * direction)
```

`direction` is the negative Riemannian gradient of `E` on the unit sphere.
Each of eight outer steps starts with `eta = 1.0`. Failed Armijo proposals halve
`eta`, with at most 24 halvings. The acceptance condition is

```text
E(proposal) <= E(q) - 1e-4 * eta * ||direction||^2 + 1e-12.
```

If `||direction|| <= 1e-12`, the state is copied unchanged. Failure to find an
accepted finite step is `HARNESS_INVALID`, not a scientific FAIL. There is no
early stopping, degree sweep, step-count sweep, temperature, learned gain, or
post-result geometry change.

The final predicted slot is `argmax_i(k_i · q8)`. The routing confidence is the
maximum final cosine.

## 6. Controls

1. **Static cosine cache (primary):** `argmax_i(k_i · q0)`.
2. **Static degree-five reader:**
   `argmax_i relu(k_i · q0)^5`. It must select the same slot as static cosine
   whenever the maximum cosine is positive; discrepancies are reported and a
   supported-query discrepancy is `HARNESS_INVALID`.
3. **Independent matched settle:** a separately coded implementation of the
   same displayed energy and line search. Candidate and matched settle must
   agree on final slots and remain within `1e-10` maximum absolute state error.
   This explicitly prevents an HLM-implementation exclusivity claim.
4. **Shuffled-target null:** one fixed CPU permutation of target slots, generated
   with seed `20260810`. Its macro accuracy must be below 5%; otherwise the
   evaluation mapping is invalid.
5. **No-settle identity:** zero outer steps must be bit-identical to the input
   embeddings and reproduce static-cosine predictions.

## 7. Metrics

The unit of dependence is the relation, not the individual paraphrase.

Primary:

- macro routing accuracy across the 38 relation-level accuracies;
- paired macro difference: candidate minus static cosine;
- deterministic 10,000-resample relation-cluster bootstrap interval, seed
  `20260810`, using the float64 arithmetic mean for both the point statistic and
  every resample; percentile endpoints use NumPy linear interpolation.

Safety/selectivity:

- supported-vs-off-support AUROC from maximum cosine, computed with average
  ranks for ties;
- maximum share of supported predictions assigned to any one relation;
- per-relation candidate/static accuracy and prediction-share table.

Certificate and geometry:

- energy at `q0` and after every outer step;
- accepted step sizes, gradient norms, and Armijo residuals;
- fraction of transitions non-increasing within `1e-12`;
- exact replay digest of inputs, trajectories, predictions, and metrics;
- key coherence, singular values, numerical rank, and condition number on the
  nonzero singular spectrum.

No language-model efficacy, perplexity, token generation, or edited-output
metric is part of this assay.

## 8. Binding verdict

Validity gates, all required:

1. source commit, file hashes, derived manifest, encoder revision, environment,
   tool, tests, registration, and command match the registration receipt;
2. population is 38 relations, 304 unique stored addresses, and the registered
   supported/off-support counts;
3. all embeddings and dynamics values are finite and normalized within `1e-10`;
4. every accepted transition satisfies the registered energy inequality;
5. no line search exhausts its 24 halvings;
6. no-settle identity and supported static cosine/degree-five equivalence pass;
7. independent matched settle passes the registered state and slot tolerances;
8. shuffled-target macro accuracy is below 5%.

If a validity gate fails, the verdict is `HARNESS_INVALID` even if scientific
bars appear favorable.

Scientific PASS, all required:

1. candidate minus static macro accuracy is at least `+0.050`;
2. the 95% relation-cluster bootstrap lower endpoint is strictly greater than
   `0.000`;
3. candidate supported/off-support AUROC is at least
   `static AUROC - 0.010`;
4. no single relation receives more than 15% of supported predictions.

Otherwise the verdict is `FAIL`. Infrastructure interruption before all rows
exist is `INCONCLUSIVE` only when no validity failure is already known.

## 9. Stop and continuation rules

- `FAIL`: conclude that this fixed degree-five certified settle does not add
  useful semantic-address recovery over a static cache on ParaRel. Do not tune
  degree, steps, Armijo constants, centering, whitening, or encoder on the spent
  population.
- `PASS`: the only licensed continuation is one preregistered replication on a
  second encoder or a second untouched paraphrase source. No LM training or
  scaling follows directly.
- Any PASS supports only “certified attractor dynamics improved routing on this
  frozen address assay.” Because the matched settle is required to agree, it
  cannot support a proprietary-HLM or cache-superiority claim.

## 10. Registration and execution order

Before any real encoder outcome is computed:

1. unit tests pass on analytic and synthetic tensors;
2. synthetic preflight verifies energy descent, failure precedence, estimator
   conventions, null behavior, and independent-control agreement;
3. the outcome-blind ParaRel manifest and source/model file hashes are frozen;
4. the protocol, implementation, tests, and manifest are committed on a clean
   research branch; generated receipts/results are the only allowed untracked
   paths after that commit;
5. a registration receipt hashes that Git commit, this protocol,
   implementation, tests, manifest, exact command,
   Python/PyTorch/NumPy/sentence-transformers versions, OS, device, dtype,
   seeds, and every scientific constant;
6. registered mode refuses every CLI override of a scientific constant and
   refuses to run unless all receipt hashes match;
7. the single registered command writes raw rows before aggregate metrics and a
   verdict. A verdict document is written before any follow-up design.

Preflight may inspect shapes, counts, hashes, and synthetic controls. It may not
encode or score a real ParaRel address before the receipt exists.

## 11. Explicit non-claims

This assay cannot establish semantic understanding, factual correctness,
language-model improvement, EU AI Act compliance, universal explainability,
HLM-specific implementation advantage, production readiness, robustness beyond
ParaRel, or authorization to train or scale a model.

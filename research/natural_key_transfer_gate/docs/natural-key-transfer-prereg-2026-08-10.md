# NKT-1 Preregistration — Cross-Domain HLM5 Natural-Key Transfer

**Date:** 2026-08-10

**Status:** pre-outcome design. No MASSIVE utterance may be encoded by the
HLM5 trunk and no routing outcome may be computed before the development
registration receipt exists.

**Scope:** HLM model research only. No KB, Flow, Ingesto, deployment,
compliance, language-model scaling, recurrence, or polynomial-Hopfield
advantage claim.

## 1. Question

The Banking77 result established that a label-fitted Fisher/LDA metric can
improve a frozen semantic reader, while its recurrent energy step changed no
decision. HLM5 itself uses a narrower operator at its inference-time memory
hook:

```text
q = (h - key_mean) @ key_transform
score(q, k) = relu(cos(q, k)) ** 5
```

The unresolved question is whether task-aligned conditioning transfers into
that actual HLM5 hidden-key geometry without fitting a label-rich metric on the
target task.

NKT-1 learns one full-rank within-class Fisher preconditioner on source domains,
then freezes it. The target memory receives only three address examples per
previously unseen intent. Held-out target utterances must route through the
same 69 individual HLM5 degree-five address slots. There is no centroid reader,
neural projection, recurrence, target-domain covariance fit, or threshold
sweep.

## 2. Fixed source and untouched status

- dataset: MASSIVE 1.1, English `en-US`
- official archive:
  `https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz`
- archive SHA-256:
  `4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577`
- `1.1/data/en-US.jsonl` SHA-256:
  `c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e`
- license: CC-BY-4.0; extracted license SHA-256:
  `c2e6ea015269147de02117ebdd91f30ef09831251f5345fa8365273b1db1d435`
- Hugging Face integration snapshot:
  `AmazonScience/massive@ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740`
- primary publication: FitzGerald et al., ACL 2023,
  `https://aclanthology.org/2023.acl-long.235/`

The archive contains 11,514 train, 2,033 development, and 2,974 test rows,
across 18 scenarios and 60 intents. An outcome-blind local-estate search found
no prior HWU64/MASSIVE experiment or result in the tracked HLM2, HLM5, or
HLM-Demos research record. Generic uses of the English word "massive" do not
constitute dataset exposure.

The official test file may be downloaded and hashed, but its utterances must
not be tokenized, encoded, or scored before a passing development verdict and
a separate test registration.

## 3. Deterministic population construction

Normalize text with Unicode NFKC, case folding, trim, and whitespace collapse.
Within training:

1. remove every row in a normalized-text group containing conflicting intents;
2. for a same-intent duplicate group, retain the lowest source index;
3. retain 11,464 clean rows (eight conflict rows and 42 duplicate rows removed).

For development, first exclude every normalized text present in clean training,
then retain the lowest source index in each remaining same-intent duplicate
group. The primary development population is 2,020 rows. The corresponding
outcome-blind test rule excludes normalized overlap with clean train or primary
development and deduplicates the remainder; its census is 2,944 rows, but it
remains unencoded.

Rank scenario names by
`SHA256("hlm5-natural-key-transfer-v1" + NUL + scenario)`. The six lowest
hashes are the fixed target domains:

```text
lists, cooking, transport, iot, takeaway, recommendation
```

They contain 23 target intents. The other 12 domains and 37 intents are the
source population. This partition is fixed by names, not model outcomes.

For each target intent, sort clean training rows by
`SHA256("nkt1-support-v1" + NUL + intent + NUL + normalized_text + NUL + id)`,
then source index, and take exactly the first three as address slots. The
memory therefore contains 69 ordered slots. No remaining target-domain train
row may enter a transform, calibration, or threshold fit.

The primary development population is the 494 cleaned official-development
rows in the six target domains, covering all 23 target intents. Source-domain
development rows are an off-support diagnostic population only. If test is
opened, its primary target population has 705 rows and 22 observed target
intents; `cooking_query` has no primary test row and is not imputed.

## 4. Frozen HLM5 representation

- checkpoint:
  `D:/HLM2/artifacts/models/demo/hlm5-136m-fineweb-g2-2026-06-10/model.pt`
- checkpoint SHA-256:
  `2addfa88808be6847e29f70c9182093c15ff420c168f9dd8b41b72c4757a8eda`
- tokenizer:
  `D:/HLM2/artifacts/models/demo/hlm5-136m-fineweb-g2-2026-06-10/tokenizer.json`
- tokenizer SHA-256:
  `15993635191a1c5f1a5dc7aeaacbdf9a44a45d90abef954fc77b686f4fbbe588`
- architecture: `HLM5LM`, 768 dimensions, 12 layers, 12 heads, context 1024,
  65,536 tied-vocabulary rows, no trained memory layer
- model source: the immutable `research/validated_memory_primitives` snapshot
  at the registered Git commit
- raw utterance only; no prompt, label name, slot annotation, or target text
  augmentation
- remove one tokenizer-added leading BOS and trailing EOS when present;
  empty or over-context inputs make the harness invalid
- extract the final-token output of `model.hidden()` in evaluation/inference
  mode on the local GPU
- cached hiddens are CPU float32; all metric fitting, scoring, and statistics
  are CPU float64

Encoding batches are grouped by exact token length, so padding cannot change a
last-token state. A fixed synthetic anchor is the only trunk input permitted
during preflight.

## 5. Frozen key operators

Let source-fit hiddens be `x_i`, source intent labels be `y_i`, their global
mean be `mu`, and intent means be `mu_c`. Fix shrinkage `alpha = 0.05`, carried
unchanged from the selected Banking77 development setting. No grid is run.

Define global and within-class covariance with population denominators:

```text
S_global = (1/n) sum_i (x_i - mu)(x_i - mu)^T
S_within = (1/n) sum_i (x_i - mu_yi)(x_i - mu_yi)^T
S_alpha(S) = 0.95 S + 0.05 trace(S)/D I
B(S) = S_alpha(S)^(-1/2)
```

Use a symmetric float64 eigendecomposition, clamp only at the positive
shrinkage floor already present in `S_alpha`, and canonicalize eigenvector
signs for hashing. Each arm applies the following transform to both address
and query hiddens before unit normalization:

1. `raw`: `x`
2. `centered`: `x - mu`
3. `blind_zca`: `(x - mu) @ B(S_global)`
4. `shuffled_within`: `(x - mu) @ B(S_shuffled)`
5. `source_fisher`: `(x - mu) @ B(S_within)`

For the shuffled null, order source rows by source index and permute the
original label vector by sorting row indices on
`SHA256("nkt1-shuffle-v1" + NUL + id + NUL + source_index)`. This preserves
the exact label multiset without using a library-dependent PRNG. Recompute
class means and within covariance under those shuffled labels.

The candidate is `source_fisher`. `blind_zca` matches its source rows,
centering, shrinkage, rank, and arithmetic but removes label alignment.
`shuffled_within` additionally tests whether any label partition would do.

## 6. Exact HLM5 address reader

All 69 address examples remain separate slots. For normalized query `q` and
slot `k_r`, compute exactly

```text
score_r = relu(q dot k_r) ** 5
```

with unit slot strength. Select the first slot at the maximum in registered
slot order and emit its target intent. An all-zero score vector is an
abstention and counts as incorrect. A matched cosine cache must return the
same slot whenever the maximum cosine is positive; this is a validity check
and an explicit acknowledgement that this reader is cache-equivalent.

No HLM5 value write, logit boost, gate threshold, energy step, soft retrieval,
temperature, recurrence, or decoder score is part of NKT-1. The fixed deployed
threshold `0.95 ** 5` is reported as an admission diagnostic only and is not
tuned or used to redefine routing accuracy.

## 7. Metrics and estimators

Primary accuracy is the float64 arithmetic mean of the 23 target-intent
accuracies. Also report row accuracy, each intent value, improved/tied/worsened
intent counts, changed-row transitions, abstentions, maximum predicted-intent
share, and the paired true-slot-versus-best-false cosine margin.

For each comparison, use a deterministic 10,000-resample paired intent
bootstrap with seed `20260810`. The point and every resample use the same
float64 arithmetic mean. Use NumPy linear percentile interpolation for the
2.5th and 97.5th percentiles. The interval is descriptive stability over the
fixed development intents, not independent frequentist coverage after outcome
access.

Report the 69-by-69 support Gram minimum eigenvalue, condition number,
within-intent cosine, between-intent cosine, and source-fit covariance spectra.
Using cleaned source-development rows as off-support, report supported versus
off-support AUROC from the maximum degree-five score and the score quantiles.

## 8. Validity gates

All gates are binding:

1. dataset, license, checkpoint, tokenizer, Git, implementation, test,
   preflight, command, and registration hashes match;
2. exact duplicate/overlap exclusions, domain hashes, 37/23 source/target
   intent split, three support rows per target intent, 69 ordered slots, and
   494-row/23-intent development population match the manifest;
3. no test utterance is tokenized, encoded, or scored;
4. the model loads strictly, is in evaluation mode, and all hiddens and
   scientific values are finite;
5. source fitting receives no target-domain row, and only the 69 support rows
   provide target intent labels to the address reader;
6. every shrunk covariance is positive definite, each inverse root is
   symmetric, and `max_abs(B @ S_alpha @ B - I) <= 1e-8`;
7. the shuffled label multiset equals the source label multiset and its
   serialized assignment hash matches registration;
8. the degree-five reader and matched cosine reader agree on every positive
   maximum; all-zero rows are consistently marked abstentions;
9. an independently structured NumPy scorer reproduces every prediction and
   all arm score matrices within `1e-10` maximum absolute error;
10. the fixed-threshold diagnostic never changes any primary prediction or
    denominator.

Any failure is `HARNESS_INVALID`. An infrastructure interruption is
`INCONCLUSIVE` only when no invalidity is known.

## 9. Development verdict

`PASS_TO_TEST` requires all validity gates and every scientific bar:

1. `source_fisher - raw` macro accuracy `>= +0.020`;
2. its paired-intent interval lower endpoint `> 0.000`;
3. `source_fisher - blind_zca` macro accuracy `>= +0.010`;
4. its paired-intent interval lower endpoint `> 0.000`;
5. `source_fisher - shuffled_within` macro accuracy `>= +0.010`;
6. its paired-intent interval lower endpoint `> 0.000`;
7. candidate maximum predicted-intent share `<= 0.100`;
8. candidate supported/off-support AUROC is no more than `0.010` below raw.

Otherwise the verdict is `STOP_BEFORE_TEST`. No aggregate improvement can
override a failed binding bar.

## 10. Test boundary and stop rules

A passing development result permits one separate, immutable test
registration. The same train cleanup, domain partition, support slots, source
mean, source transforms, arm definitions, and bars are reused. Test macro and
bootstrap operate on its 22 observed target intents. Development cannot select
a new operator, shrinkage, support count, domain split, threshold, pooling
rule, model layer, checkpoint, prompt, or statistic.

If development fails, preserve the artifacts and keep test unencoded. Do not
tune MASSIVE. A future attempt needs a new mechanism and a new untouched
population, not a variation of this split.

## 11. Exact claim boundary

A pass would establish only that a source-domain within-class Fisher
preconditioner transfers to three-shot, cross-domain intent routing in the
frozen HLM5-136M final hidden space, under this MASSIVE partition, and beats
its registered raw, blind-whitening, and shuffled-label controls.

It would not establish a Hopfield or polynomial advantage: the degree-five
reader is rank-equivalent to a matched cosine cache. It would not establish a
working production gate, factual editing, semantic understanding, universal
transfer, robustness, explainability, EU AI Act compliance, safety, or a
reason to train or scale another HLM.

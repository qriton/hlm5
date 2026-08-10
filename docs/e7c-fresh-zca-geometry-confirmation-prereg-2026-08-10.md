# E7c fresh-pool ZCA geometry confirmation

Date: 2026-08-10

Status: preregistration; no E7c fresh-pool model outcome has been computed

## Decision question

Does the single fixed direction selected on spent E7b evidence—ZCA
half-whitening of the raw target row—transfer to a third disjoint 1,200-token
pool on the pinned frozen SmolLM3-3B-Base trunk?

E7b established that ordinary BF16 arithmetic is not the binding loss once the
exact affine geometry is reachable: 893/893 reachable targets were strict
native wins and replayed exactly. The spent E7c development screen then changed
only the edit direction and recovered 1,200/1,200 exact intervals and
1,200/1,200 strict native wins, including all 307 prior refusals and losing none.
That screen was post-outcome development evidence. This protocol is the first
untouched confirmation of the selected operator.

## Bound development evidence

- E7b result SHA-256:
  `9a5232b33f8828195794cb137ee2ec83e10ec8af524418abe3608cf0ac2d4d11`;
- E7b verdict SHA-256:
  `5f5e35f71d432d2df0287c7dec1350e8b53c3dae7bea0f025634536a3a6fd016`;
- spent E7c screen SHA-256:
  `80e529870bfd9d727e85ae4115c8c399b3f23ad94143d1a01cc713e060790104`;
- spent E7c scientific SHA-256:
  `83e17daed19cfcf5b13e1d6672714d9f7537dde5ed0466a4cbc64631568c6bf2`;
- screen implementation commit:
  `ec182f9f6511a513e2a1ba923d2fe50e13e5b00d`;
- selected candidate: `zca_half_raw_r1e2`.

The spent screen is development evidence only. Its validation split prevented
candidate selection from reading 617 held-out spent-pool rows, but the whole
E7b pool is now outcome-spent and cannot confirm this claim.

## Frozen model, prompt, and fresh population

The model, tokenizer, model-file hashes, architectural invariants, prompt,
dtypes, exact certificate arithmetic, dose grid, and native-head arithmetic are
identical to E7b.

- model: `HuggingFaceTB/SmolLM3-3B-Base`;
- revision: `d78a42f79198603e614095753484a04c10c2b940`;
- parameters: 3,075,098,624;
- prompt: `The capital of Vorenia is`;
- prompt SHA-256:
  `0a30c4661b062d65055580bd1dad19d345ab3a9f6f726a28afb7c68548272e1c`;
- certificate boundary: float64 head, hidden, directions, slopes, intervals,
  doses, and margins;
- native boundary: ordinary BF16 hidden plus ordinary BF16 output head;
- target order: sorted eligible single-token ids at indices `[2400:3600]`.

The untouched E7c pool is frozen before any prompt hidden or edited output:

- eligible-token population: 41,518;
- target count: 1,200;
- first target id: 7,863;
- last target id: 10,844;
- target-list SHA-256:
  `fdde53009a17b075fadacafeb15053df08f7f9e97d12f5a9c950d0a5b58b2ff4`;
- overlap with E7 `[0:1200]`: zero;
- overlap with E7b `[1200:2400]`: zero.

Eligibility and target-list hashing are permitted in preflight because they use
only tokenizer structure. No target receives a prompt-conditioned score before
the single-use attempt receipt is created.

## Fixed candidate operator

Let `E` be the untouched BF16 unembedding converted to float32. Compute

`mu = mean_rows(E)`

`C = E^T E / V - mu mu^T`

and symmetrize `C <- (C + C^T)/2` before `torch.linalg.eigh`. Negative numerical
eigenvalues are clamped to zero. Let `m` be the median strictly positive
eigenvalue and `r = 0.01 m`. The fixed candidate operator is

`A = (C + r I)^(-1/2)`.

For target row `e_t`, the candidate direction is

`v_t = (e_t A) / (||e_t A||_2 + 1e-12)`.

There is no centering of `e_t`, no full inverse, no diagonal approximation, and
no alternative ridge or exponent. The matched raw control is

`u_t = e_t / (||e_t||_2 + 1e-12)`.

The preflight computes the operator from the head only, before loading the
registered prompt or evaluating either arm. On the registered A100/Torch
runtime it must reproduce the spent-screen basis exactly:

- mean SHA-256:
  `ae2b4c5d0c3aa902b7ae61dfb320177fea40270c1450f15a9fb5c064bed9e489`;
- eigenvalues SHA-256:
  `7f33aaca070a3539f8c4c65a0e8a56ef82a2813d950ec910335160f971da4971`;
- eigenvectors SHA-256:
  `d5e1b3f6c5fafa4b9fd46bfbcbf1eddd982a2b19ea9a40542ba236a8ebdb3fad`;
- median positive eigenvalue: `0.007535659708082676`;
- ridge: `0.00007535659708082676`.

Any mismatch is pre-outcome `IMPLEMENTATION_INVALID`; do not consume the fresh
pool. Preflight records hashes of `A`, the raw directions, and the candidate
directions, then the execution receipt freezes those exact tensors for the run.
The model weights and output head remain bit-identical. This is probe-side edit
direction preconditioning, not head rescaling and not model training.

## Exact arms and measurement

Both arms start from the same registered BF16 prompt hidden. For every target,
compute the full-vocabulary affine intercept and slope in float64, classify the
geometry exhaustively, and select beta with the unchanged 120-point open-
interval grid and cap from E7b.

For every geometrically reachable row, construct the candidate hidden in
float64, cast once to BF16, and call the untouched ordinary output head. A row
is admitted only when the target logit is strictly greater than every other
logit. Ties and wrong argmaxes are refusals. The lowest token id wins only when
identifying the stored competitor among equal non-target maxima; it never turns
a target tie into an admission.

Each arm evaluates its complete geometrically reachable set in sorted target-id
order, head batches of 16 with a final remainder. Record the full BF16 logit
vector SHA-256, target logit, lowest-id strongest competitor, competitor logit,
native margin, prediction, batch start, batch offset, and batch size for every
reachable row.

The raw control is descriptive and attribution-binding. It cannot select,
modify, or dose the candidate.

## Single-use execution and replay

Named outputs live under `results/e7c_3b_staging/`:

1. `preflight.json`;
2. `execution_receipt.json`;
3. `attempt.json`;
4. `admission.json`;
5. `result.json`;
6. `verdict.json`.

All six must be absent at preflight. Every write is atomic create-new; no output
may be overwritten. Registration binds the committed sources, this protocol,
tests, environment, native arithmetic switches, model files, fresh pool,
development evidence, basis/operator/direction hashes, and exact commands.

Admission creates `attempt.json` with status `POOL_SPENT` before computing the
fresh prompt hidden or any raw/candidate geometry. Once the attempt exists, this
pool cannot be re-registered after a valid FAIL. A known source, environment,
artifact, arithmetic, or replay failure takes precedence as
`IMPLEMENTATION_INVALID`. External interruption with no known invalidity is
`INCOMPLETE` and may resume only under the recovery rule encoded before launch.

Replay is a new Python process on the identical registered node/runtime. It
recomputes both arms over every target with identical sorted order and batch
partitions. It must reproduce all geometry decisions and all reachable-row
native values and full-logit hashes exactly.

Registered commands, to be implemented and hash-bound before outcomes:

```text
python scripts/preflight_e7c_3b.py
python scripts/register_e7c_3b.py
python scripts/admit_e7c_3b.py
python scripts/replay_e7c_3b.py
python scripts/verify_e7c_3b.py
```

## Binding validity bars

Any failure below is `IMPLEMENTATION_INVALID`:

- every protocol, implementation, model, development-evidence, pool, receipt,
  and result hash matches its registered value;
- all source files are tracked, committed, and unchanged after preflight;
- the runtime is exactly one A100-SXM-64GB with the registered Python, Torch,
  Transformers, CUDA, cuDNN, and BF16/TF32/autocast switches;
- model parameter count, dimensions, tied head, bias-free head, frozen
  parameters, tokenizer pool, prompt, and baseline native-head identity pass;
- the head-only basis and fixed operator reproduce every registered hash before
  fresh prompt evaluation;
- raw control geometry reproduces the unchanged E7b implementation on a
  deterministic synthetic fixture;
- candidate construction has a positive control and a raw/candidate inequality
  fixture, so the two arms cannot silently collapse to the same direction;
- every row receives exactly one exhaustive geometry class;
- every accepted beta is strictly inside its open interval with positive
  float64 full-vocabulary margin;
- every reachable row receives exactly one ordinary BF16 native-head check;
- all native logits are finite BF16 vectors of shape `[128256]` and their hashes
  are recomputed from the selected row vector;
- replay covers the complete frozen pool and exactly reproduces both arms;
- attempt, admission, result, and verdict preserve the observed live hash chain.

## Binding scientific bars

E7c is `PASS_3B_ZCA_GEOMETRY` only if all validity bars pass and:

1. candidate geometrically reachable count is at least 1,188 of 1,200 (99%);
2. candidate native-admitted count is at least 1,188 of 1,200;
3. `100 * candidate_native_admitted >= 99 * candidate_geometrically_reachable`;
4. candidate reachability exceeds raw-control reachability by at least 180
   targets (15 percentage points);
5. no more than 12 raw-reachable targets become candidate-unreachable; and
6. every candidate admission and every raw-control measurement reproduces in
   the new-process replay with exact values and hashes.

Failure of a scientific bar with valid execution is
`FAIL_3B_ZCA_GEOMETRY`. There is no partial PASS.

## Claim boundary after PASS

A PASS would license only:

> On the pinned SmolLM3-3B-Base revision, registered A100/BF16 runtime, fixed
> prompt, and third disjoint single-token pool, a head-only ZCA half-whitening
> preconditioner increased exact affine edit reachability by at least 15
> percentage points over raw target-row directions and produced strict ordinary
> BF16 target wins for at least 99% of all targets, with exact new-process
> replay.

It would not prove multi-token editing, natural-language routing, factual
correctness, PPL neutrality after applying arbitrary edits, cross-model
portability, semantic explanation, training scalability, attention replacement,
or legal/audit compliance.

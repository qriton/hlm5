# HLM5 E7b: Native-Precision Deployment Admission Protocol

**Date:** 2026-08-10

**Status:** frozen before E7b implementation and before any model-forward
evaluation on the disjoint target pool

**Scope:** HLM5 model research only. E7b is the bounded repair experiment
required by the valid E7 portability failure. It changes no model weight, gate,
retrieval rule, memory value, attention operator, or E7 artifact. HLM-KB, Flow,
Ingesto, model training, and model scaling are outside scope.

## Question

Can a two-part deployment admission rule fail closed around E7's
float64-to-BF16 composition gap without relaxing the exact-sign geometry and
while retaining at least 99% of geometrically reachable targets on a fresh
disjoint pool?

The rule under test is:

1. require the unchanged exact-sign float64 geometric certificate; then
2. require a strict target-versus-every-competitor win through the ordinary
   native BF16 head, refusing every tie or miss.

The second part is an observed deployment check, not a closed-form theorem.
E7b can establish a fail-closed deployment admission contract for this pinned
trunk and runtime. It cannot establish exact real-arithmetic behavior, hardware-
independent numerical guarantees, semantic explainability, multi-token editing,
or EU AI Act compliance.

## Bound E7 record

E7 remains immutable and is an input to this experiment:

- E7 result: `results/e7_3b_result.json`, SHA-256
  `42174f0e8d3658e39a8a82b730844fd0c4e77161d4ebc81456243b645141c05c`;
- E7 verdict: `results/e7_3b_verdict.json`, SHA-256
  `83c139aac4f25f98ac58dee5fd468f7a2bae920732a8f7587953033370dcf0ed`;
- formal verdict: `FAIL_3B_PORTABILITY`, implementation validity PASS; and
- spent regression sentinel: target id 922, beta
  `217.078019333642`, positive float64 margin `2.2585672901487284`,
  native margin `0.0`, native prediction id 220, and native-logit SHA-256
  `9f39d5f1f7b6c14fac9cab289915c1a8af6ef60bd8d6dbe029f45740b21929a8`.

The sentinel is excluded from the fresh scientific pool. E7b parses this exact
stored E7 row and applies the pure native-decision rule to its recorded target,
prediction, and native margin; the rule must refuse it. E7b
does not re-forward the sentinel because E7 produced its backbone hidden and
head logits inside different registered batch shapes. This stored-record check
is a regression test of the rule, not fresh evidence or a numerical replay.

## Frozen trunk and runtime

E7b uses exactly the E7 trunk and arithmetic environment:

- `HuggingFaceTB/SmolLM3-3B-Base` at revision
  `d78a42f79198603e614095753484a04c10c2b940`;
- exactly 3,075,098,624 frozen parameters, 36 layers, hidden size 2,048,
  vocabulary 128,256, tied embeddings, and a bias-free head;
- native model and head dtype BF16 on one CUDA device;
- float64 certificate arithmetic, `EPS = 0.0`, and the unchanged 120-point
  open-interval dose grid; and
- the exact E7 model-file hashes and baseline-only head-identity bar.

The implementation receipt pins Python, PyTorch, Transformers, CUDA, cuDNN,
GPU identity/capability, source hashes, and commands. A different runtime is a
new experiment, not a substitute result.

## Frozen prompt and disjoint target pool

The sole direct-injection prompt is the E7 original envelope key:

`The capital of Vorenia is`

Its compact-JSON SHA-256 is
`0a30c4661b062d65055580bd1dad19d345ab3a9f6f726a28afb7c68548272e1c`.

The eligible-token rule is unchanged: replace tokenizer prefix `Ġ` with a
space, retain tokens matching a leading space plus `[A-Za-z]{3,}`, and sort
token ids. E7 used eligible indices `[0:1200]`. E7b uses the next 1,200 eligible
ids, indices `[1200:2400]`:

- count: 1,200;
- first id: 4,671;
- last id: 7,859;
- compact-JSON SHA-256:
  `8fb09e2ceda0ad576dcb4771cbdbf757810cfd90969e3e8b4283d9582495136f`;
- overlap with E7: zero; and
- total eligible tokens under the pinned tokenizer: 41,518.

Only tokenizer metadata was read to freeze this pool. No model forward,
certificate, edited hidden state, or native edited logit was computed for it
before this protocol was frozen.

## Exact geometric stage

The exact implementation is the E7 producer
`scripts/run_e7_3b.py`, SHA-256
`48ba22bfc1e6a0824a7f659bd945e028e8d1eb5999b07c21d018a74e6a97d10e`.
E7b may factor these operations into a reusable module, but a locked synthetic
test must reproduce the E7 producer's reachability, bounds, beta, and margin.

For target `y`, native hidden `h`, and tied BF16 head `W`, detach and cast `W`
and `h` to float64 on CUDA. Compute the float64 row norms and
`v = W_y / (norm(W_y) + 1e-12)`. For every competitor `j != y`, define

`a_j = (W_y - W_j)^T h` and `b_j = (W_y - W_j)^T v`.

The target self-row is excluded. Use the exact E7 sign conventions:

- `b_j > 0` contributes a lower bound;
- `b_j < 0` contributes an upper bound;
- `b_j <= 0 and a_j <= 0` is a hard blocker;
- `EPS = 0.0` with no slope dead zone; and
- `L = max(0, max(-a_j / b_j for b_j > 0))`;
- `U = min(-a_j / b_j for b_j < 0)`, or positive infinity when that set is
  empty; and
- select beta on the same 120-point registered grid, capped at
  `L + max(50, 5L)` and by finite `U`.

The grid is constructed on CUDA in float64 as
`L + torch.linspace(0, 1, 122, dtype=float64, device=cuda)[1:-1] * (high-L)`.
Worst margins are reduced over all 128,256 rows in grid chunks of 8; target
directions/slopes are built in sorted-pool chunks of 40. `torch.argmax` selects
the first (lowest-grid-index) maximum. Positive infinity is allowed only for
`U`; any other NaN or infinity is implementation-invalid.

A target is geometrically reachable **if and only if** it has no hard blocker,
`L < U`, the selected beta obeys `L < beta < U`, and the recomputed worst-case
float64 margin is strictly positive. Every target is classified exactly once as
`REFUSE_HARD_BLOCKER`, `REFUSE_EMPTY_INTERVAL`, or `GEOMETRIC_REACHABLE`; no
other geometric refusal reason exists. No geometric refusal may enter native
admission. The new-process replay independently recomputes this exhaustive
classification for all 1,200 targets before calculating coverage.
Failure to construct an interior grid dose with positive margin for a
no-hard-blocker `L < U` row is implementation-invalid, not a new refusal class.

## Native admission stage

The prompt is encoded without special tokens and evaluated alone (batch size
one). Its final native hidden is selected from `model.model(...).last_hidden_state`
at the last attended token. Autocast is disabled. The registered process sets
`torch.backends.cuda.matmul.allow_tf32 = False`,
`torch.backends.cudnn.allow_tf32 = False`,
`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True`,
`torch.set_float32_matmul_precision("highest")`, and
`torch.use_deterministic_algorithms(False)` before model load, and records these
values in both preflight and execution receipts.

For each geometrically reachable target, construct the deployed hidden exactly
as E7 did:

`x_native = Q_BF16(h_float64 + beta * v_float64)`.

Process reachable targets in sorted target-id order, in contiguous head batches
of exactly 16 except the final remainder. Pass each contiguous `[B, 2048]` BF16
tensor through the ordinary tied output module with autocast disabled. Every
returned `[B, 128256]` tensor must be finite. Let `c` be the maximum native BF16
logit over all non-target rows. If multiple competitors share the maximum, the
competitor id is the lowest token id attaining it. Native admission is:

`ADMIT_NATIVE iff logits[y] > c and argmax(logits) == y`.

Equality is refusal. The comparison is performed on the returned BF16 values;
the reported margin is `float32(logits[y]) - float32(c)`. Every row records the
target, interval, beta, float64 margin, native target logit, competitor id and
logit, native margin, prediction, decision, and full native-logit row hash.
For row offset `i`, first select
`row_logits = logits_batch[i].detach().contiguous()`, require dtype BF16 and
shape `[128256]`, and compute SHA-256 over
`row_logits.cpu().view(torch.uint8).numpy().tobytes(order="C")`. The recorded
dtype is `bfloat16` and recorded shape is `[128256]`. Replay recomputes the global
non-target maximum from the complete ordinary-head output and never trusts the
stored competitor scalar.

Coverage cost is reported as:

- geometric coverage / 1,200;
- native-admitted coverage / 1,200;
- retained fraction = native admitted / geometrically reachable; and
- counts of native ties and native wrong-argmax misses.

No dose search, margin threshold, retry, alternate direction, or target
replacement is allowed after the native result is seen.

## New-process replay

Admission and replay are separate registered commands. The admission command
writes a single-use admission receipt. The replay command starts a new Python
process, reloads the pinned model, re-derives the prompt hidden and directions,
and reconstructs the complete frozen `GEOMETRIC_REACHABLE` set. It evaluates
that full set in the identical sorted target-id order, identical contiguous
16-row partitions, and identical final remainder, using every reachable row's
frozen beta. Native-refused rows remain in the replay batches so they cannot
change an admitted row's numerical context.

Replay must reproduce every native decision over the full reachable set. For
every admitted target it must additionally reproduce:

- the same strict target win;
- the same target and competitor BF16 values;
- the same prediction and native margin; and
- the exact full-logit SHA-256.

The replay independently recomputes the complete geometric classification for
all 1,200 targets, then recomputes each admitted target's float64 interval and
margin and confirms the stored beta remains strictly inside the interval. It
may not change admission decisions. The result records both the raw SHA-256 of
`admission.json` and a stable compact-JSON SHA-256 over its scientific payload.
Both admission and result documents contain a top-level `scientific` object and
a `scientific_sha256` equal to the SHA-256 of that object serialized as UTF-8
JSON with sorted keys, compact separators, `ensure_ascii=False`, and
`allow_nan=False`. Runtime duration and provenance live outside that object.

## Execution order and receipts

All new outputs remain under `results/e7b_3b_staging/` until a reviewed verdict
permits promotion. The named outputs are `preflight.json`,
`execution_receipt.json`, `attempt.json`, `admission.json`, `result.json`, and
`verdict.json`. Every named output uses atomic create-new semantics and refuses
to overwrite an existing path.

1. Commit this protocol alone.
2. Implement the E7b contract, preflight, registration, admission, replay,
   verifier, and
   targeted tests. Commit them before fresh candidate access.
3. Run a baseline-only preflight. All six named outputs must initially be
   absent. It may hash the fresh pool, verify E7 and the
   frozen model, run tests, load the baseline model, and repeat E7's unedited
   head-identity anchor. It must compute no certificate or edited output on the
   fresh pool.
4. Run the explicit registration command. It verifies the live preflight,
   reruns the locked tests, confirms `attempt.json`, `admission.json`,
   `result.json`, and `verdict.json` are absent, and atomically creates a
   pre-outcome execution receipt bound to the committed implementation,
   protocol, environment, tests, pool, prompt, model, E7 inputs, and commands.
5. The admission command first creates `attempt.json`, binding the execution
   receipt and declaring the fresh pool spent, and then runs native admission,
   producing `admission.json`.
6. In a new process, run the new-process replay, producing `result.json`.
7. Emit the formal verdict before changing the paper, README, or promoted
   results.

Registered commands:

```powershell
python scripts/preflight_e7b_3b.py
python scripts/register_e7b_3b.py
python scripts/admit_e7b_3b.py
python scripts/replay_e7b_3b.py
python scripts/verify_e7b_3b.py
```

## Binding implementation-validity bars

Any failure below is `IMPLEMENTATION_INVALID`:

- protocol, implementation, dependency, E7-input, model-file, prompt, pool,
  preflight, execution-receipt, and admission hashes match exactly;
- every registered source is tracked, committed, and unchanged after preflight;
- the model invariants, BF16 dtype, and baseline-only head-identity bar match E7;
- the target pool has exactly 1,200 entries, the registered endpoints/hash, and
  zero overlap with E7;
- preflight and execution receipt record that no fresh candidate output was
  computed;
- admission evaluates every pool target exactly once and partitions it into
  hard-blocker refusal, empty-interval refusal, native-tie refusal,
  native-wrong-argmax refusal, or native admission;
- the new-process replay independently recomputes the geometric class for all
  1,200 rows and matches admission exactly before coverage is evaluated;
- no geometric refusal is natively admitted;
- every admitted beta is strictly inside its exact-sign interval and has a
  positive recomputed float64 margin;
- every native decision has ordinary-head evidence and the strict comparison is
  recomputed from stored values;
- the exact stored E7 sentinel row is recovered from the hash-pinned E7 result
  and refused by the pure native-decision rule;
- replay covers exactly the frozen geometrically reachable set in identical
  head batches and changes no native decision;
- `admission.json` binds the raw SHA-256 of the live `attempt.json`;
- `result.json` binds both the raw and scientific hashes of `admission.json`;
  carries the same attempt hash, and
- `verdict.json` binds the raw and scientific hashes of `result.json` and the
  admission and attempt hashes carried by that result;
- every consumer validates the live `attempt.json`, including its execution-
  receipt hash and its recorded pre-admission absence of `admission.json`,
  `result.json`, and `verdict.json`.

An external interruption with no known validity failure is `INCOMPLETE`.

Before `attempt.json` exists, an implementation-only failure may be repaired
only by preserving the failed pre-outcome receipts in a separately named
archive and registering fresh receipts against the committed repair. Once
`attempt.json` exists, the pool is permanently spent: PASS, valid FAIL,
implementation-invalid, and interrupted attempts may not be rerun or
re-registered on these 1,200 targets. A known validity failure takes precedence
over interruption; `INCOMPLETE` applies only when no validity failure is known.

## Scientific verdict

`PASS_3B_NATIVE_ADMISSION` requires every validity bar plus:

1. at least 960 of 1,200 fresh targets are geometrically reachable and at least
   one is native-admitted;
2. native admission retains at least 99% of geometrically reachable fresh
   targets, evaluated without floating-point rounding as
   `100 * n_native_admitted >= 99 * n_geometrically_reachable`;
3. every admitted fresh target has a strict positive native margin and target
   argmax during admission;
4. every admitted target repeats the strict win in the new-process replay; and
5. all admitted native-logit hashes and recorded native values reproduce
   exactly.

`FAIL_3B_NATIVE_ADMISSION` means the implementation is valid but one or more
scientific bars fail. Coverage, refusals, margins, and the sentinel outcome must
be reported regardless of verdict. A pass does not convert the second-stage
deployment check into a mathematical certificate.

## Claim boundary

A pass licenses only:

> On the pinned SmolLM3-3B-Base revision and registered CUDA/BF16 runtime, a
> two-part HLM5 rule combining exact-sign float64 geometry with strict ordinary-
> BF16-head admission refused native ties and replayed every admitted target on
> the registered disjoint single-token pool, while retaining at least 99% of
> geometrically reachable targets.

It does not license hardware-independent exactness, a general model editor, a
trained HLM 3B model, paraphrase generalization, benchmark superiority, or a
regulatory-compliance claim.

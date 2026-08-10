# HLM5 E8: fresh full-path ZCA adapter protocol

Date: 2026-08-10

Status: preregistration; no E8 fresh-case model forward has been computed

Scope: HLM5 model research only. HLM-KB, HLM-Flow, Ingesto, semantic routing,
and EU AI Act claims are out of scope.

## Decision question

Does the single head-only ZCA direction that passed E7c remain effective when
stored behind the released multi-slot HLM5 exact-key gate, while off-support
prompts remain closed and removing the slots restores exact baseline behavior?

E7c proved a narrower fact: on one frozen prompt and a third disjoint
1,200-token pool, the fixed direction increased exact affine reachability from
705/1,200 to 1,200/1,200 and produced 1,200/1,200 strict ordinary-BF16 target
wins. E8 changes the prompt for every case and exercises the composed operator:
certificate, native admission, key whitening, multi-slot retrieval, hard gate,
ordinary output head, locality, and rollback.

## Bound prior evidence

- E7c registered implementation commit:
  `58cedff87f270f1e7047e491a80565b7da05390f`;
- E7c result SHA-256:
  `88c9819648ce7a30ba3e6d75aa5da2c317d072db8eae0160631dada212f9cf44`;
- E7c verdict SHA-256:
  `32cc956a9a56e0be56e0976c4c17e7862544a2ede07735ac45ea6faffba0f1f6`;
- E7c formal verdict: `PASS_3B_ZCA_GEOMETRY`;
- selected operator: `zca_half_raw_r1e2`.

These artifacts are development evidence for E8. They may fix the operator and
bars; they may not be changed after E8 outcomes.

## Frozen trunk and arithmetic

E8 uses exactly the E7c trunk and registered runtime:

- model: `HuggingFaceTB/SmolLM3-3B-Base`;
- revision: `d78a42f79198603e614095753484a04c10c2b940`;
- 3,075,098,624 frozen parameters, 36 layers, hidden size 2,048, vocabulary
  128,256, tied embeddings, bias-free head;
- native model and head arithmetic: BF16, autocast off;
- one `NVIDIA A100-SXM-64GB`;
- Python 3.11.6, Torch 2.5.1+cu121, Transformers 5.12.1, CUDA 12.1,
  cuDNN 90100;
- TF32 off, float32 matmul precision `highest`, BF16 reduced-precision
  reduction on, deterministic algorithms off; and
- no model training, quantization, weight mutation, alternate checkpoint,
  attention replacement, or head rescaling.

The head-only covariance and ZCA operator must reproduce E7c exactly:

- mean SHA-256:
  `ae2b4c5d0c3aa902b7ae61dfb320177fea40270c1450f15a9fb5c064bed9e489`;
- eigenvalues SHA-256:
  `7f33aaca070a3539f8c4c65a0e8a56ef82a2813d950ec910335160f971da4971`;
- eigenvectors SHA-256:
  `d5e1b3f6c5fafa4b9fd46bfbcbf1eddd982a2b19ea9a40542ba236a8ebdb3fad`;
- operator SHA-256:
  `ac2536bf5a28014c665177fb89ff907ae2456a276c5c096edc767900b4a8d065`;
- median positive eigenvalue: `0.007535659708082676`;
- ridge: `0.00007535659708082676`.

For native head row `e_t`, the primary value direction remains

`v_t = normalize(e_t (C + rI)^(-1/2))`.

There is no centering of `e_t`, exponent/ridge sweep, head mutation, or
post-outcome operator selection. The matched descriptive control is
`u_t = normalize(e_t)`.

## Fresh CounterFact population

The sole data source is `scripts/data/counterfact.json`, SHA-256
`d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f`.
The existing CounterFact certificate artifact used case IDs 0 through 999.
E8 starts at case ID 20,000.

Traverse records in file order and retain a row iff:

1. `case_id >= 20000`;
2. `" " + target_new.str.strip()` is exactly one token under the pinned
   tokenizer;
3. `requested_rewrite.prompt.format(subject)` is nonempty;
4. after stable de-duplication, at least two paraphrase prompts and two
   neighborhood prompts remain.

For each retained row, store case id, rendered exact prompt, space-prefixed
target string, target token id, the first two stably de-duplicated paraphrases,
the first two stably de-duplicated neighborhood prompts, relation id, and
subject. Take the first 64 retained rows.

The registered case IDs are:

`[20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009,
20010, 20011, 20012, 20013, 20014, 20015, 20016, 20017, 20018, 20019,
20020, 20021, 20022, 20023, 20024, 20025, 20026, 20027, 20028, 20029,
20030, 20031, 20032, 20033, 20034, 20035, 20036, 20037, 20038, 20039,
20040, 20041, 20042, 20043, 20044, 20045, 20046, 20047, 20048, 20050,
20052, 20053, 20054, 20055, 20056, 20057, 20058, 20059, 20060, 20061,
20062, 20063, 20064, 20065]`.

- selected-case object SHA-256:
  `6d81f2ccf551b76402213593aea5b12ad37b8eb556fe3ae2a5731bd6cf5cb758`;
- unique exact prompts: 64;
- unique target token ids: 50;
- overlap with E7 exact prompts: zero.

The control texts are the eight E7 neutral prompts and 18 E7 whitening texts:

- neutral-list SHA-256:
  `c535b5a8eeeae85b0918bee646c82fad78976797f6df6a9bfb939512583e1d69`;
- whitening-list SHA-256:
  `be7bf125fee1dcf350af6710a6949d77f937c2272de90e8cfd907b08d966fa42`.

Prompt-forward order is: for each selected case, exact prompt, two
paraphrases, then two neighborhood prompts; then the neutral list; then the
whitening list. Stable-de-duplicate while preserving first occurrence. This
produces 337 unique prompts, SHA-256
`7652c5c71736fc3c0b33845664638aead1c09b2ff84d128618c36bcd20d0db00`.
The frozen trunk evaluates them in this order in batches of 16, including the
final one-row batch.

Selection, tokenization, hashes, and prompt order are permitted in preflight.
No selected prompt hidden, baseline logit, certificate, gate score, or edited
output may be computed before the single-use attempt receipt.

## Fixed composed operator

### Certificate and native admission

For each exact-prompt hidden `h_i`, evaluate both raw and ZCA directions with
the unchanged float64 exact-sign certificate and 120-point open-interval dose
grid. A row is deployment-admitted only if:

1. exact geometry is reachable;
2. beta lies strictly inside `(L, U)`;
3. recomputed full-vocabulary float64 margin is positive; and
4. `BF16(h_i + beta v_i)` passed through the untouched ordinary BF16 head has
   the target strictly above every competitor.

Lowest-id tie breaking may identify the stored non-target competitor; it may
not turn a target tie into an admission. Baseline-correct facts are reported
and excluded from edit-efficacy denominators and memory insertion.

### Shared key operator

Key routing is identical for the raw and ZCA value arms and is not selected by
their outcomes. Compute `calibrate_whitening` once from all 64 exact-prompt
float32 hiddens plus the 18 registered whitening hiddens, floor fraction 0.01.
The mean uses all 64 exact keys. Freeze hashes of the resulting mean and ZCA
key transform in admission and replay.

### HLM5 memory and gate

For each arm independently:

- insert only deployment-admitted, non-baseline-correct facts;
- memory size is 72, degree 5, temperature 0.10, non-learnable;
- key is the normalized shared-whitened exact hidden;
- stored value is the arm's unit direction multiplied by its registered beta;
- attach boost is 1.0;
- hard gate threshold is 0.95.

Evaluate every exact, paraphrase, neighborhood, and neutral prompt through the
same arm-specific memory. Record selected score, selected slot, own slot,
own-slot weight, gate decision, delta hash, baseline/adapted logits and hashes,
predictions, target margin where applicable, and bit identity.

An admitted exact key succeeds only if the gate opens, its own slot is selected,
and the untouched ordinary BF16 head strictly predicts its target. A
baseline-correct, refused, paraphrase, neighborhood, or neutral row is
off-support and must remain gate-closed.

### Exact rollback

After the primary ZCA evaluation, remove every active ZCA slot using
`EditableHLM5Memory.remove`. Re-evaluate all 337 prompt hiddens through the same
adapter. The memory must have zero active slots, zero alpha on every removed
slot, no labels on removed slots, exact-zero native deltas, and logits
bit-identical to the frozen baseline for every prompt.

Rollback is tested after, never before, the primary scientific rows are frozen.

## Arms

1. frozen baseline;
2. raw target-row direct admission and full HLM5 path, descriptive control;
3. fixed ZCA target-row direct admission and full HLM5 path, primary arm;
4. primary-arm complete slot removal and exact rollback.

No arm may alter another arm's beta, keys, slot count, values, gate, or verdict.

## Single-use execution and replay

Named outputs live under `results/e8_3b_staging/`:

1. `preflight.json`;
2. `execution_receipt.json`;
3. `attempt.json`;
4. `admission.json`;
5. `result.json`;
6. `verdict.json`.

All writes are atomic create-new. All six outputs must be absent at preflight.
Registration binds committed sources, this protocol, tests, environment,
model files, E7c evidence, CounterFact data, selected cases, prompt order,
head basis/operator, target-direction hashes, commands, and absent outcomes.

Admission creates `attempt.json` with status `POOL_SPENT` before any selected
prompt forward. A known post-attempt failure is durably
`IMPLEMENTATION_INVALID`. External interruption without known invalidity is
`INCOMPLETE`; no rerun is allowed unless a recovery rule was fixed in the
execution receipt before launch.

Replay is a new Python process on the identical node/runtime. It reconstructs
the full selected population, prompt hiddens, both direct arms, both memories,
all gate rows, and rollback. Every scientific object and full-logit hash must
reproduce exactly.

Registered commands, to be implemented and hash-bound before outcomes:

```text
python scripts/preflight_e8_3b.py
python scripts/register_e8_3b.py
python scripts/admit_e8_3b.py
python scripts/replay_e8_3b.py
python scripts/verify_e8_3b.py
```

## Binding validity bars

Any failure below is `IMPLEMENTATION_INVALID`:

- every protocol, source, dependency, model, evidence, data, sample, prompt,
  tensor, receipt, and result hash matches its registered value;
- runtime and native arithmetic exactly match E7c;
- model invariants and baseline-only native-head identity pass before attempt;
- E7c head basis/operator and E8 raw/ZCA direction hashes reproduce before
  selected-prompt access;
- a synthetic fixture proves raw certificate equivalence, raw/ZCA arm
  inequality, own-slot selection, gate-off bit identity, and remove-to-empty
  rollback;
- each row receives one exhaustive geometry/native admission decision;
- every admitted beta is inside its open interval with positive float64 margin
  and strict ordinary-BF16 target victory;
- both memory receipts exactly match their admitted keys, values, labels,
  strengths, and active slots;
- every prompt receives the required baseline and arm-specific gate/head
  evidence with finite values and full-logit hashes;
- every gate-closed row is bit-identical to baseline;
- rollback covers exactly all 337 frozen prompts and is bit-identical; and
- new-process replay exactly reproduces both arms and rollback.

## Binding scientific bars

E8 is `PASS_3B_FULL_PATH_ZCA` only if every validity bar passes and:

1. at least 48 of 64 cases are non-baseline-correct and therefore edit-eligible;
2. candidate deployment admissions satisfy
   `20 * candidate_admitted >= 19 * eligible` (at least 95%);
3. candidate admits no fewer facts than the raw control and loses at most one
   raw-admitted fact;
4. every candidate-admitted fact opens its exact gate, selects its own slot,
   and strictly predicts its target through the full HLM5 path;
5. in the primary ZCA arm, every baseline-correct or refused exact prompt, all
   128 paraphrases, all 128 neighborhood prompts, and all eight neutral prompts
   have zero false gate applications; and
6. complete primary-memory removal restores bit-identical baseline logits on
   all 337 prompts.

Failure of a scientific bar with valid execution is
`FAIL_3B_FULL_PATH_ZCA`. There is no partial pass. Raw-arm efficacy, margins,
own-slot weights, gate scores, and value-direction gain are reported regardless
of outcome and cannot tune the bars.

## Claim boundary after PASS

A PASS licenses only:

> On the pinned frozen SmolLM3-3B-Base revision, registered A100/BF16 runtime,
> and 64 preregistered exact-key CounterFact cases, the fixed head-only ZCA
> value preconditioner produced deployment-admitted single-token edits through
> the released multi-slot HLM5 pre-head memory for at least 95% of eligible
> cases; every admitted exact key succeeded, every registered off-support query
> stayed closed, and removing all slots restored bit-identical baseline logits.

It does not establish factual truth, natural-language routing, paraphrase
generalization, multi-token editing, generation quality, PPL neutrality,
cross-model transfer, trained-HLM scaling, attention replacement, audit-law
compliance, or commercial readiness.

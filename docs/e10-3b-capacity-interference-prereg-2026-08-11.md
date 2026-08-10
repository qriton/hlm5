# HLM5 E10: 3B nested capacity and interference protocol

Date: 2026-08-11

Status: preregistration; no E10 candidate hidden, direct-admission result,
memory score, adapter output, or capacity-tier outcome has been computed

Scope: HLM5 model research only. HLM-KB, HLM-Flow, Ingesto, semantic routing,
and legal or commercial claims are out of scope.

## Decision question

Does the exact-key HLM5 adapter that passed E8 and E9 preserve full-path
efficacy, own-slot routing, wide locality, and exact rollback when active memory
grows from 64 to 256 to 1,024 slots under one fixed key and value operator?

E10 isolates active-set size. Every tier uses the same frozen 3B trunk, the
same 1,200-candidate source pool, the same head-only ZCA value operator, the
same shared key-whitening operator, the same registered doses, the same
1,032-slot physical allocation, the same degree/temperature/gate, and the same
12,288 spent E9 locality rows. Only the number of active prefix slots changes.

## Bound prior evidence

- E8 verdict: `PASS_3B_FULL_PATH_ZCA`;
- E8 result SHA-256:
  `ec064b4d38c1abb9d88f99bd1c138b01cbfe4f9b15dbdf28edbe415477e678b6`;
- E9 amended protocol commit:
  `b16c93199fac5d47c4f29cc64cac147517ee62d1`;
- E9 implementation commit:
  `82e744c76055bff9a4207c1d191deccc38b3ab3f`;
- E9 admission SHA-256:
  `1d67111e3d7d3bba8d59cdd7d04b2197ed380ebce4b3ec00c96982a36a94551e`;
- E9 result SHA-256:
  `4248a926676b53c7eab005bc87f894d8e7a1277e618135872e1e3ff30ddad0b7`;
- E9 verdict SHA-256:
  `a025a56b541423610659ea52e40414fd5cad7454bf4687aabf4991a21e9d90b8`;
- E9 verdict: `PASS_3B_WIDE_LOCALITY`; and
- E9 locality prompt-order SHA-256:
  `7f9c199bb1a402fa95bd46ffa879d01a7e36acb3f9921eb2ac2043b3343ff07a`.

All object hashes use canonical stable JSON. File hashes use raw SHA-256 bytes.

## Frozen trunk and arithmetic

- model: `HuggingFaceTB/SmolLM3-3B-Base`;
- revision: `d78a42f79198603e614095753484a04c10c2b940`;
- 3,075,098,624 frozen parameters, 36 layers, hidden size 2,048, vocabulary
  128,256, tied embeddings, bias-free head;
- native model arithmetic BF16 with autocast off;
- one `NVIDIA A100-SXM4-64GB`;
- Python 3.11.6, Torch 2.5.1+cu121, Transformers 5.12.1, CUDA 12.1;
- TF32 off, float32 matmul precision `highest`, BF16 reduced-precision
  reduction on, deterministic algorithms off;
- exact physical Leonardo node bound at preflight; and
- no training, quantization, trunk/head mutation, attention replacement,
  threshold sweep, or tier-specific retuning.

## Frozen positive candidate pool

The source is `scripts/data/counterfact.json`, raw SHA-256
`d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f`.
Traverse records in file order from case ID 20,077. Retain a row iff:

1. `target = " " + target_new.str.strip()` tokenizes to exactly one token under
   the pinned tokenizer;
2. `requested_rewrite.prompt.format(subject)` is nonempty;
3. the exact prompt is absent from the complete E8 and E9 prompt orders; and
4. the exact prompt has not appeared in an earlier retained E10 row.

Retain `{case_id, prompt, target, target_id, relation_id, subject}` and stop at
1,200 rows. This is data/tokenizer selection only.

- first/last case ID: 20,077 / 21,318;
- selected-case SHA-256:
  `727ab78b1ae1d50e1fc16edf85e2dfda13d452c846dbcdce1d612b7a9603ea63`;
- case-ID-list SHA-256:
  `f1b1fbc4cf575cb2d3232ac1b8f534aeaac52379521cbe20154bbbe7dad7597c`;
- exact-prompt-list SHA-256:
  `82cf0827ed57c988fde30b86b7b280663c036cee57d276b701d16583d9dcf005`;
- target-ID-list SHA-256:
  `479c6cbac4d7b5cf88ebf7ccdd6daea2e680ecb264872755f2f4508f26328442`;
- target-string-list SHA-256:
  `b15f77742627741db8f245214ecbead2a54ca449610a5b358f1dedc3e0238559`;
- E8/E9 prompt overlap: zero.

The 18 E8 whitening prompts are reused in their registered order, SHA-256
`be7bf125fee1dcf350af6710a6949d77f937c2272de90e8cfd907b08d966fa42`.

## Direct admission and deterministic capacity keys

After `attempt.json` is created, forward the 1,200 exact prompts and 18
whitening prompts in fixed BF16 batches of 16. Compute the frozen E7c head-only
ZCA directions and the E8 paired float64 geometry/native-BF16 direct admission
for every candidate. A candidate is deployment-admitted iff it is not already
baseline-correct and its registered dose produces a strict ordinary-BF16 target
win under the untouched head.

Capacity keys are the first 1,024 deployment-admitted candidate indices in
source order. The result must report all 1,200 direct rows, admission coverage,
the selected indices/case IDs, and their hashes. This is disclosed deterministic
outcome selection inside the burned E10 pool; it does not estimate unconditional
CounterFact efficacy. Fewer than 1,024 admitted candidates is a scientifically
valid `FAIL_3B_CAPACITY_1024`, not an implementation failure.

Fit one shared key whitening from all 1,200 exact hiddens plus the 18 whitening
hiddens, using floor fraction 0.01. Every tier uses those exact same mean and
transform tensors.

## Frozen nested memories

For `K in [64, 256, 1024]`, instantiate a fresh released
`EditableHLM5Memory` with:

- physical memory size 1,032;
- hidden dimension 2,048;
- degree 5;
- temperature 0.10;
- attach boost 1.0; and
- hard gate threshold 0.95.

Inject the first K deterministic capacity keys into slots `0..K-1`. Each key
is the shared-whitened exact hidden. Each value is the frozen ZCA direction
multiplied by that candidate's registered direct-admission dose. All remaining
slots stay inactive. Do not recalibrate whitening, values, doses, memory size,
or thresholds by tier.

## Measurements

### Positive full path

For every active exact key, record identity, target, native-hidden hash,
selected score/slot, own-slot score and attention weight, strongest non-own
score/slot, gate decision, delta hash/zero flag, adapted-hidden hash, ordinary
BF16 target/competitor logits and IDs, margin, prediction, and full-logit hash.

Report exact counts plus minimum/median own-slot attention, minimum target
margin, maximum non-own score, key coherence, and degree-five max crosstalk.
Every descriptive median is the float64 linear 0.5 quantile (the arithmetic
mean of the two middle order statistics when the count is even).

### Wide locality

Forward the complete spent E9 12,288-query prompt order once in batches of 16.
For every tier, evaluate the same native hiddens in consecutive 256-row gate
batches. Record the E9 row identity plus selected score/slot/label, gate,
delta hash/zero flag, native/adapted hidden hashes, and exact bit identity.
Report global and per-kind gate counts, counts at 0.10/0.50/0.90/0.95,
maximum, and deterministic top 20.

### Rollback

Remove every active slot in each tier, then reapply that tier to all tier
positives and all 12,288 locality hiddens in the registered batch schedule.
Every post-removal delta must be exact zero and every output hidden must be
bit-identical. Record the post-removal mask/alpha/label state and exact counts.

### Portable tensor bundle

If and only if the 1,024 tier is constructible, admission atomically creates
`adapter_1024.pt` containing only CPU tensors for active keys, active values,
active alphas, key mean, and key transform. The JSON receipt records every
tensor dtype, shape, SHA-256, active label list, selected case IDs, and raw
bundle-file SHA-256. E10 replay loads and verifies this bundle but never
rewrites it. E11 may consume it only after a formal E10 pass.

## Single-use execution and replay

Named outputs live under `results/e10_3b_staging/`:

1. `preflight.json`;
2. `execution_receipt.json`;
3. `attempt.json`;
4. `adapter_1024.pt` when constructible;
5. `admission.json`;
6. `result.json`;
7. `verdict.json`.

All writes are atomic create-new. Registration binds committed sources,
protocol, tests, model/runtime/node, prior evidence, both populations, ZCA
basis, commands, and absent outcomes. Admission creates the burn marker before
the first E10 candidate forward. Known post-attempt failure is
`IMPLEMENTATION_INVALID`; external interruption without known invalidity is
`INCOMPLETE` and cannot be rerun with amended code or bars.

Replay is a new Python process on the same physical node/runtime. It repeats
the direct admission, deterministic key selection, all tiers, every row and
summary, and verifies the saved bundle tensors exactly.

Registered commands:

```text
python scripts/preflight_e10_3b.py
python scripts/register_e10_3b.py
python scripts/admit_e10_3b.py
python scripts/replay_e10_3b.py
python scripts/verify_e10_3b.py
```

## Binding validity bars

Any failure below is `IMPLEMENTATION_INVALID`:

- every protocol/source/dependency/model/data/prior-evidence/population hash;
- exact E8/E9 ZCA basis and model invariants, registered native arithmetic,
  and one physical node frozen at E10 preflight;
- exhaustive 1,200-row direct measurement under fixed batching;
- deterministic first-admitted selection with no substitution;
- one shared key-whitening receipt across every tier;
- fixed 1,032-slot allocation and exact active prefix per tier;
- every recorded score, slot, gate, tensor hash, logit decision, summary, and
  diagnostic recomputes from stored rows;
- every closed gate has exact-zero delta and bit-identical hidden;
- exact rollback state and outputs;
- portable-bundle tensor receipt and raw hash when K=1,024 is constructible;
- exact admission/result hash chain; and
- exact new-process replay.

## Binding scientific verdict

`PASS_3B_CAPACITY_1024` requires:

- at least 1,024 deployment-admitted candidates among the fixed 1,200;
- for every K tier, K/K positive gates open, K/K own-slot selections, K/K
  strict ordinary-BF16 target successes, and positive nonzero deltas;
- for every K tier, 0/12,288 locality gates, exact-zero locality deltas, and
  bit-identical locality outputs globally and per E9 kind;
- for every K tier, exact-zero and bit-identical rollback on all tier positives
  plus all locality rows; and
- exact replay and a valid K=1,024 tensor bundle.

An implementation-valid miss is `FAIL_3B_CAPACITY_1024`, with the largest fully
passing nested tier reported. E11 is authorized only after
`PASS_3B_CAPACITY_1024`; no threshold, dose, whitening, case, or tier may be
changed after E10 outcomes.

## Claim boundary

A pass establishes only that this frozen single-token, exact-key adapter
survived 1,024 active slots on the pinned 3B trunk and finite registered pools.
It does not establish capacity beyond 1,024, population false-positive rate,
semantic routing, paraphrase generalization, factual truth, multi-token
editing, generation quality, PPL neutrality, cross-model transfer, a trained
3B HLM, attention replacement, legal compliance, or commercial readiness.

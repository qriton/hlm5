# HLM5 E9: wide exact-key locality protocol

Date: 2026-08-11

Status: amended preregistration; no E9 query hidden, gate score, or adapter
output has been computed

Scope: HLM5 model research only. HLM-KB, HLM-Flow, Ingesto, semantic routing,
and legal or commercial claims are out of scope.

## Decision question

Does a pre-outcome-frozen, 64-slot reconstruction of the HLM5 adapter recipe
that passed E8 remain completely closed on a much wider, disjoint CounterFact
query surface while its registered exact keys still open their own slots?

E8 established full-path efficacy and exact rollback on 64 fresh exact keys
plus 264 registered off-support controls. E9 changes the query surface and
freezes its reconstruction before outcome access. It scans 12,288 new queries
without changing the registered E9 keys, E8 values or doses, gate threshold,
or trunk after that freeze.

## Pre-outcome amendment record

Leonardo job `51801037` exited during outcome-blind preflight, before
`preflight.json`, registration, `attempt.json`, or any E9 prompt forward. The
E9 query pool therefore remains unspent. A spent-E8-only diagnostic on node
`lrdn2661` found that the ZCA direction hashes, active values, alphas, mask,
labels, slot mapping, and all 64 positive-anchor gate/slot decisions reproduced,
but three key-construction hashes differed from E8 job `51800915`, which ran on
node `lrdn1549`:

- key mean: `3d971efa2956850c92d07d6b9aebe211b1e61cef5dffff050d842247e0a988a0`
  to `c354e69088bbe73fe982cfb011351866441b86d81f68d933a43cab5103db00d3`;
- key transform:
  `ad7b15bf050b4ee0fbca3c8c366e5b9a9b2966f24fb7f999a7ad418aca865dcd`
  to `ffa3d7515d53af62977a68d9ae8ede1a90ccd25bf0ce2d17789eec380f87aef3`;
  and
- active keys:
  `209e4ff0f498490cd9e6211ba23512578e85d41901df5869dcdd693770f846bf`
  to `05321d100d20f1f0745840c4890bc436dd57a6c02fc6a4f50fe83a5a6e920c16`.

This is cross-node numerical portability drift in tensors E8 recorded only by
hash, not by value. It makes a byte-for-byte historical E8-memory
reconstruction unavailable on an arbitrary later node. The repair below was
made before E9 outcome access: freeze the same-recipe reconstruction in E9
preflight, bind its exact node and tensor hashes before the attempt, retain
byte-for-byte equality for every non-key E8 memory field, require all 64 spent
positive anchors to remain live on their own slots, and require exact
same-node admission/replay of the newly frozen reconstruction. No tolerance or
scientific threshold is introduced.

## Bound prior evidence

- E8 protocol commit:
  `02f5186fefa88e9488f0635a5276ce0a9be5999d`;
- E8 implementation commit:
  `3ee23ba8a1451f3231254fe7f7f63b4e3422468d`;
- E8 admission SHA-256:
  `96d6dd8f8528d38de52d559df86f4228b508de4487cf9ea3efd50b63adccc7b0`;
- E8 result SHA-256:
  `ec064b4d38c1abb9d88f99bd1c138b01cbfe4f9b15dbdf28edbe415477e678b6`;
- E8 verdict SHA-256:
  `5e376c4c065ecf7e7602d3e81984e6425175876bc763ffd2a427684a6b9d7deb`;
- E8 formal verdict: `PASS_3B_FULL_PATH_ZCA`;
- E8 candidate direct-row SHA-256:
  `417eb04f694821d23da067209f9f29f1e0443d0ec96fcf3d5fd4572151c272c3`;
- E8 candidate memory-receipt SHA-256:
  `cddc691e44f87a2fceec5153763211b4b112adda0ee8ac1f347c748206b612c1`.

All object hashes in this document use the repository's canonical stable-JSON
encoding. File hashes use raw SHA-256 bytes.

## Frozen trunk, memory, and arithmetic

E9 uses the E8 registered software and arithmetic runtime:

- model: `HuggingFaceTB/SmolLM3-3B-Base`;
- revision: `d78a42f79198603e614095753484a04c10c2b940`;
- 3,075,098,624 frozen parameters, 36 layers, hidden size 2,048, vocabulary
  128,256, tied embeddings, bias-free head;
- native model arithmetic BF16 with autocast off;
- one `NVIDIA A100-SXM-64GB`;
- Python 3.11.6, Torch 2.5.1+cu121, Transformers 5.12.1, CUDA 12.1;
- TF32 off, float32 matmul precision `highest`, BF16 reduced-precision
  reduction on, deterministic algorithms off; and
- no training, quantization, trunk/head mutation, attention replacement, or
  threshold sweep.

Reconstruct the primary memory from the E8 64 exact-prompt hiddens, 18 key-
whitening prompts, frozen ZCA value directions, and registered per-case doses.
Before E9 outcome access, preflight must:

- reproduce every E8 candidate-memory receipt field except exactly
  `active_keys_sha256`, `key_mean_sha256`, and `key_transform_sha256`;
- reproduce the frozen E8 ZCA basis and value-direction hashes exactly;
- require all 64 spent E8 exact-key anchors to open and select their own slots;
- record the reconstructed key mean, key transform, and active-key hashes; and
- bind the physical Leonardo node as part of the execution environment.

Registration freezes the complete reconstructed E9 memory receipt and key-
whitening receipt before `attempt.json`. Admission and replay must reproduce
those newly frozen bytes exactly on that same node. The three historical E8
key hashes remain provenance diagnostics, not E9 validity targets.

The deployed operator remains:

- memory size 72, 64 active slots;
- degree 5, temperature 0.10;
- shared key-whitening floor fraction 0.01;
- fixed head-only ZCA value direction and E8 dose per slot;
- attach boost 1.0; and
- hard gate threshold 0.95.

## Frozen wide query population

The sole data source is `scripts/data/counterfact.json`, SHA-256
`d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f`.
The complete E8 prompt order is the exclusion set.

Traverse CounterFact records in file order and consider only
`10000 <= case_id < 20000`. For each record:

1. render `requested_rewrite.prompt.format(subject)` as `exact`;
2. stably de-duplicate nonempty `paraphrase_prompts` and select the first;
3. stably de-duplicate nonempty `neighborhood_prompts` and select the first;
4. require the three selected strings to be nonempty and mutually distinct;
5. reject the record if any selected string appears in the complete E8 prompt
   order or in an earlier retained E9 row; and
6. retain `{case_id, exact, paraphrase, neighborhood}`.

Stop after 4,096 retained records. This yields:

- first retained case ID: 10,000;
- last retained case ID: 14,607;
- case-ID-list SHA-256:
  `c8a495c828e3fa4da443f7021a28cf587a978b51441326f62b3ec9fd5fc1d579`;
- selected-row SHA-256:
  `1902f1aa83e7214121e9d12db75c499b659d0077706b8373020748111643c7fa`.

Prompt-forward order is `exact`, `paraphrase`, `neighborhood` for each retained
row, preserving record order. The resulting 12,288 strings are all unique and
have zero overlap with the 337 E8 prompts.

- complete prompt-order SHA-256:
  `7f9c199bb1a402fa95bd46ffa879d01a7e36acb3f9921eb2ac2043b3343ff07a`;
- exact-list SHA-256:
  `ae8350d284f8bbf1db507bb6c9c939504e0a2c74bb14a44ebf0b6bdc44c5358c`;
- paraphrase-list SHA-256:
  `d55b79a22f0c70e0d903cbded95e6ad688a16a4c16eec49f3dbe101be6b0256a`;
- neighborhood-list SHA-256:
  `0b65a870be5f2b67bccabd0f4c1c299171a265d3b873ad3d8d29bc8e9827fdb8`.

Selection, text inspection, and hashing are permitted before registration. No
E9 prompt may be tokenized by the trunk, forwarded, scored against the memory,
or adapted before the single-use attempt receipt.

## Measurement

The frozen trunk evaluates all 12,288 E9 prompts in the registered order in
batches of 16, including only full 16-row batches. The adapter evaluates the
resulting native BF16 hiddens in consecutive 256-row batches. Batch boundaries
are part of the arithmetic contract and replay.

For each query record:

- pool index, source case ID, and kind;
- prompt and prompt SHA-256;
- native hidden dtype, shape, and full-tensor SHA-256;
- selected memory score and slot;
- selected E8 case ID and label;
- gate threshold and gate decision;
- native delta dtype, shape, SHA-256, and exact-zero flag; and
- adapted-hidden SHA-256 and exact bit identity with the native hidden.

The gate decision must be recomputed by the verifier as
`isfinite(selected_score) and selected_score >= 0.95`. The selected slot must
be active. A gate-closed row must have an exact-zero BF16 delta and an adapted
hidden tensor bit-identical to its native hidden.

Report exact counts by query kind, counts at score thresholds 0.10, 0.50, 0.90,
and 0.95, the maximum score and its first lowest-index row under ties, and the
20 highest-scoring rows ordered by descending score then ascending pool index.
These are descriptive diagnostics and cannot tune the gate or verdict.

As a non-vacuity validity control, evaluate all 64 spent E8 exact keys through
the same reconstructed memory. Every one must open the gate and select its own
slot. These anchor rows are not part of the E9 scientific denominator.

## Single-use execution and replay

Named outputs live under `results/e9_3b_staging/`:

1. `preflight.json`;
2. `execution_receipt.json`;
3. `attempt.json`;
4. `admission.json`;
5. `result.json`;
6. `verdict.json`.

All writes are atomic create-new. All outputs must be absent at preflight.
Registration binds committed sources, this protocol, tests, runtime, physical
node, model files, E8 evidence, CounterFact data, the complete pre-outcome E9
memory reconstruction, E9 population, commands, and absent outcomes.

Admission creates `attempt.json` with status `POOL_SPENT` before the first E9
model forward. A known post-attempt failure is durably
`IMPLEMENTATION_INVALID`. External interruption without known invalidity is
`INCOMPLETE`; this query pool cannot be rerun under amended code or bars.

Replay is a new Python process on the identical physical node/runtime. It
reconstructs the registered E9 memory and full E9 query population, repeats
both registered batch schedules, and must reproduce every memory tensor hash,
row, query tensor hash, score, decision, summary, and diagnostic exactly.

Registered commands, to be implemented and hash-bound before outcomes:

```text
python scripts/preflight_e9_3b.py
python scripts/register_e9_3b.py
python scripts/admit_e9_3b.py
python scripts/replay_e9_3b.py
python scripts/verify_e9_3b.py
```

## Binding validity bars

Any failure below is `IMPLEMENTATION_INVALID`:

- every protocol, source, dependency, model, E8 evidence, data, population,
  prompt, tensor, receipt, and result hash matches its registered value;
- software runtime and native arithmetic exactly match E8, and the physical
  node exactly matches the E9 preflight;
- model invariants, head-only ZCA basis, and source cleanliness pass;
- every reconstructed memory-receipt field except the three explicitly listed
  key hashes is byte-for-byte scientifically equal to frozen E8 before E9
  outcome access;
- the complete reconstructed E9 memory and key-whitening receipts are frozen
  at registration and reproduced exactly by admission and replay;
- all 64 E8 exact-key anchors open and select their own slots;
- exactly 12,288 exhaustive E9 rows are evaluated under the frozen batch
  schedules;
- every gate decision is implied by its finite selected score and threshold;
- every selected slot is active and maps to the reported E8 case/label;
- every gate-closed row has an exact-zero delta and bit-identical adapted
  hidden;
- summaries and top-score diagnostics recompute exactly from the rows; and
- new-process replay is exact.

## Binding scientific bar

E9 is `PASS_3B_WIDE_LOCALITY` only if every validity bar passes and zero of the
12,288 disjoint queries opens the E8 gate. This is exact integer arithmetic:
`gate_open_count == 0` globally and for each registered kind.

A scientifically valid nonzero count is `FAIL_3B_WIDE_LOCALITY`. There is no
partial pass and no post-outcome threshold adjustment.

## Claim boundary after PASS

A PASS licenses only:

> On the pinned frozen SmolLM3-3B-Base revision, registered A100/BF16 node and
> runtime, pre-outcome-frozen 64-slot E8-recipe HLM5 adapter, and deterministic
> 12,288-query CounterFact pool, no disjoint exact-style, paraphrase, or
> neighborhood query opened the exact-key gate; every gate-closed adapter
> output was bit-identical to its native hidden, and exact new-process replay
> succeeded.

It does not establish a population-wide false-positive probability, semantic
routing, paraphrase generalization, multi-token editing, generation quality,
PPL neutrality, cross-model transfer, trained-HLM scaling, attention
replacement, legal compliance, or commercial readiness.

# HLM5 E9: wide exact-key locality protocol

Date: 2026-08-11

Status: preregistration; no E9 query hidden, gate score, or adapter output has
been computed

Scope: HLM5 model research only. HLM-KB, HLM-Flow, Ingesto, semantic routing,
and legal or commercial claims are out of scope.

## Decision question

Does the 64-slot HLM5 adapter that passed E8 remain completely closed on a
much wider, disjoint CounterFact query surface while its registered exact keys
still open their own slots?

E8 established full-path efficacy and exact rollback on 64 fresh exact keys
plus 264 registered off-support controls. E9 changes only the query surface. It
reconstructs the exact E8 primary memory and scans 12,288 new queries without
changing keys, values, doses, gate threshold, or trunk.

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

E9 uses exactly the E8 registered runtime:

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

Reconstruct the E8 primary memory from its 64 exact-prompt hiddens, 18 key-
whitening prompts, frozen ZCA value directions, and registered per-case doses.
The reconstruction must reproduce the complete E8 candidate memory receipt,
including all key/value/alpha/mask and key-operator hashes, before E9 outcome
access.

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
Registration binds committed sources, this protocol, tests, runtime, model
files, E8 evidence, CounterFact data, E8 memory reconstruction, E9 population,
commands, and absent outcomes.

Admission creates `attempt.json` with status `POOL_SPENT` before the first E9
model forward. A known post-attempt failure is durably
`IMPLEMENTATION_INVALID`. External interruption without known invalidity is
`INCOMPLETE`; this query pool cannot be rerun under amended code or bars.

Replay is a new Python process on the identical node/runtime. It reconstructs
the E8 memory and full E9 query population, repeats both registered batch
schedules, and must reproduce every row, tensor hash, score, decision, summary,
and diagnostic exactly.

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
- runtime and native arithmetic exactly match E8;
- model invariants, head-only ZCA basis, and source cleanliness pass;
- the reconstructed E8 memory receipt is byte-for-byte scientifically equal to
  the frozen E8 candidate memory receipt before E9 outcome access;
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

> On the pinned frozen SmolLM3-3B-Base revision, registered A100/BF16 runtime,
> fixed 64-slot E8 HLM5 adapter, and deterministic 12,288-query CounterFact
> pool, no disjoint exact-style, paraphrase, or neighborhood query opened the
> exact-key gate; every gate-closed adapter output was bit-identical to its
> native hidden, and exact new-process replay succeeded.

It does not establish a population-wide false-positive probability, semantic
routing, paraphrase generalization, multi-token editing, generation quality,
PPL neutrality, cross-model transfer, trained-HLM scaling, attention
replacement, legal compliance, or commercial readiness.

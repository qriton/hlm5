# HLM5 E11: frozen 3B adapter cross-node portability protocol

Date: 2026-08-11

Status: preregistration; no E11 model forward or adapter outcome has been
computed

Scope: HLM5 model research only. HLM-KB, HLM-Flow, Ingesto, semantic routing,
and commercial or legal claims are out of scope.

## Decision question

Does the exact portable 1,024-slot tensor bundle that passed E10 reproduce its
complete positive, locality, and rollback record on a different Leonardo A100
node without reconstructing, recalibrating, retraining, or retuning anything?

## Bound E10 evidence

- verdict: `PASS_3B_CAPACITY_1024`;
- implementation commit:
  `f6086d655469ebea78e076924786d4978729c036`;
- evidence commit: `5e37272ecfcfbc9edb40ebe52d9f19c41b83a105`;
- source node: `lrdn1455.leonardo.local`;
- bundle raw SHA-256:
  `d8b437041c5f902691c1f9e801ef57cdc5bedb1ea7d07a22b77d41aa9c07a3c7`;
- admission raw SHA-256:
  `2c8b3a763a1209eb2098b679fa1fcaae9b5361d6c09da53b80dd6ac82c7dea86`;
- result raw SHA-256:
  `824a118add90b163114372e22f55d31d7503a3f90ae4579e9eccf7cf16adaf3e`;
- verdict raw SHA-256:
  `9e3aa70a01a8f43e39836fa93479beea17b6d1e8535c96c113c1209f6b48ce56`;
- frozen 1,024-tier stable-JSON SHA-256:
  `25f698e5b4569dd5c9556906bdeb7043ee32df2d57ae070b19efecc336c621fd`;
- bundle-manifest stable-JSON SHA-256:
  `6cd6abb5ebb275b1a6cd4af46f719c63f58814f1a2ee441667506277d47bbceb`;
- selected-index stable-JSON SHA-256:
  `b5ae24760e283e63e34459affb9173617311380ab02433dbb08feebe6abdf2de`;
- selected-case stable-JSON SHA-256:
  `e73c67a8a95f15e71844cf8450f924bea9452627030283013f85d633baafad09`.

## Frozen execution

Use the identical pinned SmolLM3-3B-Base revision, BF16 arithmetic, Python,
Torch, Transformers, CUDA, A100-SXM4-64GB device type, prompt populations,
batch sizes, and HLM5 adapter implementation as E10. The physical node must be
nonempty and must not equal `lrdn1455.leonardo.local`.

Load `results/e10_3b_evidence/adapter_1024.pt` with `weights_only=True` on CPU.
Verify its raw file hash and every tensor dtype, shape, and tensor hash against
the E10 admission before moving tensors to the registered GPU. Copy the stored
keys, values, and alphas into slots 0..1023 of a fresh 1,032-slot released
memory; activate exactly that prefix; restore labels only from the frozen JSON
receipt; and attach the stored key mean and transform. Do not forward any E11
prompt before `attempt.json` exists.

Forward the frozen 1,200 E10 exact prompts and complete 12,288 E9 locality
prompt order in the same BF16 batches of 16. Select the frozen 1,024 capacity
indices; do not recompute direct geometry, doses, directions, or key
whitening. Re-run the unchanged E10 positive scan, locality scan, and rollback
scan. The new cross-node measurement is compared field-for-field with the
frozen E10 1,024-tier object, including every native/adapted tensor hash, score,
slot, attention, gate, ordinary-head logit/hash/decision, summary, batch
coordinate, rollback projection, and post-removal state.

## Single-use chain

Named create-new outputs live under `results/e11_3b_staging/`:

1. `preflight.json`;
2. `execution_receipt.json`;
3. `attempt.json`;
4. `admission.json`;
5. `result.json`;
6. `verdict.json`.

Preflight binds committed sources/tests, protocol, E10 evidence and bundle,
model/runtime/device type, a physical node different from E10, and absent
outputs. Registration creates the pre-outcome receipt. Admission burns the
spent portability attempt before the first prompt forward. A known post-burn
implementation failure is `IMPLEMENTATION_INVALID`; external interruption
without known invalidity is `INCOMPLETE` and does not authorize a rerun.
Replay is a new process on the same E11 node/runtime and must exactly reproduce
the E11 admission.

Registered commands:

```text
python scripts/preflight_e11_3b.py
python scripts/register_e11_3b.py
python scripts/admit_e11_3b.py
python scripts/replay_e11_3b.py
python scripts/verify_e11_3b.py
```

## Verdict

`PASS_3B_CROSS_NODE_PORTABLE` requires:

- the E11 node differs from the E10 node while every other registered runtime
  and device field matches;
- the bundle raw bytes and every stored tensor exactly match E10;
- the entire reconstructed memory receipt exactly matches E10;
- the entire positive, 12,288-row locality, and rollback objects exactly match
  the frozen E10 1,024-tier object; and
- exact new-process E11 replay and a valid provenance chain.

Any valid numerical, routing, gate, output, hash, or rollback mismatch is
`FAIL_3B_CROSS_NODE_PORTABILITY`, not an implementation failure. No tolerance
or “close enough” substitution is admitted. E12 is authorized only after the
formal E11 pass.

## Claim boundary

A pass establishes byte- and field-exact portability of this one frozen
1,024-slot adapter across two registered Leonardo A100 nodes. It does not
establish portability to a different accelerator model, precision, software
stack, trunk, dataset, or semantic routing regime, and it adds no capacity or
generalization evidence beyond E10.

# Legacy G4 HLM5 hybrid 3B recovery contract

Date: 2026-08-11

Status: checkpoint and source provenance bound; latest-checkpoint GPU canary not
yet authorized or run.

## Recovery conclusion

The 3B hybrid is conditionally recoverable. It did not fail scientifically at
85.2% completion. The last successful allocation loaded the 412,000-step
checkpoint, trained through step 522,600, and timed out normally after 24
hours. It durably saved step 520,000 with validation perplexity
15.581647041956762. The two later attempts failed in SLURM task launch/prolog
before Python initialized, so they provide no evidence against the checkpoint
or model.

The exact legacy source is recoverable from repository commit
`b76bb14f2c245f20aa883f517b0497b1fa90f984`. The live Leonardo copies of the
trainer, model, resume validator, and hybrid launcher are byte-identical to
that snapshot. A prior 96-rank canary, job `47494090`, successfully loaded both
the then-current baseline and hybrid FSDP shards and ran an all-rank forward.
Subsequent jobs used the same repaired loader to advance the hybrid from step
36,000 to 520,000.

This evidence is sufficient to justify a new load-only canary. It is not yet
sufficient to spend roughly 15,000 allocation hours finishing the remaining
90,000 hybrid steps.

## Bound 520K state

- checkpoint: `/leonardo/home/userexternal/mdima000/HLM5/runs/g4_hybrid/ckpt_step520000`;
- marker step / payload step / world: 520,000 / 519,999 / 96;
- architecture: `huge3b`, dim 2,560, 32 layers, 32 heads, context 1,024;
- hybrid mechanism: `memory_layer=true`, `memory_size=256`;
- shard count / total bytes: 96 / 21,670,864,639;
- ordered shard-manifest SHA-256: `634a8f58e26e679323a637b4df2c726b8dd55e1964486009bf192fe8ad53f80b`;
- marker SHA-256: `6f7c425ff46a7edc8cf382d3ca66b4e547dc5b71b7c39a9698fac3d0a41f5014`;
- trainer SHA-256: `f74bf513719f8c3a110cbf45296889ad5914b3ca1cf7b5ba8538195d123d42f9`;
- model SHA-256: `e3080a257276b6d5b235a72cd9c7e18fb5a4d71e2a3fe8e30daebe5f029fd0d9`;
- resume validator SHA-256: `3acd55ffc8789bcfafa2e72fc2aad6127e758865043f93767e28e9514952dd8e`;
- historical hybrid launcher SHA-256: `7d43c087efefb0fe96ab3acc9429a4bd4f62cd739fd19d86461c277149fb4dd4`.

The ordered manifest hashes the `sha256sum` records for ranks 0 through 95,
including their fixed absolute paths. The canary also checks every rank name,
file non-emptiness, total bytes, marker, and source hashes before allocating
the model.

## Known caveat: optimizer continuity

These FSDP checkpoints are model-only. The trainer forcibly enables
`--no-save-opt` under FSDP, so each 24-hour continuation reconstructs AdamW
state. The learning-rate schedule resumes from the global step, but optimizer
moments do not. This was already true of the successful historical lineage;
it is not new corruption. It does mean that a full continuation cannot be
treated as mathematically identical to uninterrupted training.

The repair therefore uses a short stability rung before any completion run.

## Staged authorization gates

### R0 — read-only provenance: complete

Pass: 96/96 shards exist, aggregate and marker hashes are fixed, exact source
is in Git, last durable metrics are finite, and later failures occurred before
Python. All bars passed on 2026-08-11.

### R1 — load-only 96-rank canary: awaiting explicit compute approval

Launcher: `slurm/legacy_g4_hybrid_520k_canary.slurm`.

It requests 24 nodes / 96 GPUs for at most 30 minutes, loads the exact 520K
LOCAL_STATE_DICT shards at their required world size, executes one 16-token
forward per rank, and re-hashes the checkpoint afterward. It never calls the
trainer. Expected runtime is about one minute; the 30-minute wall is only a
hard ceiling.

Pass requires:

- all pre-load hashes and structural checks pass;
- payload world size is 96 and payload resumes at step 520,000;
- load mode is the known `strip-root-fsdp` compatibility path;
- every rank completes the historical `[1, 16, 65536]` shape-checked forward;
- post-load checkpoint hashes are unchanged;
- terminal marker is `REPAIR_CANARY_PASS`.

Any failure stops recovery. Do not patch the only checkpoint in place.

### R2 — 4,000-step stability rung: not yet authorized

Only after R1 passes, freeze a separate launcher capped at step 524,000 and a
two-hour wall. Preserve 520K; write a new 524K checkpoint. Pass requires exact
resume at 520K, no nonfinite events, a complete 96-shard 524K save, finite
validation at 522K and 524K, and no validation regression larger than 0.20
absolute PPL from the 15.58165 anchor. The rung is expected to consume roughly
700 allocation hours, with 1,536 as the two-hour hard maximum.

### R3 — matched no-tax recovery: not yet authorized

The baseline is only at step 484,000. A completed hybrid alone cannot establish
the old no-tax claim. Before interpreting quality, advance the baseline to the
same step under its own load/stability gates, then compare matched step, data,
schedule, and validation windows. The hybrid-minus-baseline delta—not either
standalone PPL—is the relevant product metric.

### R4 — finish or stop: not yet authorized

Only a stable R2 and matched R3 can license the remaining 86,000 hybrid steps.
Bind an allocation-hour ceiling, checkpoint cadence, retry policy, node-health
preflight, and scientific stop bar before submission. Infrastructure launch
failure may be retried on an unchanged registered job; model load, nonfinite,
or validation failure may not.

## Relation to the new frozen-trunk work

Do not transplant the E14 residual router: E14 validly showed that its
degree-five objective collapsed held-out key geometry. E13's full-Mahalanobis
readout direction and exact-key external memory remain valid components that
can later be evaluated on a consolidated legacy trunk. They do not repair or
validate this internal memory layer by themselves.

The legacy hybrid's value proposition must still be demonstrated by matched
ablation: native language-model quality at parity, a load-bearing memory
mechanism, and a task where that mechanism beats the same trunk without it.

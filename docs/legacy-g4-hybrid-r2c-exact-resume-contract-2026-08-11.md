# Legacy G4 HLM5 hybrid 3B exact-resume continuity contract (R2c)

Date: 2026-08-11

Status: ready to register. Implementation, local verification, remote import
preflight, and a no-allocation Leonardo scheduler estimate passed. No R2c
outcome has been accessed and no R2c job has been submitted.

## Question

Does the state-complete 522K checkpoint preserve the exact future training
trajectory across a process restart at the fixed 96-rank Leonardo topology?

R2b established that the repaired checkpoint contains loadable model, AdamW,
sampling-generator, CPU-RNG, and CUDA-RNG state. It did not establish that a
save/process-restart is observationally identical to uninterrupted execution.
R2c measures only that continuity property. It does not measure a memory gain,
a baseline tax, final model quality, or portability.

## Frozen source

- source checkpoint:
  `/leonardo/home/userexternal/mdima000/HLM5/runs/g4_hybrid/ckpt_step522000`;
- source payload step / next update: 521,999 / 522,000;
- world size: 96 ranks on 24 Leonardo nodes, four GPUs per node;
- shards / bytes: 96 / 43,294,668,998;
- ordered source aggregate SHA-256:
  `e32a2b0945e2197cc080d585bd16c7a33a817d63a6e5d96fb436127f8fdecae8`;
- source marker SHA-256:
  `8e83af00aed53b74e39aaee0ca090aef81a8232db9d0fd46aad1a16f82fcfe42`;
- stateful trainer SHA-256:
  `1f10b45c9d35bbaba346e07756c38edb681ea9bb8daee1c4c08ab4d2e58aa0c4`;
- model SHA-256:
  `e3080a257276b6d5b235a72cd9c7e18fb5a4d71e2a3fe8e30daebe5f029fd0d9`.

The registered execution receipt will bind the final launcher, this protocol,
the semantic comparator, the trainer, the model, the source marker and
aggregate, the exact command, and the allocated node list before either arm
executes a changed trajectory.

## Paired intervention

Both arms begin independently from the same immutable 522K source. Both use
the original 610K cosine horizon, batch one per rank, context 1,024, BF16 FSDP,
the trained memory layer at size 256, LR `2e-4`, warmup 2,000, gradient clipping
1.0, no nonfinite skips, and no checkpoint pruning. No restart LR ramp or RNG
offset is permitted because all required state now exists.

Both arms evaluate and save every 500 updates. Matching the midpoint save and
evaluation in both arms prevents checkpoint work itself from being unique to
the treatment.

- continuous arm: one process executes updates 522,000 through 522,999 and
  saves state-complete checkpoints at 522,500 and 523,000;
- split arm, segment one: a new process executes updates 522,000 through
  522,499 and saves at 522,500;
- split arm, segment two: another new process requires a complete-state resume
  at exactly 522,500, executes updates 522,500 through 522,999, and saves at
  523,000.

The arms execute sequentially inside one fixed allocation. All four new
checkpoints are written under
`/leonardo_work/AIFAC_L14_039/hlm5-g4-hybrid-r2c`; the 522K source remains in
place and is only linked read-only into each run directory. At least
260,000,000,000 free bytes are required before the attempt begins. No R2c
checkpoint may be deleted by the registered job.

## Exact semantic measurement

Raw `.pt` byte equality is not the bar because `torch.save` container bytes can
differ while tensor state is identical. The registered comparator initializes
the original 96-rank process group, loads the corresponding shard on every
rank, and recursively compares:

- every local model tensor and `ShardedTensor` local shard plus shard metadata;
- the complete AdamW state and parameter-group metadata;
- the sampling-generator, PyTorch CPU, and CUDA RNG tensors;
- payload step, world, schema, architecture metadata, and stored validation
  state.

Tensor equality is shape-, dtype-, layout-, and value-exact (`torch.equal`).
The comparator writes one create-new rank report and a create-new aggregate
report with canonical semantic SHA-256 digests. It must cover exactly 96 ranks.
The two marker JSON files and printed validation PPL at each matched boundary
must also be byte/numerically identical.

## Identification and verdicts

The midpoint is a pre-intervention reproducibility control. It is compared
after the continuous arm and split segment one have independently executed the
same first 500 updates, before the split checkpoint is reloaded.

| Condition | Formal verdict | Meaning |
| --- | --- | --- |
| source, implementation, runtime, save, load, report, or hash contract fails | `IMPLEMENTATION_INVALID` | no scientific result |
| external interruption occurs with no known invalidity or numerical failure | `INCOMPLETE` | no scientific result; retry needs a separately recorded unchanged attempt |
| either arm has a nonfinite event or OOM under the already demonstrated configuration | `FAIL_STABILITY` | valid failure of this bounded continuation |
| midpoint markers, PPL, or semantic state differ | `INVALID_REPRODUCIBILITY` | restart effect is unidentified; do not interpret the endpoint |
| midpoint is exact but final markers, PPL, or semantic state differ | `FAIL_CONTINUITY` | state-complete restart is not exact |
| midpoint and final state are exact on all 96 ranks, with all other bars passing | `PASS_CONTINUITY` | restart preserves this 500-update future on this fixed runtime/topology |

Known validity failure takes precedence over interruption. A valid
`FAIL_STABILITY` or `FAIL_CONTINUITY` burns this definition; it may not be tuned
and rerun against the same evidence. A PASS is bounded to the registered source,
software, Leonardo runtime, topology, and 500 post-restart updates.

## Promotion boundary

`PASS_CONTINUITY` licenses preparation of the next recovery ladder only. It
does not authorize that paid ladder. The next decision must separately bind:

- a step and allocation-hour ceiling;
- intermediate PPL and nonfinite stop bars enforced by the launcher;
- checkpoint cadence and retry policy;
- matched baseline recovery before any no-tax claim;
- final 96-rank consolidation and verified local preservation.

The R2c launcher has a two-hour hard ceiling. Based on R2b, the expected usage
is roughly 700--900 local allocation-hours; the absolute scheduler ceiling is
1,536. Submission requires a new explicit user approval after the final hashes
and dry-preflight result are shown.

## Pre-outcome verification record

Local verification on 2026-08-11 passed:

- `python -m pytest -q tests`: 237 passed, one unrelated SciPy/NumPy warning;
- targeted comparator/launcher suite: 13 passed;
- Ruff on the comparator and focused tests: passed;
- Python compilation and Bash syntax checks: passed.

The exact staged comparator, launcher, and protocol reproduced their local
hashes on Leonardo. Under `cineca-ai/4.1.1`, both the comparator and trainer
completed import/CLI smoke checks, the fixed work root was absent, and the
source remained untouched. Leonardo accepted the 24-node definition under
`sbatch --test-only` as estimator `51819906`; the identifier was absent from
both `squeue` and `sacct`, so it created no job and consumed no allocation.
The final protocol hash must be recomputed after this record and the exact
frozen pair must receive one last test-only check before registration.

# Legacy G4 hybrid 3B R2d one-update determinism contract

Date: 2026-08-11

Status: ready to register. Local verification, remote API/import checks, exact
staged-file hashes, and a no-allocation Leonardo scheduler estimate passed. No
R2d outcome has been accessed and no R2d job has been submitted.

## Question

At which exact boundary do two independent 96-rank launches from the same
state-complete 522K checkpoint first diverge?

R2c was invalid before its restart treatment: the two ordinary launches already
differed after 500 identical updates. R2d does not rerun R2c, change its bar, or
test the memory mechanism. It localizes the uncontrolled source with exactly
one update per launch.

## Frozen source and runtime

R2d uses the untouched R2b 522K source:

- 96 shards / 43,294,668,998 bytes;
- marker SHA-256
  `8e83af00aed53b74e39aaee0ca090aef81a8232db9d0fd46aad1a16f82fcfe42`;
- absolute-path ordered aggregate SHA-256
  `e32a2b0945e2197cc080d585bd16c7a33a817d63a6e5d96fb436127f8fdecae8`;
- payload step 521,999; next update 522,000;
- model, AdamW, sampling-generator, CPU-RNG, and CUDA-RNG state required on
  every rank.

Both launches run sequentially inside one 24-node/96-GPU allocation using the
same rank/world layout, legacy model, FSDP wrapping, BF16 mixed precision,
activation checkpointing, batch one per rank, FineWeb mmap, context 1,024,
original 610K cosine horizon, LR `2e-4`, and gradient clipping 1.0. No
determinism flag is changed in this diagnostic: it measures the live arithmetic
that invalidated R2c.

## Measurement

Each launch restores the source and executes exactly one ordinary update. It
writes no checkpoint. Every rank emits exact SHA-256 digests at these ordered
boundaries:

1. `loaded`: local model parameters, live AdamW state, sampling/CPU/CUDA RNG;
2. `sampled`: input and target tokens plus post-sampling RNG;
3. `forward`: scalar loss, a frozen 3-position × 256-vocabulary logits probe,
   and post-forward RNG;
4. `backward`: local gradients after FSDP backward/reduction;
5. `clipped`: local gradients and global clipping norm;
6. `stepped`: local model parameters, AdamW state, and RNG after `optimizer.step`.

The comparator requires exactly 96 reports from each launch and identical
rank-to-host/device/runtime topology. It identifies the earliest boundary with
any exact mismatch. Later differences are descriptive only because they are
downstream of the first one.

## Frozen classifications

| First mismatch | Classification | Consequence |
| --- | --- | --- |
| rank/host/device/runtime topology | `TOPOLOGY_MISMATCH` / `IMPLEMENTATION_INVALID` | no arithmetic diagnosis |
| loaded model, AdamW, or RNG | `LOAD_STATE_DIVERGENCE` | state-complete restore contract is incomplete or load is nondeterministic |
| sampled tokens or post-sample RNG | `SAMPLING_OR_RNG_DIVERGENCE` | data/RNG restoration is incomplete |
| loss/logits probe or forward RNG | `FORWARD_DIVERGENCE` | forward CUDA arithmetic is nondeterministic |
| gradients after backward/reduction | `BACKWARD_REDUCTION_DIVERGENCE` | backward kernel or NCCL reduction is nondeterministic |
| gradients/global norm after clipping | `CLIP_DIVERGENCE` | clipping reduction is nondeterministic |
| model/AdamW/RNG after step | `OPTIMIZER_STEP_DIVERGENCE` | optimizer update is nondeterministic |
| no mismatch after one update | `ONE_UPDATE_EXACT` | divergence begins after more than one update; register a step-count localization next |

Every non-topology classification is a `VALID_DIAGNOSIS`, not a pass or failure
of HLM5. An implementation/source/report error is `IMPLEMENTATION_INVALID`.
External interruption with no known invalidity is `INCOMPLETE`. Known invalidity
takes precedence over interruption.

## Decision boundary

R2d cannot authorize the completion ladder. A localized arithmetic boundary
licenses one separately preregistered repair:

- pin deterministic algorithms/topology if the divergence is arithmetic; or
- extend the checkpoint contract if loaded or sampled state differs.

If one update is exact, the next gate may increase the update count by a frozen
binary ladder without checkpoint writes. A tolerance-based functional claim is
allowed only as a new protocol; the exact R2c bar may not be weakened after its
outcome.

## Cost and authorization

The launcher uses a 30-minute hard ceiling on 24 nodes / 96 GPUs: at most 384
local allocation-hours. Expected use is approximately 50--100 local hours,
because it performs one load plus one update twice and writes only small JSON
reports. The job may not be submitted without a new explicit user approval
after implementation hashes and the no-allocation Leonardo preflight are shown.

## Pre-outcome verification record

Verification on 2026-08-11 passed:

- the full local suite: 248 passed with one unrelated SciPy/NumPy warning;
- focused R2c evidence and R2d implementation suite: 11 passed;
- Ruff, Python compilation, and Bash syntax checks;
- exact staged hashes on Leonardo;
- target `cineca-ai/4.1.1` imports for the probe, comparator, stateful FSDP
  context, legacy model, and activation-checkpoint API;
- absence of the fixed single-use R2d work root.

Leonardo accepted the 24-node definition under `sbatch --test-only` as
estimator `51823367`. A test-only estimator creates no job and consumes no
allocation. The final protocol hash must be recomputed after this record and
the exact frozen pair must receive one last test-only check before submission.

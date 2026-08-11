# Legacy G4 hybrid 3B R2c verdict: pre-restart reproducibility invalid

Date: 2026-08-11

Formal verdict: **INVALID_REPRODUCIBILITY**

## Result

Leonardo job `51821287` executed the registered continuous arm from 522K to
523K and the independently launched split arm from 522K to its 522.5K
midpoint. The midpoint is before the restart intervention. It failed all three
registered reproducibility controls:

- precise validation PPL: 15.520899957093135 continuous versus
  15.514838289543917 split, a split-minus-continuous difference of
  -0.006061667549217554;
- marker bytes: different because the precise stored PPL differed;
- semantic state: all 96 of 96 rank reports differed.

The semantic comparator covered the model, AdamW state, captured sampling/CPU/
CUDA RNG state, and payload metadata. Per-rank mismatch counts ranged from 134
to 623 (median pair 135/135). Reports retain at most 32 example mismatches per
rank; the largest absolute tensor difference in those retained examples was
`8.56444239616394e-4`. Aliased FSDP parameter views appear more than once, so
these counts are diagnostic entries, not unique parameter counts.

The launcher therefore stopped before the split arm's restart segment and
wrote `INVALID_REPRODUCIBILITY`. Job exit `2:0` is the registered invalid
terminal, not an infrastructure crash. Runtime was 26m26s, approximately
338.35 local allocation-hours at billing 768. The 522K source passed the
launcher's exact post-control aggregate check and remains unchanged.

## Interpretation

R2c does **not** say checkpoint restore is broken: the two arms diverged before
one of them was restored from the midpoint. It also does not implicate the HLM5
memory layer. The differences were small, distributed across the model, and
both midpoint PPL values improved on the 522K source marker value
15.575561649711549.

The live hypothesis is ordinary distributed numerical nondeterminism or
runtime state outside the checkpoint contract—BF16 kernel/reduction ordering,
CUDA library algorithm choice, NCCL reduction order, or a still-unrecorded
state source. That mechanism is not yet identified.

## Decision

Do not weaken the exact R2c bar after seeing this result, and do not authorize
the completion ladder. The next bounded gate is a two-launch, one-update probe
from the untouched 522K source that records exact digests at load, sampling,
forward, backward/reduction, clipping, and optimizer-step boundaries without
writing another 43 GB checkpoint. Its first divergent boundary will decide
whether to pin deterministic arithmetic or replace exact replay with a newly
preregistered restart-excess tolerance claim.

Evidence is under `results/legacy_g4_hybrid_522k_r2c/`; the formal hash chain is
`verdict.json`.

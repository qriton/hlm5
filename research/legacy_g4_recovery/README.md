# Legacy G4 recovery

This directory contains the bounded recovery implementation for the preserved
520K HLM5 hybrid checkpoint. The trainer is derived from the exact historical
model-only source (`f74bf5...`) and makes state continuity explicit.

Its state-complete checkpoints include, per rank:

- the FSDP model shard;
- the AdamW state owned by that fixed rank/world-size layout;
- the dedicated FineWeb sampling-generator state;
- the PyTorch CPU RNG state;
- the active CUDA RNG state used by dropout.

`--require-complete-state` fails closed if any of those are unavailable, and
`--stop-after-resume` provides a load-only proof before a checkpoint can be
promoted. The old 520K checkpoint cannot become exact retroactively because it
never stored Adam moments or CUDA RNG. Its one registered bridge therefore:

1. reconstructs the known 108,000-draw data-sampling segment from the final
   uninterrupted job (412K to 520K);
2. applies one frozen 2,000-step LR ramp from 0.1x to 1.0x;
3. writes a new state-complete 522K checkpoint without pruning any source;
4. requires PPL <= 15.77 and a fresh 96-rank optimizer/RNG reload.

A bridge PASS establishes a restartable anchor, not a completed model or a
no-tax claim. The next scientific step would still be a separately registered
exact-resume continuation rung.

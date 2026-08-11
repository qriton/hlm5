# Legacy G4 hybrid R2 evidence

Formal verdict: `NARROW_FAIL_STABILITY_AT_522K`.

Leonardo job `51809238` validly resumed the exact 96-shard 520K hybrid model,
trained with finite losses through logged step 522,000, and produced a binding
522K validation PPL of `15.79`. The registered printed ceiling was `15.77`, so
the necessary R2 bar had failed. The job was stopped at 26m52s instead of
spending another roughly 2,000 steps on a run that could no longer pass.

This was a narrow miss: `+0.20835` absolute / `+1.337%` relative to the
15.581647 anchor, and only `0.02` printed PPL above the conservative ceiling.
It is not evidence that the model is unusable or that the HLM architecture
failed.

The prior uninterrupted job `47988845` evaluated the same step 521,999 at
`15.59`. R2 reached `15.79` after loading the same 520K model weights but
resetting both Adam moments and the per-rank sampling-generator state because
the historical checkpoint is model-only. The measured `+0.20` same-step gap
therefore localizes the issue to restart continuity as a class, but does not
separate optimizer-state loss from data-stream reset.

No 524K checkpoint or marker was created. The original 520K marker and all 96
shards remained exact: 21,670,864,639 bytes, aggregate SHA-256
`634a8f58e26e679323a637b4df2c726b8dd55e1964486009bf192fe8ad53f80b`.

Job allocation was 24 nodes / 96 A100 GPUs for 1,612 seconds, approximately
343.89 allocation-hours. The scheduler state is `CANCELLED by 132583` after
the explicit scientific stop.

Evidence SHA-256:

- attempt: `b86f2ffc80baa5ce74e2bf6feb7ce530d34483f45c0eebd893a60f58b97d8cd5`;
- R2 train log: `6d3baf35b18d05a790c9ef00b06d1be6e8198cbb51edcb4be60a7261b8df07ad`;
- Slurm stdout: `84515f23996a5447a11ef37f6440ebea1217b4ad317119c7e5a0f577dfbde115`;
- Slurm stderr: `d64c886ffdd3d585c9a4138a4c9e2575828e1e7adfec9c58c38f0819792bcfbb`;
- uninterrupted comparator log:
  `1fddeb1b7259f70bb9936e595df427e2e84ecb33b25fd8b034e35986c0b1015d`;
- preserved 520K marker:
  `6f7c425ff46a7edc8cf382d3ca66b4e547dc5b71b7c39a9698fac3d0a41f5014`.

Next recovery work must address checkpoint continuity explicitly. Repeating
the same model-only resume or relaxing the bar after seeing this result is not
a valid repair.

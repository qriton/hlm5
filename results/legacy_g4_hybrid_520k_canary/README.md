# Legacy G4 hybrid 520K recovery canary

Result: `REPAIR_CANARY_PASS`.

Leonardo job `51803021` ran on 24 nodes / 96 A100 GPUs and completed with exit
`0:0` in 1m14s. The actual `srun` load/forward step took 21 seconds. The job
used launcher commit `286924672a38d93200f3733d75ff237542e2e25a`.

The canary passed every bound condition:

- 96/96 checkpoint shards existed and totaled 21,670,864,639 bytes;
- ordered shard-manifest SHA-256 was
  `634a8f58e26e679323a637b4df2c726b8dd55e1964486009bf192fe8ad53f80b`;
- the exact legacy trainer, model, validator, and marker hashes matched;
- the loader selected the registered `strip-root-fsdp` compatibility path;
- it reported zero missing and 33 known duplicate/internal unexpected keys;
- marker step 520,000, payload step 519,999, and world size 96 matched;
- the all-rank 16-token forward completed;
- every checkpoint hash reproduced after loading;
- terminal marker was `REPAIR_CANARY_PASS`.

The checkpoint is therefore loadable and operational under its historical
runtime. This does not authorize the 4,000-step stability rung or the remaining
training run.

Evidence SHA-256:

- stdout: `0aa436f1f5c9d0b6554a8e25f89b374f70f7e3c4ae2dcba68297719651f06b9b`;
- stderr: `2a0751033f6c791b9ff19c401f68a611fa47eb915ca0481253d8199ee4c8b06b`.

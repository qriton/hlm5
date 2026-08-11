# Legacy G4 hybrid R2b

R2b is the authorized, single-attempt warm-restart bridge from the preserved
model-only 520K checkpoint to a state-complete 522K checkpoint.

Leonardo job `51813699` completed successfully on 2026-08-11 in 30m21s. The
registered bridge produced printed validation PPL `15.58` against ceiling
`15.77`, wrote a 96-shard state-complete 522K checkpoint, and loaded its AdamW
and rank-local sampling/CPU/CUDA RNG state in a fresh 96-rank process.

The formal status is `PASS_BRIDGE`. This licenses a separately preregistered
exact-resume continuity rung; it does not license completion training, a
no-tax claim, or product promotion.

Evidence hashes:

- attempt: `c9a3f2761624595f5b0fde5328f7326af2a9d173c2c6424716205007eef370a5`;
- result: `ce36261cef7097463f164a07b7c21aa8cbbba21e0fd0b1183664eaf073033e9e`;
- train log: `1c1c31d5af3c1fbd02efaa59c12ee1d397cf35cbbda8526d1c90db79994d27c8`;
- marker: `8e83af00aed53b74e39aaee0ca090aef81a8232db9d0fd46aad1a16f82fcfe42`;
- stdout: `de9fb9347bc139a98a87bb0e0db6251ec62733b7fce25c66173f3f22455f8c9d`;
- stderr: `2a0751033f6c791b9ff19c401f68a611fa47eb915ca0481253d8199ee4c8b06b`.

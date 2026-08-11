# E13 terminal 3B readout-conditioning evidence

Formal verdict: `PASS_3B_READOUT_REPAIR_1024`.

Leonardo job `51802403` ran on `lrdn2723.leonardo.local`, distinct from the
E10-E12 nodes, and completed with exit `0:0` in 6m24s. The evidence was
produced from implementation commit
`c75a8b87808658d5bde52f54eaa5c7439619ed07` under protocol commit
`6d36509ce17c6d80053812ca5b9aa9700074c0ae`.

On the first 1,024 of 1,197 cases admitted by both frozen direction arms, the
ZCA-half baseline reproduced the E12 failure pattern at 1,023/1,024 strict
wins. Changing only the value direction to the preregistered full-Mahalanobis
operator produced 1,024/1,024 strict wins at K=1,024 with minimum margin 4.5.
It also passed 256/256 at K=256. All gates opened, all own slots won, and all
deltas were nonzero in both candidate tiers.

The candidate opened 0/12,288 locality gates; every closed output had an
exact-zero delta and bit-identical hidden. Exact rollback passed for all 1,024
positives and all 12,288 locality prompts. The portable tensor bundle and the
complete measurement reproduced byte-for-byte in a new process.

Raw SHA-256:

- `preflight.json`: `f1d5db00eab93665d4f5f8749c48d7bbb145de2d2c179fd73f0889b4367883ad`
- `execution_receipt.json`: `ba3a54b7b59655a4e47f0f5d71c133e002312d6b3c7af3e0245eb231b2f50e9b`
- `attempt.json`: `27fbb50aafa5d864378f5dc6846bb1268450ef598a3a434f9de0c229b142b56c`
- `final_mahalanobis_adapter_1024.pt`: `fa9305475714e63f656353789d2dd9ec359eb7dc3c371d50b60b9dafa5500a9e`
- `admission.json`: `2eea1f5925402a5e0846f2f99d3bd33fdafdfea57fcd633b762d8d0cbb010151`
- `result.json`: `717bee9ec4f9fe63967e0cb9ffaa32694b19209feb60a6cd83ababa78a82edab`
- `verdict.json`: `e430b6dd13d478048681aa7a41478c11d7c7538be5a4efbb7fd29b53e1938389`

This valid pass burns the population and ends the E10-E13 capacity line. It is
evidence for a frozen-trunk exact-key memory component, not a trained 3B HLM,
attention replacement, or semantic editing system.

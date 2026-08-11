# E14 frozen-3B router-training evidence

Formal verdict: `FAIL_3B_ROUTER_TRAINING`.

Leonardo job `51802878` ran on `lrdn2535` and completed with exit `0:0` in
5m27s. The valid single-use attempt used implementation commit
`67e2a6793634059d014358f8a5b7dd26b5b155c6` under protocol commit
`ed4082bf765b7420204798049a156d65ca9226b8`.

The frozen 3B trunk, 1,024-slot exact path, hard-zero locality path, rollback,
portable router bundle, and exact new-process replay all behaved as specified.
The learned routers failed the registered training and held-out paraphrase
bars:

| Arm | train own-slot | held-out paraphrase own-slot | paraphrase gates | exact strict | locality gates |
|---|---:|---:|---:|---:|---:|
| identity | n/a | 273 / 1,024 | 0 | 1,024 / 1,024 | 0 / 2,994 |
| cosine control | 246 / 256 | 159 / 1,024 | 0 | 1,024 / 1,024 | 0 / 2,994 |
| HLM degree-5 | 241 / 256 | 2 / 1,024 | 0 | 1,024 / 1,024 | 0 / 2,994 |

The failure is geometric rather than an infrastructure or readout failure.
The HLM router raised active-key coherence from 0.6653 for identity to 0.9632
and degree-5 maximum crosstalk from 0.1304 to 0.8288. It collapsed distinct
keys while fitting the 256 training pairs, so more training on the same
objective is not licensed. E13's exact-key capacity result remains intact.

Raw SHA-256:

- `preflight.json`: `a24eefef4a850e6643bc5e4f4c5d0cdfa13e28dd3bd7546766cc6ca584464437`
- `execution_receipt.json`: `bc366b0903755d2d77b404e6093abd2c7590611532d15789b21c83b7630e72e7`
- `attempt.json`: `d6789b3665971710e70f6ec1f83d8912b6bd11a5005af9a0b7b398e586db2ccd`
- `routers.pt`: `466d3dab74515e28e6685fd601dd1ff69731b2cbc69659405438ea6abd64a53c`
- `admission.json`: `bd91a892c5f5ff787de85eacb40179262f6690a398eaa27fa359eb6cd0309bd9`
- `result.json`: `e6337d24edada17b172ec86e875d9f1e4d4221c9c6988b14e74332bf7641fead`
- `verdict.json`: `389f543c54bd74abe0c9bedc55c2777076774122a7fd9419b97c7bf280da64c8`

This result licenses a new alignment-preserving repair experiment on a fresh
population. It does not license a rerun of E14, additional steps on the same
loss, 3B pretraining, attention replacement, or a product claim.

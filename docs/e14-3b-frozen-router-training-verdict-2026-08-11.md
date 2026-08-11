# HLM5 E14: frozen-3B router-training verdict

Date: 2026-08-11

Formal verdict: `FAIL_3B_ROUTER_TRAINING`

## Result

E14 validly falsified the proposed 262,144-parameter residual-router recipe.
The failure was not remote infrastructure, certificate validity, readout
capacity, exact-key memory, locality, rollback, or replay. Both learned arms
preserved 1,024/1,024 exact strict wins and opened 0/2,994 locality gates, but
neither met the frozen training or held-out paraphrase bars.

| Arm | train own-slot | paraphrase own-slot | paraphrase strict | paraphrase gate |
|---|---:|---:|---:|---:|
| identity | n/a | 273 / 1,024 | 5 / 1,024 | 0 / 1,024 |
| cosine control | 246 / 256 | 159 / 1,024 | 5 / 1,024 | 0 / 1,024 |
| HLM degree-5 | 241 / 256 | 2 / 1,024 | 5 / 1,024 | 0 / 1,024 |

All three arms passed 1,024/1,024 exact gates, own-slot selections, strict
ordinary-BF16 wins, and nonzero deltas. Every locality output was unchanged and
bit-identical to native. Rollback passed for every exact and locality row. The
frozen trunk hash was unchanged, and the serialized bundle and complete
measurement reproduced exactly in a new process.

## Diagnosis

The learned HLM key geometry collapsed. Its active-key coherence was 0.9632,
versus 0.6653 for identity, while degree-5 maximum crosstalk rose from 0.1304
to 0.8288. The cosine control degraded less severely but still reduced
held-out own-slot routing from 273 to 159. The HLM loss therefore amplified a
training-set alignment at the expense of separation between unseen keys—the
same alignment-versus-amplification distinction identified in the wider HLM
audit.

This is strong feedback against extending the schedule. By step 400 the HLM
arm still had only 241/256 training own-slot wins and the control 246/256;
neither met the prerequisite even before held-out evaluation. More steps on
the same objective are not a repair because the measured inter-key geometry
already moved in the wrong direction.

## What remains live

E13 remains a valid 1,024-slot exact-key result. E14 narrows the next semantic
routing experiment to an alignment-preserving key mechanism: either explicit
entity-relation canonicalization, which already has a winning record, or a
paired objective that constrains the off-diagonal key Gram instead of only
raising own-pair scores. A fresh experiment must use a new population and must
bind a separation/crosstalk bar before outcome access.

## Execution and evidence

- protocol commit: `ed4082bf765b7420204798049a156d65ca9226b8`;
- protocol SHA-256: `96528a33993e9877f9243c5b732fc2943f3bac03762e2d0129ae87f0459d2c3b`;
- implementation commit: `67e2a6793634059d014358f8a5b7dd26b5b155c6`;
- Leonardo job / node: `51802878` / `lrdn2535`;
- exit / elapsed / allocation: `0:0` / `00:05:27` / one A100 GPU;
- direct admission: 1,198/1,200; frozen selection: first 1,024;
- admission SHA-256: `bd91a892c5f5ff787de85eacb40179262f6690a398eaa27fa359eb6cd0309bd9`;
- replay result SHA-256: `e6337d24edada17b172ec86e875d9f1e4d4221c9c6988b14e74332bf7641fead`;
- router bundle SHA-256: `466d3dab74515e28e6685fd601dd1ff69731b2cbc69659405438ea6abd64a53c`;
- verdict SHA-256: `389f543c54bd74abe0c9bedc55c2777076774122a7fd9419b97c7bf280da64c8`;
- evidence directory: `results/e14_3b_evidence/`.

The claim boundary remains a pinned frozen SmolLM3-3B-Base trunk, a finite
CounterFact pool, single-token targets, transductively fit key whitening, and
the registered A100/BF16 runtime. It is not 3B language-model training,
attention replacement, semantic editing in general, or HLM-KB product
validation.

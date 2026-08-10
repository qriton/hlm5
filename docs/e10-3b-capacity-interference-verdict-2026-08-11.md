# HLM5 E10: 3B nested capacity and interference verdict

Date: 2026-08-11

Formal verdict: `PASS_3B_CAPACITY_1024`

## Result

The frozen HLM5 pre-head adapter preserved exact-key efficacy, own-slot
routing, wide locality, and exact rollback as active memory grew from 64 to
256 to 1,024 slots on the pinned frozen SmolLM3-3B-Base trunk.

The registered 1,200-row population produced 1,197 deployment-admitted
direct edits. The first 1,024 admitted rows in source order formed the fixed
capacity population. No case, dose, key operator, threshold, physical memory
size, or verdict bar changed after the E10 pool was burned.

| Active slots | gates / own slots / strict wins | min / median own attention | min target margin | max non-own score | key coherence | degree-5 max crosstalk |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 64 / 64 / 64 | 0.997148 / 0.997148 | 1.750 | 0.000003220 | 0.079721 | 0.000003220 |
| 256 | 256 / 256 / 256 | 0.988553 / 0.988555 | 1.750 | 0.005053 | 0.347298 | 0.005053 |
| 1,024 | 1,024 / 1,024 / 1,024 | 0.955582 / 0.955617 | 1.625 | 0.061559 | 0.572610 | 0.061559 |

Every positive delta was nonzero. At every tier, 0 of the complete 12,288
spent E9 locality prompts opened the gate; every locality delta was exact zero
and every output hidden was bit-identical. Removing the active prefix restored
exact-zero deltas and bit-identical hiddens for all tier positives and all
12,288 locality rows.

The independent new-process replay matched the admission exactly. The
1,024-slot portable CPU tensor bundle is 33,585,959 bytes with raw SHA-256
`d8b437041c5f902691c1f9e801ef57cdc5bedb1ea7d07a22b77d41aa9c07a3c7`.

## Interpretation

Interference is visible and well resolved: minimum own-slot attention declines
from 99.71% to 95.56%, while the strongest non-own degree-five score grows from
3.22e-6 to 0.06156. It is not yet binding at 1,024 slots. The registered
polynomial kernel retained the correct slot and enough value concentration for
every selected edit to remain a strict ordinary-head win.

This is evidence for a scalable inference-time memory component, not a trained
3B HLM language model. The trunk and head remain untouched; attention is not
replaced. The result covers exact single-token CounterFact prompts selected
inside one burned population. It does not establish paraphrase or multi-hop
generalization, factual truth, multi-token editing, generation quality,
population-wide false-positive rate, legal compliance, or capacity above
1,024.

## Execution and evidence

- protocol commit: `f73b657012f9b28593a92c521670390087c306a3`;
- implementation commit: `f6086d655469ebea78e076924786d4978729c036`;
- Leonardo job: `51801427`;
- node: `lrdn1455.leonardo.local`;
- state / exit / elapsed: `COMPLETED`, `0:0`, `00:04:51`;
- admission SHA-256:
  `2c8b3a763a1209eb2098b679fa1fcaae9b5361d6c09da53b80dd6ac82c7dea86`;
- replay result SHA-256:
  `824a118add90b163114372e22f55d31d7503a3f90ae4579e9eccf7cf16adaf3e`;
- verdict SHA-256:
  `9e3aa70a01a8f43e39836fa93479beea17b6d1e8535c96c113c1209f6b48ce56`;
- evidence directory: `results/e10_3b_evidence/`.

E11 cross-node portability is authorized only for this exact passing bundle.

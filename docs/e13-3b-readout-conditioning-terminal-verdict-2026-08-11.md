# HLM5 E13: terminal 3B readout-conditioning verdict

Date: 2026-08-11

Formal verdict: `PASS_3B_READOUT_REPAIR_1024`

## Result

The preregistered full-Mahalanobis value direction repaired the paired
ZCA-half baseline's sole strict-margin failure at 1,024 active exact-key
memories. Both arms used the same 1,024 cases, frozen trunk, keys, routing,
gate, dose rules, memory slots, locality population, and native BF16 head. The
only scientific intervention was the value-direction spectral power.

| Arm | Active slots | gates / own slots / strict wins | min target margin | locality gates |
|---|---:|---:|---:|---:|
| ZCA-half baseline | 1,024 | 1,024 / 1,024 / 1,023 | 0.0 | 0 / 12,288 |
| Full Mahalanobis | 256 | 256 / 256 / 256 | 5.0 | 0 / 12,288 |
| Full Mahalanobis | 1,024 | 1,024 / 1,024 / 1,024 | 4.5 | 0 / 12,288 |

All candidate positives produced nonzero deltas. Every locality output was
exact-zero and bit-identical to native. Exact rollback passed for all 1,024
positives and all 12,288 locality prompts. The portable 1,024-slot bundle and
every measurement field reproduced exactly in a new process.

The paired baseline miss occurred at candidate index 858 / CounterFact case
2,152: it selected its own slot with score 1.0000006 and 95.56% attention but
had zero ordinary-head margin. The full-Mahalanobis direction preserved the
identical route and raised that margin to 4.5. This is direct evidence that the
failure was readout conditioning rather than memory retrieval.

## Interpretation and stop rule

The E10-E13 ladder now establishes a finite 1,024-slot operating point for the
frozen 3B exact-key HLM5 component, exact portability within the pinned A100
stack, wide measured locality, and exact rollback/replay. E12's valid fail is
preserved: the ZCA-half recipe is not universally sufficient. E13 shows that a
frozen task-aligned Gram preconditioner repairs that specific composed
memory/readout wall without changing gain, keys, or the language-model trunk.

Per preregistration, this ends the capacity line. No additional capacity hunt,
ridge/dose/threshold tuning, or larger-trunk scaling is authorized by E13.

## Execution and evidence

- protocol commit: `6d36509ce17c6d80053812ca5b9aa9700074c0ae`;
- implementation commit: `c75a8b87808658d5bde52f54eaa5c7439619ed07`;
- Leonardo job: `51802403`;
- node / exit / elapsed: `lrdn2723.leonardo.local`, `0:0`, `00:06:24`;
- common admission: 1,197/1,200; frozen shared selection: first 1,024;
- admission SHA-256: `2eea1f5925402a5e0846f2f99d3bd33fdafdfea57fcd633b762d8d0cbb010151`;
- replay result SHA-256: `717bee9ec4f9fe63967e0cb9ffaa32694b19209feb60a6cd83ababa78a82edab`;
- bundle SHA-256: `fa9305475714e63f656353789d2dd9ec359eb7dc3c371d50b60b9dafa5500a9e`;
- verdict SHA-256: `e430b6dd13d478048681aa7a41478c11d7c7538be5a4efbb7fd29b53e1938389`;
- evidence directory: `results/e13_3b_evidence/`.

The claim remains narrow: a frozen-trunk external exact-key memory. It is not
semantic/paraphrase application, multi-token editing, factual correctness,
reasoning propagation, a population false-positive guarantee, a trained 3B
HLM, attention replacement, or legal compliance.

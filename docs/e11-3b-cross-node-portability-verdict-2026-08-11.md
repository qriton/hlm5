# HLM5 E11: frozen 3B adapter cross-node portability verdict

Date: 2026-08-11

Formal verdict: `PASS_3B_CROSS_NODE_PORTABLE`

The exact E10 1,024-slot tensor bundle was loaded on
`lrdn2661.leonardo.local`, distinct from E10's
`lrdn1455.leonardo.local`, without reconstructing keys, values, doses, or the
key operator. Every stored tensor matched its E10 receipt before the first
forward.

The cross-node run reproduced the complete frozen E10 1,024-tier object
field-for-field: 1,024 positive routes and ordinary-head wins, all attention
and score values, all native/adapted hidden and logit hashes, all 12,288
locality rows and summaries, and the complete rollback state. Independent
new-process replay on the E11 node was also exact.

This establishes portability only across two Leonardo A100-SXM4-64GB nodes
under the same pinned software and arithmetic stack. It does not establish
portability across accelerator types, precision modes, trunks, or datasets.

## Evidence

- E11 protocol commit: `8543e5a36c7220836e2ac75ed279b0694b598ae6`;
- E11 implementation commit: `7a43980ac910f4be9e2997a1eaef67063a8bc711`;
- E10 bundle SHA-256:
  `d8b437041c5f902691c1f9e801ef57cdc5bedb1ea7d07a22b77d41aa9c07a3c7`;
- Leonardo job: `51801493`;
- node / exit / elapsed: `lrdn2661.leonardo.local`, `0:0`, `00:04:36`;
- admission SHA-256:
  `bfaa56c885c5a52316bf77d0799fdaea479c1637db14f5a9067dac0f3971b5c4`;
- replay result SHA-256:
  `44edb635ce5b564928dac5d0a48823db5b53c60869112adc27d72d8da42f5879`;
- verdict SHA-256:
  `2bbba2db94c82693c87aa631330a2b84607cb862d234780a4bac7aaa3595c129`;
- evidence directory: `results/e11_3b_evidence/`.

E12 untouched confirmation is authorized for the exact E10 bundle and frozen
E10/E11 evidence chain.

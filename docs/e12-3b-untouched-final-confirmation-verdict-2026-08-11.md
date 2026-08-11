# HLM5 E12: untouched 3B final confirmation verdict

Date: 2026-08-11

Formal verdict: `FAIL_3B_UNTOUCHED_FINAL`

## Result

The unchanged E10 recipe was reconstructed on a fully disjoint 1,200-case
positive population and 12,288-prompt locality population on a third Leonardo
A100 node. K=64 and K=256 passed every registered efficacy, routing, locality,
rollback, bundle, and replay bar. K=1,024 missed the universal strict
ordinary-head bar by one row.

| Active slots | gates / own slots / strict wins | min / median own attention | min target margin | max non-own score | key coherence | degree-5 max crosstalk |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 64 / 64 / 64 | 0.997148 / 0.997148 | 14.000 | 0.000269 | 0.193224 | 0.000269 |
| 256 | 256 / 256 / 256 | 0.988551 / 0.988555 | 1.375 | 0.009011 | 0.389904 | 0.009011 |
| 1,024 | 1,024 / 1,024 / 1,023 | 0.955389 / 0.955617 | 0.000 | 0.187483 | 0.715471 | 0.187483 |

Every positive opened its gate, selected its own memory slot, and produced a
nonzero delta at every tier. Every one of the 12,288 fresh locality prompts
remained closed with an exact-zero delta and bit-identical hidden output.
Removing the memories restored exact-zero deltas and bit-identical hiddens for
all positives and all locality prompts. New-process replay reproduced the
admission exactly.

The sole failed row was candidate index 352, CounterFact case 362:
`Which position does Daniel Royer play? They play as`. The registered target
was ` linebacker` (token 48800). At K=1,024 the target and ` cornerback`
(token 68100) both had BF16 logit 35.75. The target remained the returned
argmax because it had the lower token ID, but the target margin was exactly
zero, so the preregistered strict-win bar correctly failed. The energy memory
itself did not misroute: its own slot was selected with score 1.0 and 95.56%
own-slot attention.

## Interpretation

E10's 1,024-slot pass was real and exactly portable across nodes in E11, but
it was not population-independent under the universal strict-margin bar. E12
establishes an untouched fully passing registered tier of 256 and an exact
1,023/1,024 result at 1,024. It does not locate the capacity boundary between
those registered tiers.

The failure is especially informative: routing and locality remained intact,
while the composed ordinary-head margin became singular on one row. Relative
to E10, K=1,024 key coherence rose from 0.572610 to 0.715471 and maximum
degree-five non-own score rose from 0.061559 to 0.187483. The next scientific
question is therefore conditioning of the target/competitor readout under
interference, not whether the energy memory found its own key.

No threshold, dose, whitening operator, population, or pass bar may now be
changed and described as E12. Any intermediate-capacity or readout-conditioning
study must use a new preregistered population and must preserve the E12 fail.

## Execution and evidence

- protocol commit: `34c25ca489314ab14ed1594c6fd8badd17bba704`;
- implementation commit: `b2e1caf47623b8f6082a6c24eedba6e8f4ffd991`;
- Leonardo job: `51801708`;
- node / exit / elapsed: `lrdn2974.leonardo.local`, `0:0`, `00:06:49`;
- admission SHA-256:
  `155c35b7dac2d3805faf6fff2c7c8b2df597d318e3e5246d6d2a2cda4caeb120`;
- replay result SHA-256:
  `6fb90a9bce3404cc421086b85dbfb02b0df0cb0a450bc51b10ce56200f1fcdd7`;
- bundle SHA-256:
  `bd409bcf7c0421a6cded00c18f5d0b555f1eb167f91a98b6042f3a2cbd2a3c63`;
- verdict SHA-256:
  `bea89e68cffc9004db51665cf351d7db6f5525ed433869acdfef45d910096661`;
- evidence directory: `results/e12_3b_evidence/`.

This remains evidence for a frozen-trunk exact-key memory component, not a
trained 3B HLM language model, attention replacement, semantic edit system,
or population-wide false-positive guarantee.

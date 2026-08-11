# HLM5 E10-E13: frozen-3B evidence ladder synthesis

Date: 2026-08-11

## What the ladder established

- E10: one registered population passed nested 64/256/1,024 exact-key memory,
  12,288-prompt locality, exact rollback, bundle export, and replay.
- E11: the exact E10 1,024-slot bundle reproduced field-for-field on a second
  Leonardo A100 node.
- E12: a fully disjoint population on a third node passed 64 and 256, while
  1,024 produced 1,023 strict wins and one exact target/competitor logit tie.
- E13: on a fourth node and another disjoint population, the paired ZCA-half
  baseline again produced 1,023/1,024 strict wins while changing only the
  value direction to full Mahalanobis produced 1,024/1,024, with 0/12,288
  locality gates, exact rollback, bundle export, and exact replay.

The mechanism is therefore real, portable within the pinned A100 stack, and
measured at 1,024 active exact-key memories when its value direction is aligned
to the ordinary head Gram. E12's valid failure remains: ZCA-half alone is not a
population-independent 1,024 recipe. E13 identifies and repairs the composed
readout-conditioning wall without changing keys, gain, gate, or trunk.

## Most useful diagnosis

At the E12 failure, memory routing still selected the correct slot with score
1.0; all gates opened, all own slots won, locality stayed exact, and rollback
stayed exact. What failed was the strict ordinary-head margin: ` linebacker`
and ` cornerback` quantized to the same BF16 logit. This is a composed
memory-plus-readout conditioning limit, not a retrieval miss.

E13 confirms that diagnosis. Full head-Gram preconditioning raised the paired
zero margin to 4.5 while leaving routing metrics identical. The capacity line
is now closed by preregistration; do not amplify the gate, rerun burned pools,
or start another capacity/scaling ladder from this result.

## Defensible claim now

On a pinned frozen public 3B trunk, HLM5 provides a portable external
polynomial-energy exact-key memory with exact rollback and zero observed gates
on three separate 12,288-prompt locality populations. Its registered,
head-conditioned value direction passed 1,024/1,024 simultaneous exact-key
memories on the terminal E13 population. The weaker ZCA-half direction remains
documented at 1,023/1,024 on two relevant 1,024-slot measurements.

No claim is made for semantic/paraphrase application, multi-token editing,
truth, reasoning propagation, capacity beyond the tested tiers, a trained 3B
HLM, attention replacement, or legal compliance.

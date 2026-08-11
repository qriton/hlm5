# HLM5 E10-E12: frozen-3B evidence ladder synthesis

Date: 2026-08-11

## What the ladder established

- E10: one registered population passed nested 64/256/1,024 exact-key memory,
  12,288-prompt locality, exact rollback, bundle export, and replay.
- E11: the exact E10 1,024-slot bundle reproduced field-for-field on a second
  Leonardo A100 node.
- E12: a fully disjoint population on a third node passed 64 and 256, while
  1,024 produced 1,023 strict wins and one exact target/competitor logit tie.

The mechanism is therefore real, portable within the pinned A100 stack, and
robust at 256 active exact-key memories on the two measured populations. A
universal 1,024-slot claim is rejected by E12. The registered tests do not
identify the boundary between 256 and 1,024.

## Most useful diagnosis

At the E12 failure, memory routing still selected the correct slot with score
1.0; all gates opened, all own slots won, locality stayed exact, and rollback
stayed exact. What failed was the strict ordinary-head margin: ` linebacker`
and ` cornerback` quantized to the same BF16 logit. This is a composed
memory-plus-readout conditioning limit, not a retrieval miss.

That distinction decides the next work. Do not amplify the gate or rerun the
burned population. A new bounded experiment should measure the target versus
strongest-competitor Gram geometry as active-set interference grows, with
registered intermediate tiers and an untouched population. Any repair should
act on the readout/edit direction while leaving the frozen trunk and locality
contract unchanged.

## Defensible claim now

On a pinned frozen public 3B trunk, HLM5 provides a portable external
polynomial-energy exact-key memory with exact rollback and zero observed gates
on two separate 12,288-prompt locality populations. It passed all registered
bars through 256 simultaneous memories on both measured populations; one of
two populations also passed 1,024, while the untouched confirmation produced
1,023/1,024 strict wins at 1,024.

No claim is made for semantic/paraphrase application, multi-token editing,
truth, reasoning propagation, capacity beyond the tested tiers, a trained 3B
HLM, attention replacement, or legal compliance.

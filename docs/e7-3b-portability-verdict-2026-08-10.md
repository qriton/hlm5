# HLM5 E7 Frozen-3B Portability Verdict

**Date:** 2026-08-10

**Formal verdict:** `FAIL_3B_PORTABILITY`

**Implementation validity:** PASS; zero validity failures

**Scope:** pinned frozen `HuggingFaceTB/SmolLM3-3B-Base`, single-token targets,
exact-key HLM5 adapter, no training, no attention replacement

## Outcome

E7 is a valid scientific failure, not a broken run. The adapter, gate, full
faithful memory path, and exact off-support identity transferred to the frozen
3B trunk. The preregistered all-direct-target certificate bar missed by one
native-precision tie:

| Measurement | Result |
| --- | ---: |
| Envelope | 60 keys × 1,200 targets |
| Mean reachable fraction | 0.8596528 (population SD 0.0089640) |
| Original-key targets certified reachable | 1,028 / 1,200 |
| Certified direct targets succeeding through ordinary BF16 head | 1,027 / 1,028 |
| Faithful facts admitted / refused | 11 / 7 |
| Admitted exact-key full-memory successes | 11 / 11 |
| False gates on refused exact keys + paraphrases + neutrals | 0 / 69 |
| Neutral logits bit-identical | 8 / 8 |
| Gate/locality rows with runtime evidence | 80 / 80 |

The sole direct miss was target id 922 (`" about"`) against predicted id 220
(`" "`). Its float64 certificate selected beta 217.078019 within `(L, U) =
(36.430569, infinity)` and recomputed a positive worst-case margin of 2.258567.
After casting the edited hidden state to BF16 and applying the ordinary tied
head, the target and competitor tied at margin 0; argmax selected id 220.

## What transferred

- The public-trunk pre-head adapter matched the released HLM5 hook in a locked
  synthetic equivalence test.
- Model identity held: 3,075,098,624 frozen BF16 parameters, 36 layers, hidden
  size 2,048, vocabulary 128,256, tied embeddings, and a bias-free head.
- The 60-key vectorized envelope matched an independent scalar reduction on all
  1,200 original-key targets.
- Every accepted faithful exact key opened the intended gate, selected its own
  slot, and produced the target through the full memory path.
- Every registered off-support query stayed closed, and all neutral logits were
  byte-identical to the frozen baseline.

The gate-matched static-logit control also succeeded on 11/11 admitted faithful
facts. That result is descriptive and does not rescue the failed direct bar.

## What did not transfer

The affine certificate is exact for the real-valued map

`W_head (h + beta v)`.

The deployed map measured here is instead

`BF16_head(Q_BF16(h + beta v))`,

where the hidden-state cast and final logit representation can collapse a
strict real-valued margin into a native tie. E7 therefore exposes a composition
error: exactness of the isolated float64 certificate is not exactness of the
composed deployment operator. The miss is only 1/1,028, but the registered bar
was universal, so the correct verdict is FAIL.

This does not measure a 3B co-trained memory or the no-tax claim. Those require
training and remain open.

## Next bounded experiment

Do not rerun E7 with a relaxed bar or tune beta on the spent 1,200-target pool.
The next experiment should preregister a finite-precision-aware admission rule
and validate it on a disjoint tokenizer target pool. Two legitimate candidates
are:

1. a conservative analytic lower bound that subtracts hidden-cast, matmul, and
   output-rounding error from every competitor margin; or
2. a two-part receipt that keeps the float64 geometric certificate but treats a
   strict ordinary-native-head replay as a required deployment admission check.

The first preserves an a-priori theorem if the numerical bound is sound. The
second is simpler and operationally safe, but it is a verified deployment check,
not a pure closed-form certificate. Either must refuse native ties, report the
coverage cost, keep `EPS = 0`, and pass 100% of admitted targets on the disjoint
pool without changing the 3B trunk or gate.

## Bound artifacts

- Protocol commit: `6a73bab5a37622b14fb30800fd92f298c479d781`
- Implementation commit: `436396deba0234987ec5a0a8b66dbda406a7beea`
- Preflight SHA-256: `93e95bfb21b2efc5c04b69c09b6ede17c07b1fca739cdfd37d6e391d672fbd19`
- Execution receipt SHA-256: `c89928493758258c83b12b0f067531f11f40f92549f4d7207acc16d72f8cebc5`
- Result SHA-256: `42174f0e8d3658e39a8a82b730844fd0c4e77161d4ebc81456243b645141c05c`
- Verdict SHA-256: `83c139aac4f25f98ac58dee5fd468f7a2bae920732a8f7587953033370dcf0ed`

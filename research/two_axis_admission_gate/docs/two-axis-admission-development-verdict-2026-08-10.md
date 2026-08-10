# SCA-2 Development Verdict — Two-Axis HLM5 Admission Transfer

**Date:** 2026-08-10

**Formal verdict:** `STOP_BEFORE_TARGET`

**Harness status:** `VALID` — 35/35 binding validity checks passed

**Replay status:** `EXACT_REPLAY`

**Target boundary:** preserved; neither the target-development population nor
official HWU64 fold 2 was tokenized, encoded, or scored

## Decision

Set aside the raw degree-five class-gap gate. Blind ZCA still exposed useful
HLM5 routes, but `s1 - s2` did not behave as a second admission axis. On the
disjoint source-audit bank, adding the registered margin changed none of the
candidate's decisions: it removed 0/18 supported admissions and 0/10
off-support admissions.

The candidate therefore failed three of six source-audit bars and correctly
stopped before target access. No threshold was changed and no target row was
used to rescue the mechanism.

## Frozen source-audit result

| Blind-ZCA bar | Measured | Required | Pass |
|---|---:|---:|:---:|
| Dual macro coverage | 0.1875 (18/96) | >= 0.20 | no |
| Dual macro correct-admission | 0.15625 (15/96) | >= 0.15 | yes |
| Dual selective accuracy | 0.83333 (15/18) | >= 0.65 | yes |
| Selective gain over absolute-only | 0.00000 | >= +0.05 | no |
| Correct-admission retention | 1.00000 | >= 0.60 | yes |
| Off-support false-admission rate | 0.10417 (10/96) | <= 0.05 | no |

The coverage miss is small in count but is not the main failure. The margin
provided exactly zero selective gain and left more than twice the permitted
number of false routes.

| Arm | Macro top-1 | Absolute coverage | Dual coverage | Dual correct-admission | Dual selective accuracy | Dual off-support FAR |
|---|---:|---:|---:|---:|---:|---:|
| raw | 0.52083 | 0.08333 | 0.08333 | 0.08333 | 1.00000 | 0.04167 |
| centered | 0.52083 | 0.07292 | 0.07292 | 0.07292 | 1.00000 | 0.05208 |
| diagonal standardization | 0.51042 | 0.11458 | 0.11458 | 0.11458 | 1.00000 | 0.07292 |
| spectrum-matched random basis | 0.51042 | 0.07292 | 0.07292 | 0.07292 | 1.00000 | 0.05208 |
| **blind ZCA** | **0.40625** | **0.18750** | **0.18750** | **0.15625** | **0.83333** | **0.10417** |

For all five geometry arms, dual and absolute-only decisions were identical
on both source-audit populations. Because the dual mask is a subset of the
absolute mask, the equal counts prove that the registered margin removed no
source-audit row in any arm.

## What worked

1. **The staged measurement contract worked.** Source calibration and audit
   ran first; failure left the separately registered target cache absent.
   Official fold 2 remained sealed.
2. **The result is valid and reproducible.** Every implementation, data,
   checkpoint, population, reader, numeric, ordering, and cache-isolation gate
   passed, and an independent invocation reproduced the scientific object and
   all 384 row records exactly.
3. **Frozen HLM5 hiddens still contain useful intent structure.** Raw keys
   reached 0.52083 top-1 over 12 unseen audit intents from three supports per
   intent. Blind ZCA admitted 18 rows, 15 correctly, for 0.83333 selective
   accuracy.
4. **Blind ZCA still moves the useful-coverage frontier.** Relative to raw, it
   increased admitted audit positives from 8 to 18 and correct admissions from
   8 to 15. That gain came with six additional off-support false routes, so it
   is useful geometry evidence rather than a deployable admission rule.

## What failed, mechanically

The absolute maximum and the proposed class gap were nearly the same statistic
under the degree-five reader. For positive cosine scores,

`s1 - s2 = c1^5 - c2^5`.

Unless the two leading cosines are very close, `c2^5` becomes negligible and
the gap approaches `s1`. The registered source-audit rows show exactly that
regime.

Post-result diagnostics computed only from the preserved row artifact are
descriptive and non-binding:

- Pearson correlation between `s1` and `s1 - s2` was 0.99999875 on audit
  positives and 0.99999982 on audit negatives.
- Among blind-ZCA absolute admissions, median `(s1 - s2) / s1` was 0.99985
  for audit positives and 0.99962 for audit negatives.
- Replacing the gap by the scale-free ratio and applying the same source
  calibration rule removed two audit positives but still removed 0/10 false
  routes. Coverage fell to 16/96, selective accuracy rose only to 14/16, and
  off-support FAR remained 10/96.

So this is not a threshold-tuning miss. The supposed second axis was
effectively a rescaled copy of the first, and the false routes were
unambiguous under the support bank rather than near-ties.

One additional source-cache-only diagnostic calibrated the absolute threshold
against the actual audit support bank using the fixed disjoint calibration
negative pool. It controlled audit FAR at 3/96, but retained only 7/96 audit
positives and 6 correct admissions. Bank-conditioned calibration repairs the
false-route side only by collapsing useful coverage; it does not rescue the
registered mechanism.

## Consequence and next bounded question

Do not encode the SCA-2 target group, do not sweep the two thresholds, and do
not relabel this as a near pass. Preserve blind covariance conditioning as a
measured component and retire the claim that an absolute degree-five class gap
adds transferable routing-risk information.

The next candidate must introduce genuinely new information. The narrow live
question is whether support-relative local density or a task-aligned canonical
address can distinguish a confidently wrong nearest class from a genuinely
in-class query. Any follow-up may use the now outcome-spent source cache for
design, but it must freeze the statistic and controls before touching the still
unencoded target group. A same-count proximity control and a conventional
embedding baseline remain mandatory.

## Provenance

- preregistration commit:
  `56a3cb1818f814d0a8cae589512027228b8a2723`
- frozen implementation commit:
  `0e60bab2a8ea7c410a230a5bd1a7914ed9e61f98`
- preflight SHA-256:
  `1daa251fcd8a0504f78f9a1cc5dd2334a8eb72f9f0e759948025589178d4d546`
- development registration SHA-256:
  `b94c093d024ade7d5dbd98c24d9a5f4979bcba0388c23438e986062d27dd284f`
- result SHA-256:
  `2f2336d6c003e459d8adf6d88e7a0c09aa36a30553f2879633ccb155daa06ed2`
- scientific-object SHA-256:
  `5b0e02673a0e3b53f6628aa7838719828787796c1bde6fec06a30505f4717c96`
- row JSONL SHA-256:
  `6dc4daead91f59ed529c8779cfe7898c61214bd78b33261245a816615bbe848c`
- exact replay receipt SHA-256:
  `6ba38471bff5095d14e1e108a62798f1d445db1b465291f9048d1bfa8eacb345`
- source hidden tensor SHA-256:
  `9c77843334cdc54450f6c861e2aaf29cec91662f88f847bec6f1337e49e20667`
- local source-cache file SHA-256 (generated, not committed):
  `dbf9bc07133c2029095ddad20a990586b73bf354d331dad4e035f9400d1be6f7`
- target-development cache: absent by construction

The result concerns a static, three-shot selective-routing probe over frozen
HLM5-136M final hiddens. It establishes no recurrence, Hopfield advantage,
model-native explanation, production routing, compliance, or scaling claim.

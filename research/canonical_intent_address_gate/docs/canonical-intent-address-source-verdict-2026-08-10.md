# CA-1 Source Verdict — HLM5 Canonical Intent Addresses

**Date:** 2026-08-10

**Formal verdict:** `CLOSED_SOURCE_ONLY`

**Validity:** 19/19 gates passed; exact cache replay passed

**Scope:** outcome-spent HWU64 source groups only. No target intent name,
target utterance, official test utterance, or off-support test utterance was
tokenized, encoded, or scored.

## Decision

Do not encode the untouched target population. A single deterministic intent
name is not a viable semantic address for the frozen HLM5-136M final-hidden
reader under the registered source bars.

The result closes this exact address construction:

```text
intent: {official_label.replace("_", " ")}
```

It also closes prompt, synonym, punctuation, threshold, degree, and geometry
sweeps on these spent source groups. Those would be post-result fitting, not a
new mechanism.

## Primary result

The primary raw HLM5 canonical reader passed only the prediction-balance bar
and failed the other eight scientific bars.

| Source-audit measure | Registered bar | Raw HLM5 | Pass |
| --- | ---: | ---: | :---: |
| Macro top-1 accuracy | >= 0.4000 | 0.1771 | no |
| Macro coverage | >= 0.2000 | 0.0104 (1/96) | no |
| Macro correct-admission rate | >= 0.1500 | 0.0000 (0/96) | no |
| Selective accuracy | >= 0.6500 | 0.0000 | no |
| Off-support false-admission rate | <= 0.0500 | 0.0833 (8/96) | no |
| Maximum predicted-intent share | <= 0.2500 | 0.2500 | yes |
| Top-1 gain over shuffled labels | >= +0.2000 | -0.0208 | no |
| Correct-admission gain over shuffled | >= +0.1000 | 0.0000 | no |
| Top-1 drop versus MiniLM | <= 0.1000 | 0.3958 | no |

The only admitted raw-HLM5 audit positive was wrong. Raw canonical labels also
scored slightly below the identical score matrix with the registered shuffled
label assignment: 0.1771 versus 0.1979 macro top-1. The canonical words
therefore did not supply a useful label-to-query alignment to this reader.

## Controls and descriptive arms

The pinned MiniLM control reached 0.5729 macro top-1 on the identical canonical
texts. Its 0.1667 coverage was perfectly selective but its 0.1354 off-support
false-admission rate remained too high. This is evidence that the label names
carry semantic information and that the HLM5 failure is not merely an
uninformative-label artifact; MiniLM itself is not promoted as a complete
open-set gate by this run.

No descriptive HLM5 transformation rescued the candidate:

| HLM5 arm | Top-1 | Coverage | Correct admission | Selective accuracy | FAR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw | 0.1771 | 0.0104 | 0.0000 | 0.0000 | 0.0833 |
| Centered | 0.2083 | 0.1458 | 0.0521 | 0.3571 | 0.2083 |
| Diagonal standardization | 0.2812 | 0.0938 | 0.0312 | 0.3333 | 0.1146 |
| Spectrum-matched random-basis ZCA | 0.2188 | 0.1667 | 0.0521 | 0.3125 | 0.2188 |
| Blind ZCA | 0.1875 | 0.0625 | 0.0104 | 0.1667 | 0.0417 |

Diagonal standardization produced the best HLM5 top-1, but it remained 11.9
points below the primary accuracy bar and admitted only 3/96 correct rows while
falsely admitting 11/96 off-support rows. Blind ZCA met the FAR bar but reduced
the useful side to one correct admission.

## Frozen three-support references

The prior SCA-2 source-audit result used three utterance supports per intent on
the same audit population. These figures are references, not recomputed CA-1
evidence:

| SCA-2 arm | Top-1 | Absolute coverage | Correct admission | Selective accuracy | FAR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw three-support | 0.5208 | 0.0833 | 0.0833 | 1.0000 | 0.0417 |
| Blind-ZCA three-support | 0.4062 | 0.1875 | 0.1562 | 0.8333 | 0.1042 |

Replacing support examples with one canonical name did not solve admission; it
destroyed much of the already-measured routing signal. The failure is therefore
not just a missing confidence statistic. The frozen HLM5 hidden geometry does
not align utterances to this natural-language class-name address strongly
enough for the required reader.

## What survives

This result does not weaken the validated exact-address lifecycle. Exact IDs
can still create, edit, deactivate, serialize/restore, reactivate, delete, and
reject off-support reads through K=4096. It instead clarifies the system
boundary:

1. semantic interpretation and canonicalization need an upstream task-aligned
   mechanism;
2. HLM5's measured value begins after an exact address exists: admission,
   certified dose, reversible lifecycle, and replay; and
3. the frozen HLM5 final-hidden reader should not be marketed as that semantic
   router.

That split is constructive. It keeps the genuine HLM5 memory/certificate result
and stops spending on another prompt or gain variation of a route that did not
identify its labels.

## Provenance

- Protocol commit: `b3bd4297fc76e8282b506736a3f5f3564513ffb9`
- Implementation commit: `726a2ab`
- Protocol SHA-256:
  `312c37dd3488c1bf7f296d13636c20586192f6b4a75f5bbd3df635eac5c67082`
- Registration SHA-256:
  `50528a1a3d1cf7ad4817d8c43aef7af693a01fff4e0b097f4a07d91d0c2dcc3a`
- Result SHA-256:
  `92e8af5a0d6f208006aad740e7ca5832d065a868dd31674fbb8b3e42aea4542a`
- Scientific SHA-256:
  `2e2b302abb5139c71c539cdf572293f44aa18fa260da07d82fa8d622c84946f0`
- Row JSONL SHA-256:
  `559fc66c6c26c3e6707d344c15473bc8bc84cdddc6e75f4d71987190cb27e4e9`
- Replay SHA-256:
  `e1d53ddb9920e9b6d578375803126f25eae9eff2c2cfe9c53a7dc6d8d8102129`
- SCA-2 reference result SHA-256:
  `2f2336d6c003e459d8adf6d88e7a0c09aa36a30553f2879633ccb155daa06ed2`

The result supports only a source-development claim about the exact frozen
template, models, populations, and readers above. It establishes no HLM-specific
semantic advantage, target/test performance, recurrence benefit, product
routing result, explanation, compliance, or scaling claim.

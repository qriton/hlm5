# HLM5 Support-Consensus Admission — Source-Only Feasibility Screen

**Date:** 2026-08-10

**Status:** `CLOSED_SOURCE_ONLY`

**Outcome boundary:** post-result analysis of the already-spent SCA-2 source
cache; target development and official HWU64 fold 2 remained unencoded

## Question

SCA-2 showed that the degree-five class gap was almost identical to the
winning score. This screen asks whether agreement from a second, distinct
support of the predicted class provides genuinely new admission information.

For each query, keep the registered top-slot prediction and identify the three
supports carrying that predicted intent. The consensus statistic is the
second-largest positive-cosine degree-five score among those three supports.
It cannot be supplied by the winning support alone.

This is an outcome-spent design screen, not a preregistered result. It uses the
frozen blind-ZCA operator, calibration/audit rows, and source hidden cache from
the valid SCA-2 run. No model forward pass or target access was performed.

## Fixed feasibility reference

The SCA-2 source gate required, over 96 audit positives and 96 audit negatives:

- coverage at least 0.20 (at least 20 supported admissions);
- correct-admission at least 0.15 (at least 15 correct admissions);
- selective accuracy at least 0.65; and
- off-support FAR at most 0.05 (at most 4 false admissions).

The screen does not turn those bars into a new formal verdict. They are kept as
the unchanged feasibility reference so a weaker post-result rule is not called
promising merely because it improves one metric.

## Source-audit screen

| Rule | Supported admitted | Correct admitted | Selective accuracy | Off-support admitted | Feasible |
|---|---:|---:|---:|---:|:---:|
| Registered absolute-only | 18 | 15 | 0.8333 | 10 | no |
| Registered absolute + raw class gap | 18 | 15 | 0.8333 | 10 | no |
| Absolute + normalized class gap | 16 | 14 | 0.8750 | 10 | no |
| Absolute + second-support, error-q90 | 10 | 9 | 0.9000 | 1 | no |
| Absolute + second-support, negative-q95 | 10 | 9 | 0.9000 | 1 | no |
| Absolute + second positive support | 17 | 14 | 0.8235 | 6 | no |
| Second-support negative-q95 alone | 12 | 9 | 0.7500 | 3 | no |

`error-q90` uses the registered higher 0.90 quantile of the second-support
scores on incorrectly routed calibration positives. `negative-q95` uses the
registered higher 0.95 quantile on calibration negatives. `second positive
support` is parameter-free and requires only that the second same-class cosine
be positive.

The second-support signal is real: the calibrated conjunction removed 9/10
false routes, whereas the registered class gap removed none. But the same rule
also removed 8/18 supported routes and 6/15 correct routes. Removing the old
absolute gate did not restore coverage. The parameter-free rule retained more
useful rows but still missed the coverage, correct-admission, and FAR
references simultaneously.

A separate bank-conditioned absolute threshold, calibrated with the fixed
negative pool routed against the actual audit support bank, admitted 7/96
audit positives (6 correct) and 3/96 audit negatives. It controlled false
routes only by collapsing coverage.

## Decision

Close second-support consensus as a standalone admission repair for the
current three-shot blind-ZCA bank. It is a genuinely independent statistic,
but the source evidence shows an unresolved precision/coverage trade rather
than a transferable gate.

Do not encode the untouched target group for this mechanism and do not sweep
thresholds on the audit rows. The next mechanism must add task information:
canonical intent addresses, explicit descriptions, or another task-aligned key
space with a matched conventional embedding control. That is the alignment
class already supported elsewhere in the HLM record; another scalar derived
only from the same nearest-support geometry is not justified.

## Provenance

- SCA-2 result scientific SHA-256:
  `5b0e02673a0e3b53f6628aa7838719828787796c1bde6fec06a30505f4717c96`
- SCA-2 row JSONL SHA-256:
  `6dc4daead91f59ed529c8779cfe7898c61214bd78b33261245a816615bbe848c`
- source hidden tensor SHA-256:
  `9c77843334cdc54450f6c861e2aaf29cec91662f88f847bec6f1337e49e20667`
- local source-cache file SHA-256:
  `dbf9bc07133c2029095ddad20a990586b73bf354d331dad4e035f9400d1be6f7`
- target-development cache: absent

All figures above are deterministic derivations from that source cache and
the merged SCA-2 implementation. They are exploratory source evidence, not an
independent confidence statement or a test result.

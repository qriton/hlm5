# CA-1 Protocol — HLM5 Canonical Intent Addresses, Source Screen

**Date fixed:** 2026-08-10

**Status:** fixed before any HLM5 or conventional-model encoding of canonical
HWU64 intent text

**Evidence class:** source-only, outcome-spent mechanism design; not an
independent development or test result

## 1. Question

Can a deterministic task-aligned address recover useful HLM5 routing and
admission where example-only nearest-support statistics failed?

SCA-2 and its support-consensus continuation exhausted scalar confidence
rules derived from the same three-example bank. CA-1 changes the information
given to the address: the class name itself. It does not change HLM5 weights,
train a classifier, or inspect an untouched target outcome.

## 2. Frozen boundary

Reuse the exact SCA-2 HWU64 manifest and source hidden cache:

- manifest scientific SHA-256:
  `43ca391013300dd88a76c7e4a17b33503cb6682e295d5eac4071a50387590fd0`;
- source hidden tensor SHA-256:
  `9c77843334cdc54450f6c861e2aaf29cec91662f88f847bec6f1337e49e20667`;
- source cache file SHA-256:
  `dbf9bc07133c2029095ddad20a990586b73bf354d331dad4e035f9400d1be6f7`.

Allowed intent groups are `calibration_supported` and `audit_supported` only.
Allowed query populations are calibration positive/negative and audit
positive/negative only. The `target_supported` intent names, target support
utterances, target development utterances, target test utterances, and
off-support test utterances must not be tokenized, encoded, or scored.

The source cache is already outcome-spent. Passing this screen authorizes only
the drafting and commit of a separate untouched-target preregistration.

## 3. Exact canonical address

For an official intent label `label`, construct exactly:

```text
intent: {label.replace("_", " ")}
```

Use the lowercase official label bytes. Add no period, synonym, manual word
split, example, template variant, or prompt sweep. Sort intent labels
lexicographically before encoding. There is exactly one canonical key per
intent.

Encode HLM5 addresses with the pinned HLM5-136M checkpoint/tokenizer and the
same final-token hidden extractor as SCA-2. Encode only the 24 allowed source
labels. Canonical-address hiddens are a separate immutable cache; the existing
SCA-2 utterance cache is read-only.

## 4. Primary and controls

Primary candidate:

- `hlm5_raw_canonical`: unit-normalized raw HLM5 query hidden and raw canonical
  HLM5 key.

Required HLM5 controls:

- `hlm5_raw_shuffled_labels`: identical raw key matrix and scores, with the
  key-to-label assignment permuted by CPU `torch.randperm`, seed `20260813`;
- centered, diagonal-standardized, spectrum-matched random-basis ZCA, and
  blind ZCA canonical keys, reported descriptively under the frozen SCA-2
  source operator;
- the frozen SCA-2 three-support raw and blind-ZCA source-audit results as
  references, never recomputed or promoted as new evidence.

The primary is fixed as raw before outcomes. A descriptive transformed arm
cannot substitute for a failed primary.

Required conventional control:

- `sentence-transformers/all-MiniLM-L6-v2`, local snapshot revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
- one normalized embedding for the identical canonical address text and one
  normalized embedding per allowed source query;
- cosine top-1 routing, with the same source calibration construction.

No network download or model substitution is permitted during the screen.

## 5. Reader and calibration

For HLM5, compute positive-cosine degree-five scores and choose the maximum
canonical key, breaking ties by lexicographic key order. Degree five is
top-1-equivalent to positive cosine. For MiniLM, use cosine directly and the
same tie order.

Calibrate each arm separately:

1. route the 96 calibration-negative queries against the 12 calibration
   canonical keys;
2. set the absolute threshold to the NumPy 0.95 quantile with
   `method="higher"`; and
3. admit only when the maximum score is strictly greater than the threshold.

Apply the unchanged threshold to the disjoint audit canonical bank, 96 audit
positives, and 96 audit negatives. There is no target-label fitting, margin,
threshold sweep, or per-class threshold.

## 6. Source feasibility bars

`SOURCE_SCREEN_LIVE` requires every primary raw-HLM5 bar:

1. audit macro top-1 accuracy `>= 0.40`;
2. audit macro coverage `>= 0.20`;
3. audit macro correct-admission rate `>= 0.15`;
4. audit aggregate selective accuracy `>= 0.65`;
5. audit-negative false-admission rate `<= 0.05`;
6. maximum predicted-intent share `<= 0.25`;
7. top-1 gain over the label-shuffled control `>= +0.20`;
8. macro correct-admission gain over the label-shuffled control `>= +0.10`;
9. raw-HLM5 macro top-1 is no more than `0.10` below MiniLM; and
10. all implementation, cache, prompt, permutation, reader, threshold, and
    target-isolation validity gates pass.

Otherwise the verdict is `CLOSED_SOURCE_ONLY`. Passing does not authorize
target encoding; it authorizes only a new preregistration commit.

## 7. Required evidence

Preserve:

- hashes of this protocol, implementation, dependencies, source cache,
  checkpoint, tokenizer, MiniLM snapshot files, and canonical-address cache;
- exact source intent names and canonical-text hashes;
- the registered permutation vector and hash;
- per-row truth, prediction, maximum score, threshold, admission, correctness,
  arm, and model family;
- calibration construction rates, audit metrics, all bars, and a bounded
  verdict; and
- an exact replay receipt from the immutable caches.

The result can support only a claim about deterministic intent-name addresses
in frozen HLM5-136M final-hidden geometry. It cannot establish recurrence,
Hopfield advantage, general semantic understanding, product routing,
explanation, compliance, or scaling.

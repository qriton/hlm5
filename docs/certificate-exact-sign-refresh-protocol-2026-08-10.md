# HLM5 Exact-Sign Certificate Refresh Protocol

**Date:** 2026-08-10

**Status:** frozen before implementation and before any refreshed model output

**Scope:** the six certificate/dosing artifact families identified by
`certificate-artifact-refresh-audit-2026-08-10.md`. This work changes no trunk
weights, prompts, target pools, facts, gates, training data, or model-forward
arithmetic. It does not reopen semantic routing, HLM3 scaling, HLM-KB, Flow, or
Ingesto.

## Question

When the published HLM5 certificate producers are made consistent with the
packaged exact-sign contract, which quantitative claims survive?

This is a provenance refresh, not a search for a better score. A changed number
is evidence. It must not be tuned back toward the release value.

## Registered inputs

The frozen HLM5 producers use:

- baseline checkpoint
  `models/hlm5_lm_baseline_fineweb_g3_final.pt`, SHA-256
  `3e1c94d28125c2f86f3eeca030db3610f2fa679512c29c6b6608b24fc363e0f1`;
- tokenizer `models/tokenizers/fineweb-65536-compat.json`, SHA-256
  `15993635191a1c5f1a5dc7aeaacbdf9a44a45d90abef954fc77b686f4fbbe588`;
- random seed 0, the existing prompts/facts, the sorted 1,200-token pool, the
  first 40 unreachable synthesis cases, the 60 multi-key prompts, and the 17
  faithful facts exactly as currently encoded.

The CounterFact producer uses:

- model id `gpt2-xl`, revision
  `15ea56dee5df4983c59b2538573817e1667135e2`;
- float32 model-forward weights, with no fp16 fallback;
- `scripts/data/counterfact.json`, SHA-256
  `d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f`;
- the first 1,000 usable records under the existing selection rule.

An input hash mismatch is `IMPLEMENTATION_INVALID`; it is never repaired by
editing metadata or silently selecting another input.

## Arithmetic boundary

The model forward remains in its registered native dtype. Immediately after a
hidden state and tied head are obtained, the certificate path must:

1. detach and cast `W` and `h` to `torch.float64`;
2. construct and normalize `v` in float64 with norm epsilon `1e-12`;
3. compute base logits, slopes, intercepts, ratios, bounds, margins, synthesis,
   and beta grids entirely in float64;
4. use `EPS = 0.0`, `b_j > 0` for lower bounds, `b_j < 0` for upper bounds, and
   `b_j <= 0 and a_j <= 0` for hard blockers; and
5. cast a certificate-dosed value back to the released head dtype only at the
   memory-injection boundary.

Whitening, gate scoring, and ordinary model logits remain on their released
path. The first-record head-identity check compares the native model logits to
the same native pre-head hidden; it is not replaced by a float64 surrogate.

Unbounded `U`/slack values are serialized as JSON `null`; non-standard
`Infinity`/`NaN` payloads are forbidden. Scientific JSON writes are atomic and
use `allow_nan=False`.

## Required contract on every output

Every result JSON contains `certificate_contract` with:

- schema `hlm5-certificate-contract-v1`;
- `eps: 0.0` and `arithmetic_dtype: "float64"`;
- this protocol commit;
- producer repository path and SHA-256;
- checkpoint and tokenizer paths and SHA-256 for HLM5, or model id/revision and
  CounterFact data path/SHA-256 for GPT-2 XL;
- model-forward dtype, PyTorch version, seed, and the producer-specific fixed
  sample contract; and
- hashes of any upstream artifact consumed by that producer.

The output cannot contain its own SHA-256. A separate refresh manifest binds
each completed payload hash, the preflight receipt, and the comparison report.

CounterFact JSONL rows carry the SHA-256 of the full run contract. Resume is
allowed only from a staging JSONL for which every retained row has the exact
current contract hash and unique `case_id`/`usable_index`. An old or malformed
row causes a fail-closed error; it is not dropped or retained silently. The
summary binds the final JSONL SHA-256.

## Staging and execution order

Existing tracked result files are evidence and remain byte-identical during the
run. Producers write to `results/certificate_refresh_staging/`.

Before the first model forward, a preflight receipt must bind this protocol,
all producer/helper/test hashes, input hashes, exact commands, environment,
device, and dtypes. Preflight may load/check inputs but may not compute a
refreshed certificate outcome.

Execution order is:

1. one-key envelope and synthesis;
2. beta-star dose;
3. synthesis flip verification;
4. 60-key envelope, consuming only the staged one-key artifact;
5. faithful 17-fact certificate dosing; and
6. CounterFact last, after its data hash and contract-bound staging/resume path
   pass locally.

No paper, README headline, figure, or tracked result is replaced until all
available stages validate. CounterFact absence may yield `INCOMPLETE`, but must
not invalidate the independently complete frozen-HLM5 refresh.

## Validity bars

All bars below are binding:

- the preflight and every output contract match exact registered hashes;
- producer source contains no float32 certificate boundary and reports
  float64 tensors at the boundary;
- target/fact/key counts remain 1,200 / 40 / 60 / 17 / 1,000 as applicable;
- multi-key scalar and vectorized results agree on the original key;
- all certified candidate doses lie strictly inside their reported open
  interval and have positive recomputed worst-case margin;
- the faithful arm's reported flip agrees with an ordinary model forward for
  every fact, and locality bit-identity is measured on all eight neutral
  prompts;
- CounterFact has 1,000 unique rows with ordered usable indices 0 through 999,
  a passing first-record native logit identity check, and a summary hash that
  matches the JSONL; and
- the staged comparison validator covers all seven refreshed primary payloads;
  the later promotion commit must extend `scripts/verify_artifacts.py` to assert
  those same payloads at their tracked locations before publication readiness.

Any failed bar is `IMPLEMENTATION_INVALID`, even if headline values look good.
An interruption without a known validity failure is `INCOMPLETE`.

## Registered comparison record

The pre-refresh anchors are descriptive, not targets:

| Artifact | Existing anchor |
| --- | --- |
| one-key envelope | reach 0.784; risks 888 safe / 38 narrow / 15 brittle / 259 unreachable |
| 60-key envelope | mean reach 0.7938, population SD 0.0221 |
| beta-star | 941 reachable; median beta 72.134; median worst margin 18.390 |
| synthesis flip | 40/40; median beta 20 |
| faithful dosing | global 9/17; naive 13/17; synth 17/17; locality 8/8 bit-identical |
| CounterFact | 1,000/1,000 reachable and head-reachable; 0 hard blockers |

The comparison tool records every headline delta plus row-level categorical
changes where rows exist. It emits one of:

- `VALID_REFRESH_STABLE_CLAIMS`: all validity bars pass and every published
  headline remains exactly true under its original rounding/count convention;
- `VALID_REFRESH_CHANGED_CLAIMS`: all validity bars pass but at least one
  published headline changes; all dependent prose/tables must be regenerated;
- `IMPLEMENTATION_INVALID`; or
- `INCOMPLETE`.

Matching old numbers is not a validity bar. If the refreshed claim changes, the
new number replaces the old one; no dtype, epsilon, norm constant, seed, sample,
or threshold may be swept after output access.

## Promotion rule

Promotion is a separate reviewed commit. It atomically replaces the six result
families, refresh manifest, verifier assertions, dependent analysis/figures,
and every cited number. The provenance audit must then emit
`READY_TO_PUBLISH`. Until that happens, the current `REFRESH_REQUIRED` warning
stays visible.

## Pre-outcome amendment record

- 2026-08-10: removed one duplicated `b` from the tokenizer digest. The prior
  text contained 65 hexadecimal characters and therefore could not be a
  SHA-256 value. The corrected 64-character value above is the digest of the
  registered tokenizer and matches the independent provenance audit. This was
  corrected before implementation commit, preflight, or refreshed model output.

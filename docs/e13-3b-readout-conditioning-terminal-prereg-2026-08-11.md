# HLM5 E13: terminal 3B readout-conditioning protocol

Date: 2026-08-11

Status: preregistration; no E13 model hidden, geometry, dose, routing, logit,
locality, or rollback outcome has been computed

Scope: final decision experiment for the frozen-3B exact-key HLM5 line.
HLM-KB, HLM-Flow, Ingesto, trained-HLM, legal, and commercial claims are out
of scope.

## Decision question and stop rule

On the same new untouched cases, does full head-Gram preconditioning of the
external memory value direction convert a 1,024-slot strict ordinary-head miss
into 1,024 strict wins while preserving own-slot routing, wide locality, and
exact rollback?

E13 ends this capacity line. Candidate PASS freezes the measured 1,024 result.
Scientific FAIL freezes 256 as the largest cross-population registered tier.
No later capacity hunting, dose/ridge/threshold tuning, or larger-trunk scaling
is authorized by this protocol.

## Bound diagnosis

E12 formal verdict `FAIL_3B_UNTOUCHED_FINAL` is bound by verdict SHA-256
`bea89e68cffc9004db51665cf351d7db6f5525ed433869acdfef45d910096661`.
At K=1,024 it measured 1,024/1,024 open gates, own slots, and nonzero deltas,
but 1,023/1,024 strict ordinary-head wins. The sole miss retained the target as
argmax while tying ` linebacker` and ` cornerback` at BF16 logit 35.75. E13 is
therefore a readout-direction intervention, not a key/gate amplification test.

## Frozen operators

Let `E` be the untouched BF16 unembedding converted to float32, `mu` its row
mean, and `C = E^T E / V - mu mu^T`, symmetrized before float32
`torch.linalg.eigh`. Clamp eigenvalues at zero. Let `m` be the median strictly
positive eigenvalue and `r = 0.01 m`. Normalize every direction in float64 by
`norm + 1e-12`.

- baseline `zca_half_raw_r1e2`: `(C+rI)^(-1/2) e_t`;
- candidate `mahalanobis_raw_r1e2`: `(C+rI)^(-1) e_t`.

Both were frozen members of the pre-outcome E7c candidate family. E13 changes
only the spectral power from 1/2 to 1. No centering, ridge search, head rewrite,
gate change, or post-outcome dose is permitted. Each arm receives its own
unchanged exact float64 affine geometry and ordinary-BF16 direct-admission dose
under the E7c/E10 rules.

## Frozen positive population

Traverse `scripts/data/counterfact.json` from case 0. Apply the E10
single-token/nonempty/unique exact-prompt rule. Exclude every complete E8, E9,
E10, and E12 positive/locality prompt order. Retain the first 1,200 rows.

- first/last case ID: 1,247 / 2,507;
- selected-case SHA-256:
  `97baae5bdf9f0bb983462dabb0b217d8903d62d9ef17ec79026e09de0dc4c86e`;
- case-ID SHA-256:
  `13dfd78f1760d5a3d682744d5fa6a3bb3b54a29d8fb4a8f4cfe92b0ac24263d4`;
- prompt SHA-256:
  `ba8ebb4d2a45e406d64172f35441434500d7cb16009e0b286d9ef0bfc4ac14c2`;
- target-ID SHA-256:
  `b10544132207124b63b7822346dcbd77a77bc71480dbe62a373717150dab63c8`;
- target-string SHA-256:
  `329774d462e8d4549e157ec76d0fe1b36e10a452a6ef8d8c14f9dce63efedcb5`;
- overlap with all prior prompt orders: zero.

Run both direct-admission arms on all 1,200 rows. `COMMON_ADMITTED` is iff both
arms have `deployment_admitted == true`; select the first 1,024 common rows in
source order. Fewer than 1,024 is a valid scientific FAIL. Every measured arm
uses exactly this shared selected index/order.

## Frozen locality population

Traverse from case 2,508. For each record require nonempty pairwise-distinct
exact, first unique paraphrase, and first unique neighborhood prompts. Exclude
all prior and E13-positive prompts, require global uniqueness, and retain the
first 4,096 cases / 12,288 prompts.

- first/last case ID: 2,509 / 18,941;
- row SHA-256:
  `e2bb44addf8f557ea47ffb97ddfa6de32768d1b977ecd22d314123fa79e61326`;
- case-ID SHA-256:
  `3f037b021cb8c3c03e5e2f8237b96b47e5d04902d9e1c87b18e7b5d0bcfb8e59`;
- prompt-order SHA-256:
  `91c16924d49ecdbded39348a75195c88b336d280f918c21eda8bb21429ea79ac`;
- exact/paraphrase/neighborhood SHA-256:
  `b9add2884e12f69a39f48faf1f7a4fad160dd0277fece9b5b9022e7a177b106d`,
  `039527409f9e5b5f306cfe2b668d20376a3119e089c38c92d4a9619063a46070`,
  `6bd882fe2996689b515dec917c236b0c656f503fe957b98c8ebc7e8f0c024d93`;
- measured-row identity SHA-256:
  `15e769d70f2e02129b77062424685425a91bde0d14f65e68956be6cc458e19e5`;
- overlap with every prior/E13-positive prompt: zero.

## Frozen memory measurements

Compute one shared key whitening from all 1,200 exact hiddens plus the same 18
E8 calibration prompts, floor fraction 0.01. Every memory has 1,032 physical
slots, degree 5, temperature 0.10, boost 1.0, and gate threshold 0.95. Keys,
selected indices, active slots, batching, native BF16 head arithmetic, and
locality prompts are matched across arms.

Measure exactly three objects:

1. baseline ZCA-half, K=1,024;
2. candidate Mahalanobis, K=256;
3. candidate Mahalanobis, K=1,024.

For each object record every positive route/head result, complete 12,288-row
locality scan, and exact rollback. Export candidate K=1,024 as create-new
`final_mahalanobis_adapter_1024.pt`. New-process replay must reproduce every
measurement field and tensor byte exactly.

## Verdicts

`PASS_3B_READOUT_REPAIR_1024` requires candidate K=256 and K=1,024 each to
have K/K open gates, own-slot selections, strict ordinary-head wins, and
nonzero deltas; 0/12,288 locality gates with exact-zero/bit-identical outputs
globally and per kind; exact rollback; valid portable bundle; exact replay;
and baseline K=1,024 to have fewer strict wins than the candidate. This supports
paired repair attribution.

If candidate passes all absolute K=1,024 bars but baseline also has 1,024
strict wins, emit `PASS_3B_1024_REPLICATED_NO_REPAIR_ATTRIBUTION`. This supports
the candidate's finite-population efficacy but not a causal repair claim.

Any candidate absolute miss emits `FAIL_3B_READOUT_REPAIR_FINAL` and freezes
the prior robust 256 claim. Known implementation/provenance failure is
`IMPLEMENTATION_INVALID`; external interruption with no known invalidity is
`INCOMPLETE`. A valid FAIL burns the population and ends the line.

## Runtime and evidence

Use one Leonardo A100-SXM4-64GB under the E10-E12 pinned software/arithmetic
stack. The physical node must differ from E10, E11, and E12 nodes. Create-new
preflight, execution receipt, attempt, bundle, admission, result, and verdict
must be hash-chained. The attempt precedes the first E13 forward. No registered
outcome file may be overwritten.

## Claim boundary

A pass is finite-population evidence for a frozen-trunk external exact-key
memory. It is not semantic/paraphrase application, multi-token editing, truth,
reasoning propagation, population-wide false-positive rate, a trained 3B HLM,
attention replacement, cross-accelerator portability, or legal compliance.

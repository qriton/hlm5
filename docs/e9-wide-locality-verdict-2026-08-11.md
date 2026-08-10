# HLM5 E9 Wide Exact-Key Locality Verdict

Date: 2026-08-11

**Formal verdict:** `PASS_3B_WIDE_LOCALITY`

**Implementation validity:** PASS; exact new-process replay

**Scope:** pinned frozen `HuggingFaceTB/SmolLM3-3B-Base`, pre-outcome-frozen
64-slot E8-recipe HLM5 adapter, 12,288 deterministic disjoint CounterFact
queries, registered A100/BF16 physical node and runtime, single-token values,
no training, and no attention replacement.

## Decision

The adapter stayed completely closed on the registered wide query surface:

| Registered measurement | Result |
| --- | ---: |
| Disjoint queries | 12,288 / 12,288 |
| Exact-style gates opened | 0 / 4,096 |
| Paraphrase gates opened | 0 / 4,096 |
| Neighborhood gates opened | 0 / 4,096 |
| Exact-zero BF16 deltas | 12,288 / 12,288 |
| Adapted hiddens bit-identical | 12,288 / 12,288 |
| Rows scoring at least 0.10 | 0 / 12,288 |
| Maximum selected score | 0.00011861914390465245 |
| Hard gate threshold | 0.95 |
| Positive E8 anchors open | 64 / 64 |
| Positive E8 anchors select own slot | 64 / 64 |
| Positive E8 anchors have nonzero delta | 64 / 64 |
| New-process replay | exact |

The maximum score was a neighborhood prompt from CounterFact case 12,363. It
selected slot 18 but remained about 8,008 times below the hard gate threshold.
The maximum exact-style score was `0.00009375385707244277`; the maximum
paraphrase score was `0.0000016642054561089026`.

## Why this matters

E8 proved efficacy and exact rollback through the full released adapter, but
its off-support set contained only 264 rows. E9 expands the finite locality
surface by about 46.5 times and makes it harder: 4,096 exact-style prompts,
4,096 paraphrases, and 4,096 neighborhood prompts, all unique and disjoint from
the 337 E8 prompts. Zero gates opened in every class, while all 64 positive keys
remained live.

This is strong evidence that the released exact-key gate is not merely passing
because the original locality set was tiny. The separation is also not a
knife-edge threshold result: no row reached 0.10, while the gate remained fixed
at 0.95. Because closed gates return an exact-zero delta, locality here is an
ordinary-forward identity property, not an approximate KL or tolerance claim.

## Portability correction

Outcome-blind job `51801037` stopped before registration or E9 prompt forward
when one reconstruction missed the historical E8 key hashes. The protocol was
amended before outcome access to freeze the reconstructed keys and physical
node, permit drift in only the three key hashes, retain exact E8 equality for
every other memory field, and require exact admission/replay.

In the final valid job, even that narrow exception was unnecessary: active-key,
key-mean, and key-transform hashes all matched historical E8 exactly. The
formal claim remains the narrower preregistered E8-recipe claim; the failed
preflight remains useful feedback that hash-only numerical artifacts are a
fragile portability boundary.

## Claim boundary

This does not estimate a population-wide false-positive probability and does
not convert the exact-key memory into semantic routing. Paraphrases correctly
stayed closed; they were locality probes, not generalization successes. The
result does not establish factual truth, multi-token editing, generation
quality, PPL neutrality, cross-model transfer, a trained 3B HLM trunk,
attention replacement, legal compliance, or commercial readiness.

## Provenance

- initial protocol commit: `580261859c90fd0d7bb743fe413981423c083888`;
- amended pre-outcome protocol commit:
  `b16c93199fac5d47c4f29cc64cac147517ee62d1`;
- implementation commit: `82e744c76055bff9a4207c1d191deccc38b3ab3f`;
- Leonardo job: `51801213`, exit `0:0`, elapsed 3m12s, node
  `lrdn2661.leonardo.local`;
- result SHA-256:
  `4248a926676b53c7eab005bc87f894d8e7a1277e618135872e1e3ff30ddad0b7`;
- verdict SHA-256:
  `a025a56b541423610659ea52e40414fd5cad7454bf4687aabf4991a21e9d90b8`;
- complete chain: `results/e9_3b_evidence/`.

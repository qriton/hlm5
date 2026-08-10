# HLM5 E8 Fresh Full-Path ZCA Verdict

Date: 2026-08-11

**Formal verdict:** `PASS_3B_FULL_PATH_ZCA`

**Implementation validity:** PASS; exact new-process replay

**Scope:** pinned frozen `HuggingFaceTB/SmolLM3-3B-Base`, 64
preregistered exact-key CounterFact cases, fixed head-only ZCA values, released
HLM5 multi-slot pre-head memory, registered A100/BF16 runtime, single-token
targets, no training, and no attention replacement.

## Decision

The fixed value-side ZCA operator survived the full released HLM5 path. All 64
fresh cases were non-baseline-correct. Raw target-row values were directly
deployment-admitted for 29/64. The preregistered ZCA values were admitted for
64/64, gaining 35 cases and losing none of the raw-admitted cases.

Every ZCA-admitted exact key then:

- opened the degree-5 hard gate;
- selected its own memory slot; and
- strictly predicted the target through the untouched ordinary BF16 head.

The primary memory opened zero false gates on all 264 registered off-support
queries: 128 paraphrases, 128 neighborhood prompts, and eight neutral prompts.
Every gate-closed output was bit-identical to baseline. Removing all 64 active
slots restored an empty memory, exact-zero deltas, and bit-identical baseline
logits on all 337 registered prompts.

| Registered measurement | Result |
| --- | ---: |
| Edit-eligible cases | 64 / 64 |
| Raw deployment admissions | 29 / 64 |
| Fixed-ZCA deployment admissions | 64 / 64 |
| ZCA gain / raw admissions lost | +35 / 0 |
| Admitted exact gates open | 64 / 64 |
| Admitted exact own-slot selections | 64 / 64 |
| Admitted full-path target successes | 64 / 64 |
| False gates on registered off-support rows | 0 / 264 |
| Gate-closed outputs bit-identical | 264 / 264 |
| Post-removal outputs bit-identical | 337 / 337 |
| New-process replay | exact |

## Why this matters

E7c had established a readout-geometry result on one hidden state: ZCA changed
the feasible value directions. E8 establishes the missing composed result on
fresh prompts. The same fixed operator works after per-prompt certificate
admission, shared key whitening, multi-slot storage, degree-5 retrieval, the
hard gate, and the native 3B output head. The correction therefore is not just
a probe-side geometric artifact; it is usable by the released HLM5 adapter in
the registered exact-key regime.

The raw-versus-ZCA difference is large in this fresh population: 29 versus 64
deployment admissions with identical trunk, cases, key operator, memory class,
gate, and native head. This supports the program's alignment/preconditioning
diagnosis: changing the value direction unlocked the existing readout channel;
amplifying the raw direction was not required.

## Claim boundary

This is a strong bounded adapter result, not a new 3B foundation model. It does
not establish factual truth, semantic routing, paraphrase generalization,
multi-token editing, generation quality, perplexity neutrality, cross-model
transfer, a co-trained HLM trunk, attention replacement, legal compliance, or
commercial readiness. Paraphrases were registered as off-support locality
probes and correctly stayed closed; they were not generalization successes.

## Provenance

- protocol commit: `02f5186fefa88e9488f0635a5276ce0a9be5999d`;
- implementation commit: `3ee23ba8a1451f3231254fe7f7f63b4e3422468d`;
- Leonardo job: `51800915`, exit `0:0`, elapsed 3m49s;
- result SHA-256:
  `ec064b4d38c1abb9d88f99bd1c138b01cbfe4f9b15dbdf28edbe415477e678b6`;
- verdict SHA-256:
  `5e376c4c065ecf7e7602d3e81984e6425175876bc763ffd2a427684a6b9d7deb`;
- complete chain: `results/e8_3b_evidence/`.

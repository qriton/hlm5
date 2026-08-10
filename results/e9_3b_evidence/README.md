# E9 3B wide exact-key locality evidence

This directory preserves the complete byte-identical receipt chain from
Leonardo job `51801213` (`lrdn2661.leonardo.local`, A100-SXM4-64GB, exit 0,
3m12s, 2026-08-11). The registered implementation commit is
`82e744c76055bff9a4207c1d191deccc38b3ab3f`; the amended pre-outcome protocol
commit is `b16c93199fac5d47c4f29cc64cac147517ee62d1`.

The formal verdict is `PASS_3B_WIDE_LOCALITY`. The frozen 64-slot adapter
opened zero gates across 12,288 unique CounterFact queries disjoint from the
complete E8 prompt set: 4,096 exact-style prompts, 4,096 paraphrases, and 4,096
neighborhood prompts. Every delta was exactly zero in BF16 and every adapted
hidden was bit-identical to its native hidden. The maximum selected score was
`0.00011861914390465245`, versus the preregistered hard threshold `0.95`; zero
rows reached even the lowest descriptive threshold `0.10`.

The non-vacuity controls remained live: all 64 spent E8 exact keys opened the
gate, selected their own slots, and produced nonzero deltas. A new Python
process reproduced the complete memory receipt, all 64 anchors, all 12,288
query rows, every tensor hash, score, decision, summary, and diagnostic exactly.

An earlier outcome-blind preflight job, `51801037`, failed before registration,
the burn marker, or any E9 prompt forward because one reconstruction did not
match the historical E8 key hashes. The protocol was amended before outcome
access to freeze the reconstructed keys and physical node while retaining exact
E8 equality for every non-key field. In the final valid run, all three allowed
key hashes also reproduced the historical E8 receipt exactly, so the exception
was not exercised.

The result licenses only the bounded claim in `verdict.json`: selectivity of
this pre-outcome-frozen, exact-key, single-token HLM5 adapter on the pinned
SmolLM3-3B-Base revision, registered node/runtime, and deterministic finite
query pool. It does not establish a population false-positive probability,
semantic routing, paraphrase generalization, factual truth, multi-token
editing, generation quality, PPL neutrality, cross-model transfer, a trained
3B HLM trunk, attention replacement, legal compliance, or commercial
readiness.

SHA-256:

- `preflight.json`: `2fdba053f1cb58027801e0d890b2ed86c438206b60b195a8a9d48ba60d58b4bc`
- `execution_receipt.json`: `708fbc110b1c348ea742deb7070a0c349c2df3aa22b804d6dc47864777e4967c`
- `attempt.json`: `bf359cf5c99b82f08b5552af562bc44faea4c5cc0f3d3df05abb7a7a27e5e637`
- `admission.json`: `1d67111e3d7d3bba8d59cdd7d04b2197ed380ebce4b3ec00c96982a36a94551e`
- `result.json`: `4248a926676b53c7eab005bc87f894d8e7a1277e618135872e1e3ff30ddad0b7`
- `verdict.json`: `a025a56b541423610659ea52e40414fd5cad7454bf4687aabf4991a21e9d90b8`

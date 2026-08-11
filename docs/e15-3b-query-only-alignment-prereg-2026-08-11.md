# HLM5 E15: frozen-key query-only alignment gate

Date: 2026-08-11

Status: preregistration; only outcome-blind prompt enumeration and hashing have
been performed. No E15 router has been trained, and no fresh E15 prompt has
been encoded or scored.

Scope: one bounded repair after the valid E14 failure. The pinned public 3B
trunk, E13 value/readout operator, E14 whitening operator, exact memory keys,
memory degree, temperature, gate, and dose remain frozen. HLM-KB, HLM-Flow,
Ingesto, full-trunk training, attention replacement, and broad semantic claims
are out of scope.

## Decision question

Can a router trained only on spent E14 training pairs align incoming
paraphrases to a fixed exact-key space, generalize to fresh paraphrases of
disjoint facts, and preserve the exact-memory and locality contract?

E14 transformed both queries and stored keys. Its degree-five key coherence
rose from 0.6653 to 0.9632 and maximum crosstalk rose from 0.1304 to 0.8288;
held-out own-slot routing collapsed to 2/1,024. E15 removes that failure mode by
construction: the stored key matrix is never passed through the router. This
is an alignment experiment, not an amplification or schedule extension.

## Bound prior evidence

- E13 verdict: `PASS_3B_READOUT_REPAIR_1024`, SHA-256
  `e430b6dd13d478048681aa7a41478c11d7c7538be5a4efbb7fd29b53e1938389`;
- E14 formal verdict: `FAIL_3B_ROUTER_TRAINING`, SHA-256
  `389f543c54bd74abe0c9bedc55c2777076774122a7fd9419b97c7bf280da64c8`;
- E14 admission SHA-256:
  `bd91a892c5f5ff787de85eacb40179262f6690a398eaa27fa359eb6cd0309bd9`;
- E14 replay result SHA-256:
  `e6337d24edada17b172ec86e875d9f1e4d4221c9c6988b14e74332bf7641fead`;
- E14 router/whitening bundle SHA-256:
  `466d3dab74515e28e6685fd601dd1ff69731b2cbc69659405438ea6abd64a53c`.

Only `key_mean` and `key_transform` are inherited from the E14 bundle. Their
respective tensor SHA-256 values are
`7e91d5342dcf05cc432d0e188eafabb5cd67b889803005b3fdad8f4622d26533`
and
`a22988b5d922106f205dd6901fd2926ae301d6e7f40b77cdb8d8b3e429a8a29a`.
No failed E14 router parameter is reused.

## Frozen populations and contamination boundary

Reconstruct the exact E14 train, evaluation, and locality rows from the pinned
CounterFact file and E14 contract. Reconstruct the full E8–E14 spent-prompt
set. For each row, traverse `paraphrase_prompts` in source order after stable
de-duplication and take the first string that is nonempty, differs from the
exact prompt, is absent from the spent-prompt set, and has not been selected by
an earlier E15 row.

This deterministic traversal yields 2,454 pairwise-unique fresh prompts and
zero overlap with the 46,707 distinct spent prompts. The combined fresh-row
SHA-256 is
`1f22fe845528960e9f1b4f42bba4689cf8521912edb95e8bb584e6b9d254e5a9`.

### Spent training support

Train only on E14's 256 exact/first-paraphrase pairs, case IDs 2,508–3,415.
Their result is already spent and supplies no E15 scientific evidence. E15
must not train on any fresh prompt.

### Fresh same-fact transfer check

The unused paraphrase for each of the 256 training facts is held back until
the router passes the spent-data envelope and is serialized.

- row SHA-256:
  `f63c54edb3a99de818613b9c5f157dd13751149cc44486c01ec96c1bbec16db7`;
- query SHA-256:
  `357b8bc53088f11f4046e7d0f9c41d4c3b2eec303e5c675075bee03eaa22d673`;
- case-ID SHA-256:
  `b0657598f2cde50b0c82d11f61136e6cf20b785dd87da15c952d072e02004bae`.

### Fresh disjoint-fact evaluation

The unused paraphrase for each of the 1,200 E14 evaluation candidates is held
back. Candidate case IDs remain 3,420–8,678.

- row SHA-256:
  `8411d96c13b0e1d565e2794a80a513503e38eab3ef64fe5c85bdfc9f4e1ee206`;
- query SHA-256:
  `6ec5f724864bc081cc6b82c38ab2d5e233fb30fd8b612b1cf2b1c9f2cbb405a7`;
- case-ID SHA-256:
  `b7e37abf71506f490a9803541c00d488ef5b3130125126233e8c000e68437fd4`.

Reuse E14's first 1,024 direct admissions. The selected-index SHA-256 is
`9860f7919aa0032630c59f79a17ac6852f7afebd258582ca48ffa33f8946f952`.
The active fresh rows have:

- first/last case ID: 3,420 / 7,995;
- row SHA-256:
  `6e7d03c90a2b227c0108d36a064d2c20cf3da694312e0b99f637fd483c07beee`;
- query SHA-256:
  `23bd5053681e4f3ae94db5442edff65783651d73e33c375dd4eb0a05348834aa`;
- case-ID SHA-256:
  `28354380ae06be5aede0880fb3e042b7851866a484d114d11468148d7056d604`.

The exact addresses, whitening fit, and direct-admission selection are reused
and outcome-spent. E15 evidence is therefore fresh with respect to query
phrasing, not an independent replication of exact-key capacity.

### Fresh locality

Use the unused paraphrase from every E14 locality fact: 998 prompts, case IDs
405–21,917.

- row SHA-256:
  `e614ef5a1a5ea58ea86f8e3eb647cac91e469b271d9ebd4011b102c8495600df`;
- query SHA-256:
  `1b159757daed192a8591a9c29d486ab9171e3c02d7cea2607b036c4ad23d10af`;
- case-ID SHA-256:
  `ae512eaf6d80bcd8ce0acd65cb499378baa9442c8f33b080ae260f952b52c353`.

## Frozen memory and candidate

Recompute native final-token BF16 hiddens under the E14 runtime and whiten
with the bound E14 `key_mean` and `key_transform`. Unit-normalize in float32.
The 1,024 active evaluation exact keys and their E13 full-Mahalanobis values
are fixed and must reproduce E14 before training. The physical memory remains
1,032 slots with 1,024 active prefix slots, degree 5, temperature 0.10, boost
1.0, and gate threshold 0.95.

The sole learned candidate is the E14 rank-64 residual map

`R(q) = unit(q + (q A) B)`,

with CPU seed 1,401, `A ~ Normal(0, 0.02)`, and `B = 0`. It has 262,144
float32 parameters. It is applied to queries only. Stored keys and values are
never transformed.

Train full-batch on the 256 spent E14 first-paraphrase queries against the 256
fixed corresponding exact keys. Use cosine own-slot cross-entropy with
temperature 0.02 for exactly 400 AdamW steps: learning rate 3e-3, betas
0.9/0.999, epsilon 1e-8, weight decay 1e-4, and gradient clip 1.0. There is no
early stopping, checkpoint selection, loss sweep, degree sweep, or gain sweep.

At inference, first compute the unmodified degree-five score against the fixed
keys. If its maximum is at least 0.95, bypass the router exactly. Otherwise
route with `R(q)` and apply the unchanged E13 degree-five memory. This branch
protects the already-valid exact path rather than asking a semantic adapter to
rewrite a query the memory can already address.

Arms:

1. `identity`: no router;
2. `query_only`: the frozen trained router and exact-path bypass;
3. `canonical_address_oracle`: replace a query by the exact key identified by
   its dataset `(subject, relation_id)` tuple.

The oracle is an interface-positive control only. It assumes the semantic
parse and cannot contribute to the scientific PASS or support a natural-
language routing claim. It represents the contract an upstream structured
router must satisfy before HLM5 exact memory begins.

## Two-stage outcome firewall

Training and the following admission checks use spent prompts only. Before any
fresh prompt is tokenized or encoded, serialize and hash the candidate and
require:

1. finite loss, gradients, and parameters, with nonzero parameter update;
2. 256/256 spent training paraphrases select their fixed own key;
3. the stored evaluation key tensor and its complete Gram SHA-256 are
   bit-identical before and after training;
4. on the 256 spent training queries, routed-query maximum off-diagonal cosine
   is no more than the identity value plus 0.10; and
5. routed-query Gram effective rank
   `(trace(G)^2 / trace(G^2))` is at least 80% of the identity value.

Failure of any item emits the valid scientific verdict
`FAIL_QUERY_ALIGNMENT_ENVELOPE` and stops before fresh 3B inference. Passing
the envelope licenses exactly one evaluation of the frozen fresh populations.

## Fresh measurements and bars

Measure identity and query-only arms on the same fresh prompt order. For every
row record native and routed scores, selected slot, own-slot status, gate,
ordinary-head strict status, margin, and full-logit hash. Reuse E14's exact
rows only as regression controls. Run exact rollback and new-process replay,
and require the trunk parameter hash to remain unchanged.

`PASS_3B_QUERY_ONLY_ALIGNMENT` requires all of the following:

- query-only on the 1,024 fresh disjoint-fact queries: at least 308 own-slot
  selections, at least 103 open gates, and at least 103 ordinary-head strict
  wins;
- query-only improves over identity by at least 52 rows on own-slot, open-gate,
  and strict-win counts separately;
- on the 256 fresh same-fact queries: at least 128 own-slot selections and at
  least 64 open-gate strict wins;
- exact regression remains 1,024/1,024 for gate, own slot, strict win, and
  nonzero dose;
- fresh locality remains 0/998 open gates, every delta exactly zero, and every
  output hidden bit-identical to native;
- stored key/Gram hashes remain exact, rollback is exact, the trunk is
  unchanged, all values are finite, and new-process replay is exact.

The absolute primary own-slot bar is 30% rounded up. Gate/strict bars retain
E14's 10% threshold, and paired gains retain E14's five-percentage-point row
bar. Aggregate PPL is not a promotion signal.

If validity passes but any scientific bar misses, emit
`FAIL_3B_QUERY_ONLY_ALIGNMENT`. Known implementation/provenance failure is
`IMPLEMENTATION_INVALID`; external interruption without known invalidity is
`INCOMPLETE`. A valid FAIL freezes learned residual routing on this trunk and
moves semantic interpretation upstream to a structured router; it does not
retract E13 exact memory.

## Execution and claim boundary

Implementation must be independently reviewed before registration. Run at
most one registered one-A100 Leonardo attempt under the exact E14
software/arithmetic stack. The source/test receipt and single-use attempt must
exist before training; the serialized adapter must exist before fresh-query
inference; all outputs are create-new and hash-chained.

A pass would show that one query-only adapter aligned fresh CounterFact
paraphrases to a frozen 1,024-key HLM5 store on one pinned 3B trunk while
preserving the measured exact/locality contract. It would not establish broad
semantic parsing, factual truth, unknown-key generalization, HLM-specific
router superiority, a trained 3B foundation model, legal compliance, or
product readiness.

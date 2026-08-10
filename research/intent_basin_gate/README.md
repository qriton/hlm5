# HLM5 Class-Balanced Intent Basin Gate

**Outcome (2026-08-10): valid `STOP_BEFORE_TEST`.**  Raw-centroid routing
scored `0.9179` macro accuracy, one settle step scored `0.9066`, and eight
steps scored `0.7662`.  All validity gates passed; five of six scientific bars
failed.  The official test split remains sealed.  See the
[`development verdict`](docs/intent-basin-development-verdict-2026-08-10.md)
for the frozen numbers, correction lineage, digests, interpretation, and stop
rule.  The registered implementation is preserved in parent commit
`3dd844fb7b2804bf249e257b856b9c57632038f0`.

This package tests one bounded follow-up to the failed ParaRel density settle:

> Can repeated degree-five state dynamics improve natural intent routing when
> every intent has equal support, common-mode geometry is removed, and movement
> is constrained to a training-derived trust region?

The official CLINC150 validation split is development-only. The test split is
sealed unless the frozen validation gate returns `PASS_TO_TEST`. The protocol is
in
[`docs/intent-basin-gate-prereg-2026-08-10.md`](docs/intent-basin-gate-prereg-2026-08-10.md).

Development attempts 01 and 02 were formally `HARNESS_INVALID` because one
cap-boundary state exposed an unstable near-one `acos` log-map calculation.
Their receipts and outputs are archived byte-for-byte under `results/`.  The
implementation-only repair uses stable `atan2(||tangent||, dot)` geometry.  No
scientific conclusion is taken from either invalid attempt; only the third,
fully valid lineage supports the development verdict above.

Workflow (completed):

1. generate the outcome-blind manifests from the pinned official source;
2. run analytic/synthetic tests and preflight;
3. commit the protocol, code, tests, and manifests before any real embedding;
4. create the development registration receipt;
5. execute the development command once;
6. write its verdict before any continuation;
7. register test only if the immutable development result passes every bar.

No model training, GPU allocation, remote compute, KB/Flow/Ingesto integration,
or parameter ladder is authorized.

# HLM5 Class-Balanced Intent Basin Gate

This package tests one bounded follow-up to the failed ParaRel density settle:

> Can repeated degree-five state dynamics improve natural intent routing when
> every intent has equal support, common-mode geometry is removed, and movement
> is constrained to a training-derived trust region?

The official CLINC150 validation split is development-only. The test split is
sealed unless the frozen validation gate returns `PASS_TO_TEST`. The protocol is
in
[`docs/intent-basin-gate-prereg-2026-08-10.md`](docs/intent-basin-gate-prereg-2026-08-10.md).

Development attempt 01 was formally `HARNESS_INVALID` because one
cap-boundary state exposed an unstable near-one `acos` no-movement check.  Its
receipts and outputs are archived byte-for-byte in `results/invalid-attempt-01/`.
The implementation-only chord-distance repair is awaiting a fresh preflight
and registration; no scientific conclusion is taken from the invalid attempt.

Workflow:

1. generate the outcome-blind manifests from the pinned official source;
2. run analytic/synthetic tests and preflight;
3. commit the protocol, code, tests, and manifests before any real embedding;
4. create the development registration receipt;
5. execute the development command once;
6. write its verdict before any continuation;
7. register test only if the immutable development result passes every bar.

No model training, GPU allocation, remote compute, KB/Flow/Ingesto integration,
or parameter ladder is authorized.

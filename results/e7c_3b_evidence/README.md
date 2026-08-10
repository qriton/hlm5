# E7c 3B fresh ZCA confirmation evidence

This directory preserves the complete byte-identical receipt chain from
Leonardo job `51800064` (A100-SXM-64GB, exit 0, 3m55s, 2026-08-10).
The registered implementation commit is
`58cedff87f270f1e7047e491a80565b7da05390f`.

The formal verdict is `PASS_3B_ZCA_GEOMETRY`. On the untouched third
1,200-token pool, raw target-row directions were geometrically reachable for
705 targets and produced 704 strict ordinary-BF16 wins. The frozen raw-row ZCA
half-whitening direction was geometrically reachable for 1,200 of 1,200
targets and produced 1,200 of 1,200 strict ordinary-BF16 wins. It gained 495
reachable targets and lost zero raw-reachable targets. A new Python process
reproduced both arms, every decision, every recorded value, and every native
full-logit hash exactly.

The result licenses only the bounded claim in `verdict.json`: it is a
single-prompt, single-token, pinned-model and pinned-runtime result. It does not
establish factual correctness, multi-token editing, PPL neutrality,
cross-model transfer, training scalability, or legal compliance.

SHA-256:

- `preflight.json`: `8977c02e4cdc40d080ea99d21e295b29b2b037344aa423154a3a120b52cfc77d`
- `execution_receipt.json`: `d22244ec469660c2a2f56ab426de8f3bf9449c67cfaf48d59999bfa86193a42f`
- `attempt.json`: `d4b1b2d4464480a2f9606873cbc2b845b2918819280c9e83d3903b3495e361e0`
- `admission.json`: `f7bf2d05721a50852cc905a0d7da6200a4b51cbe7b974be972de628ad3cb8895`
- `result.json`: `88c9819648ce7a30ba3e6d75aa5da2c317d072db8eae0160631dada212f9cf44`
- `verdict.json`: `32cc956a9a56e0be56e0976c4c17e7862544a2ede07735ac45ea6faffba0f1f6`

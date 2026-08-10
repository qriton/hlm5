# Invalid development attempt 01

This directory preserves the first registered development execution byte for
byte.  Its formal verdict is `HARNESS_INVALID`; none of its scientific metrics
is eligible for interpretation or promotion.

The sole failed validity gate was the independent final-state match.  One of
3,097 development rows differed by `3.2364530500039734e-09`, above the frozen
`1e-10` tolerance, although its prediction agreed.  Every other row matched at
approximately machine precision.  The affected row was on the spherical cap:
`acos(dot)` reported a `1.4901161193847656e-08` movement for a projected
roundoff-scale displacement.  The implementation-only repair uses stable
Euclidean chord distance for the registered no-movement predicate.  Scientific
constants, data, model, controls, bars, and seeds are unchanged.

Lineage:

- Git head: `a448690e0f06c0c67a6d8287ab1e4e07cd91e5b5`
- Registration scientific digest:
  `51e037d6fe99e2a399a6a6fb4d36430927140bff73074839e0150c7563cf7586`
- `preflight.json` SHA-256:
  `49714345ffd94023dac756bba04306f70f536a4c0c140c84ca19ae92c4413191`
- `development_registration.json` SHA-256:
  `6aff639ef86be6262220fb47e515e4bcbcb5abc1d72eb3b67ba9f134c75316a4`
- `intent_basin_development_rows.jsonl` SHA-256:
  `14199f7067db48f9ff756ec3d71099a118c3b88802b33eb2cbea9df896fa26b6`
- `intent_basin_development_result.json` SHA-256:
  `4d4426b3642fba8f17cdefd56e73ff9250b3409da2dfe523aafbea3ddec89c80`
- Result digest:
  `cf388ba1aa8657e2228ad7269a19a8241877274ba8217a53dd9f922ca7ef9053`

The official CLINC150 test split was not encoded.

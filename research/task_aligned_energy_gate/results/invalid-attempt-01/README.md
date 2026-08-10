# Invalid attempt 01

The first registered Banking77 development execution was `HARNESS_INVALID`,
not a scientific result. Every validity gate passed except independent matched
state equality. Primary and independent predictions agreed for all 770 rows,
their final cap norms agreed to floating-point precision, and the maximum state
difference along the cap was `5.102135665069341e-07`, above the registered
`1e-10` tolerance.

The implementation-only repair makes both paths share the same canonical CPU
float64 primitives for softmax, row norm, displayed-energy evaluation, and ball
projection while retaining separately structured optimizer/Armijo control
flow. Scientific constants and selection rules are unchanged. Files in this
directory are the exact original artifacts:

- `preflight.json` SHA-256
  `2e572324e287e109754eceb7be242a88fb865222b42779d163681860e1cbd349`;
- `development_registration.json` SHA-256
  `e6976fe3fc391c8d7bf1d8cfa43886cae95e12a1b3e0424b331776b8e5206968`;
- raw rows SHA-256
  `310bd7546285ffc569e55f43e6c50ad37273bbe7b238e6a5224993694902735a`;
- result SHA-256
  `8daac76c699647b9219ac97b1a84664396218683dbd8b0b0d0c7fcd8a5998d9a`.

The official Banking77 test split remained unencoded.

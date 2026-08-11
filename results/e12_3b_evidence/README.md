# E12 untouched 3B confirmation evidence

Formal verdict: `FAIL_3B_UNTOUCHED_FINAL`.

Leonardo job `51801708` ran on `lrdn2974.leonardo.local`, distinct from the
E10 and E11 nodes, and completed with exit `0:0` in 6m49s. The evidence was
produced from implementation commit
`b2e1caf47623b8f6082a6c24eedba6e8f4ffd991` under protocol commit
`34c25ca489314ab14ed1594c6fd8badd17bba704`.

K=64 and K=256 passed every registered bar. K=1,024 had 1,024/1,024 open
gates, own-slot routes, and nonzero deltas, but 1,023/1,024 strict
ordinary-head wins. The single miss was an exact BF16 logit tie between the
registered target ` linebacker` and competitor ` cornerback`; the target
remained the returned argmax because its token ID was lower. All 12,288
locality controls, exact rollback, the portable bundle, and new-process replay
passed. The largest fully passing registered tier is 256; capacity between 256
and 1,024 was not measured.

Raw SHA-256:

- `preflight.json`: `195543a099cae91da4b9983a912306eca385e74ff2f9c8c72ed97a07e5c12da6`
- `execution_receipt.json`: `7020ddea6da073b4c557dda8ca25dc7714d2fe1dc2dedce7dbaceff77fb1622b`
- `attempt.json`: `7f37364f078ba61f4cb18598f3706a351aec6cb20719b4e30a0aeb06d53a8d92`
- `final_adapter_1024.pt`: `bd409bcf7c0421a6cded00c18f5d0b555f1eb167f91a98b6042f3a2cbd2a3c63`
- `admission.json`: `155c35b7dac2d3805faf6fff2c7c8b2df597d318e3e5246d6d2a2cda4caeb120`
- `result.json`: `6fb90a9bce3404cc421086b85dbfb02b0df0cb0a450bc51b10ce56200f1fcdd7`
- `verdict.json`: `bea89e68cffc9004db51665cf351d7db6f5525ed433869acdfef45d910096661`

The valid fail burns this population. Do not retune or rerun it as a fresh
confirmation.

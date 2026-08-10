# Invalid development attempt 02

This directory preserves the second registered development execution byte for
byte.  Its formal verdict is `HARNESS_INVALID`; none of its scientific metrics
is eligible for interpretation or promotion.

The chord-distance no-movement repair from attempt 01 was insufficient.  The
same row exceeded the independent final-state tolerance by
`3.235401779821956e-09`.  Stepwise diagnostics localized the divergence to
step 8: the states agreed through step 7 within `5.6e-17`, and both
implementations generated the same projected proposal within approximately
`2e-16`.  One local dot product rounded to exactly `1.0`, making `acos(dot)`
return a zero log-map and changing the independent line-search branch.  The
next implementation-only repair evaluates the mathematically identical angle
as `atan2(||tangent||, dot)`.  It changes no scientific constant or primary
candidate proposal; on the failing step, the primary accepted state is
unchanged.

Lineage:

- Git head: `26dab05d3a1e997c760484928942bb98edcb6115`
- Registration scientific digest:
  `f599d7dd618882ea52d1471374b0f428ed9e258cbb5bb48cd26149761e46fb50`
- `preflight.json` SHA-256:
  `336464a6b73c38b73769c181323ef42cd15a3ca2b655ccaec23e3fe99de2bb48`
- `development_registration.json` SHA-256:
  `d8bc480d12f918d8d34f0724d4d65c356d24e401b4ba414f740392805d6a37d3`
- `intent_basin_development_rows.jsonl` SHA-256:
  `027874f9f7451b978b34f11c5c0260f0cd0a5a8e6a2edaa6a6cb428494c0ce6a`
- `intent_basin_development_result.json` SHA-256:
  `05d105c4dbf4454549bd40b1c4805c86ec86af2fdd380b5fd8b6d55728e07efc`
- Result digest:
  `80860bb56fe72d15217da8a61b8ffda988748ec4eb62bd6c3e713afc262a952a`

The official CLINC150 test split was not encoded.

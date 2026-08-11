# Legacy G4 hybrid 3B R2c evidence

Formal verdict: `INVALID_REPRODUCIBILITY`.

Leonardo job `51821287` compared independently launched continuous and split
arms at the common 522.5K midpoint before the split arm was restarted. All 96
ranks differed semantically, the precise midpoint PPL values were
15.520899957093135 and 15.514838289543917, and the registered launcher stopped
before the restart intervention. The immutable 522K source passed the launcher's
post-control aggregate check.

This is a valid harness result and no claim about restart continuity, memory
utility, or model failure follows from it. The large checkpoint directories
remain on Leonardo; this directory preserves the attempt/result chain, logs,
markers, 96 per-rank comparison reports, aggregate report, and their compressed
evidence archive.

See `verdict.json` for the hash chain and
`../../docs/legacy-g4-hybrid-r2c-invalid-reproducibility-verdict-2026-08-11.md`
for the interpretation and next gate.

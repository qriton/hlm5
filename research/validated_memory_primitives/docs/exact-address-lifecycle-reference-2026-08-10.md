# HLM5 Exact-Address Lifecycle — Runnable Reference

**Status:** `CACHE_EQUIVALENT_REFERENCE`  
**Scope:** deterministic exact addresses and memory lifecycle only

This reference packages the part of E5-v3 that genuinely worked without
turning it into a larger claim. It runs entirely on CPU and exercises:

1. canonical-ID address generation;
2. create;
3. edit;
4. reversible deactivate;
5. serialized-bundle restore into a fresh memory object;
6. reactivate;
7. hard delete to a gate-off base state; and
8. an off-support negative assay.

Every phase runs both the degree-5 support-masked reader and its matched cosine
cache. The reference fails closed unless their routes, slots, supports, gains,
outputs, and predictions are bit-identical.

Run it from `D:\HLM5`:

```powershell
python -B demos\exact_address_lifecycle_reference.py
python -B tests\test_exact_address_lifecycle_reference.py
```

The demo writes
`baselines/results/exact_address_lifecycle_reference.json`; the CLI rejects
every other output path. It first verifies the hashes and scientific digest of
the registered E5-v3 implementation and result, and cannot overwrite those
pinned artifacts.

## Honest claim boundary

The reference demonstrates a deterministic exact-address lifecycle. It does
not demonstrate semantic routing, paraphrase generalization, language-model
quality gain, a polynomial-Hopfield advantage over a cache, or a reason to run
K>4096 or resume G4.

The registered multi-support follow-up has now been run. Balanced two-share
reconstruction worked, but degree-five accuracy was only 40% across the locked
skew ladder versus 60% for cosine-soft aggregation, and an independently coded
degree-five cache was bit-exact in every phase. The formal status is
`NARROW_MULTI_SUPPORT_CACHE_EQUIVALENT`; see
`docs/multi-support-kernel-equivalence-verdict-2026-08-10.md`.

Primary evidence remains:

- `baselines/results/address_conditioning_v1.json`;
- `baselines/RESULTS.md`;
- `baselines/results/multi_support_kernel_equivalence_reference.json`;
- `docs/multi-support-kernel-equivalence-verdict-2026-08-10.md`;
- `D:\HLM2\conclusions\hlm5-e5v3-address-conditioning-verdict-2026-08-10.md`.

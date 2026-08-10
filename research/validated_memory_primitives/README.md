# HLM5 Validated Memory Primitives

This is the small, self-contained front door for the two HLM5 memory results
that survived matched controls on 2026-08-10.

From the repository root, run:

```powershell
cd research\validated_memory_primitives
python -B run_workbench.py
```

Additional verification commands:

```powershell
python -B run_workbench.py
python -B run_workbench.py --verify-only
python -B tests\test_workbench.py
```

The registered replay environment is Python 3.11.7, PyTorch 2.5.1+cu121,
CPU. The workbench fails closed if a frozen source, protocol, registration, or
result changes.

## What is working

1. **Exact-address lifecycle:** create, edit, deactivate, serialize/restore,
   reactivate, hard-delete, and off-support rejection all replay. The HLM5
   degree-5 reader is bit-identical to a matched cache in every phase.
2. **Bounded multi-support composition:** two stored values can jointly solve a
   task that hard top-1 cannot solve. In the registered orthogonal construction,
   degree-5 gets 40%, cosine-soft gets 60%, and hard top-1 gets 0%. Removing one
   required value or shuffling values reduces accuracy to 0%; off-support output
   is bit-zero. The HLM5 reader is again bit-identical to a matched degree-5
   cache.

## What this means

Use the governed lifecycle now. Use support-gated nonlinear composition only
inside the measured boundary. Prefer the matched-cache implementation for a
production reference until an HLM-specific property is independently measured.

## What this does not mean

This package does not demonstrate an HLM-specific retrieval advantage,
semantic/paraphrase routing, language-model gain, robustness over the full
mixture range, or a reason to train or scale another model. It uses no
checkpoint and performs no language-model forward pass.

`snapshot_manifest.json` pins the minimal source and evidence snapshot from the
development research tree. The historical local source path is retained in the
manifest as provenance; no external checkout is required. Generated combined
output is written only under `outputs\` and is not part of the frozen evidence.

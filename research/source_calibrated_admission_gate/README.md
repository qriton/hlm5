# HLM5 SCA-1 source-calibrated admission gate

This package tests whether blind covariance conditioning plus a source-only
admission threshold transfers to three-shot HLM5 routing in held-out TOPv2
domains. It uses the frozen HLM5-136M trunk and exact degree-five
positive-cosine reader. It contains no recurrence, training, target-domain
metric fit, or KB/Flow/Ingesto integration.

Read the
[`preregistration`](docs/source-calibrated-admission-prereg-2026-08-10.md)
before running anything. The TOPv2 test population stays unencoded unless all
development bars pass and a separate test registration is committed.

## Outcome-blind population preparation

Place the official TOPv2 1.1 files under
`D:\HLM2\runtime\datasets\topv2-1.1\official`, then run:

```powershell
python -m research.source_calibrated_admission_gate.prepare_topv2
python -m unittest discover -s research\source_calibrated_admission_gate\tests -p "test_*.py" -v
```

Preparation normalizes and hashes utterances, removes overlaps and duplicate
conflicts, and fixes source/target populations. It does not tokenize an
utterance with HLM5 or compute a route.

## Claim boundary

Even a pass is a conventional static key-conditioning result in frozen HLM5
hidden space. The reader is cache-equivalent. It is not evidence for a
Hopfield advantage, recurrence, production routing, explanation, compliance,
or scaling.

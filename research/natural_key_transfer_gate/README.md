# HLM5 NKT-1 natural-key transfer gate

This package tests one bounded question: whether a source-domain,
within-class Fisher preconditioner transfers to three-shot HLM5 hidden-key
routing on unseen MASSIVE intents. It uses the frozen HLM5-136M trunk and the
exact degree-five positive-cosine address rule. It contains no recurrence,
training, target-domain covariance fit, or KB/Flow/Ingesto integration.

The design was committed before any MASSIVE utterance was encoded. Read
[`docs/natural-key-transfer-prereg-2026-08-10.md`](docs/natural-key-transfer-prereg-2026-08-10.md)
before running anything.

## Outcome-blind preparation

The official MASSIVE 1.1 archive is external. With the registered English
JSONL and license under `D:\HLM2\runtime\datasets\massive-1.1\1.1`:

```powershell
python -m research.natural_key_transfer_gate.prepare_massive
python -m unittest discover -s research\natural_key_transfer_gate\tests -p "test_*.py" -v
```

Preparation writes only IDs, hashes, counts, domain partitions, and support
descriptors. It does not tokenize an utterance or compute a route.

After committing the implementation and manifest, the execution order is:

```powershell
python -m research.natural_key_transfer_gate.run_natural_key_transfer_gate preflight
python -m research.natural_key_transfer_gate.run_natural_key_transfer_gate register
python -m research.natural_key_transfer_gate.run_natural_key_transfer_gate development
```

Preflight encodes one fixed synthetic anchor only. Development fails closed
unless the preflight, registration, Git commit, code/tests, source, model,
tokenizer, and manifest are unchanged. The official MASSIVE test population
is not encoded by this package's development command.

## Claim boundary

Even a pass is a conventional key-conditioning result in an HLM5 hidden
space. The degree-five reader is top-slot-equivalent to cosine cache lookup.
It is not evidence for a Hopfield advantage, recurrence, scaling, production
routing, factual editing, explainability, compliance, or safety.

# HLM5 SCA-2 two-axis admission gate

This package tests the smallest repair implied by SCA-1: retain the working
source-calibrated proximity gate, then add a separately source-calibrated
winner-versus-runner-up class margin to reject confident-looking ambiguous
routes. It uses the frozen HLM5-136M trunk and the exact degree-five,
three-support reader. It contains no recurrence, training, target-label fit, or
KB/Flow/Ingesto integration.

Read the
[`preregistration`](docs/two-axis-admission-prereg-2026-08-10.md) before
running anything. The official HWU64 fold 2 remains unencoded unless all
development bars pass and a separate test registration is committed.

## Outcome-blind population preparation

Place the official repository at
`D:\HLM2\runtime\datasets\hwu64-official`, checked out at the pinned commit,
then run:

```powershell
python -m research.two_axis_admission_gate.prepare_hwu64
python -m unittest discover -s research\two_axis_admission_gate\tests -p "test_*.py" -v
```

Preparation reconstructs the official 64-intent benchmark from its ten Rasa
test folds, removes two same-label duplicates and one held-out overlap,
assigns intent populations by a frozen salt, and writes hashes and row
descriptors. It does not tokenize an utterance with HLM5 or compute a route.

## Registered development run

After committing the implementation with a clean worktree, run:

```powershell
python -m research.two_axis_admission_gate.run_two_axis_admission_gate preflight
python -m research.two_axis_admission_gate.run_two_axis_admission_gate register
python -m research.two_axis_admission_gate.run_two_axis_admission_gate development
python -m research.two_axis_admission_gate.run_two_axis_admission_gate replay
```

Preflight encodes only one fixed synthetic anchor. Development creates a
source-audit cache first. The target cache is created only if every source
audit bar passes. Replay reads the immutable caches, compares the complete
scientific object and row evidence, and writes a separate receipt without
overwriting the original result.

## Claim boundary

Even a pass would validate a static selective-routing primitive in frozen
HLM5 hidden space. Degree five is top-slot-equivalent to positive cosine. It
would not establish a Hopfield advantage, recurrence, production routing,
explanation, compliance, or scaling.

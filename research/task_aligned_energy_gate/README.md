# Banking77 task-aligned energy gate

This package asks two deliberately separate questions on a new intent
population:

1. Does label-informed Fisher/LDA preconditioning improve a frozen MiniLM
   static reader over unaligned centroid controls?
2. After that metric is frozen, do repeated steps of exact descent on its
   class-mixture energy improve the identical zero-step reader and one-step
   dose?

The first question tests the alignment/preconditioning fix class that has
survived elsewhere in HLM5. The second tests whether dissipative recurrence
adds value once the scalar is genuinely task-aligned. A dynamics failure does
not erase an alignment pass.

The fixed workflow is:

```powershell
python -B prepare_banking77.py --source-repo D:\HLM5-research-data\banking77-57ec275d
python -B run_task_aligned_energy_gate.py --preflight
python -B run_task_aligned_energy_gate.py --register-development
python -B run_task_aligned_energy_gate.py --run-development
```

Registration must occur from the frozen implementation commit before any real
Banking77 embedding is computed. `--run-development` verifies the receipt,
encodes fit/calibration first, selects the registered shrinkage by calibration
static macro accuracy, and only then encodes development. It never loads test.

The official test can be registered only for the claim class that passed:
alignment alone or alignment plus dynamics. There is no scaling authorization.

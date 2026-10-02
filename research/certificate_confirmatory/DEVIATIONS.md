# Deviations from the registered procedure

1. **2026-10-01 ~21:40 (CEST): ROME ran out of GPU memory on 27 of its first 87 records.**
   Another process on this machine holds 10.5 GB of the 20 GB GPU (it predates this study and
   is not ours to stop), leaving ~8.6 GB; ROME on fp32 gpt2-xl peaks right at that limit. The
   runner writes an error row for such a record, and its resume logic would then skip it, so
   the failures would be silently dropped, and not at random (memory peaks vary with prompt
   length). Action: the run was stopped; the 27 error rows were moved to
   `results/editors_window.oom-errors.jsonl` (kept for audit) and removed from
   `results/editors_window.jsonl`; the run resumed with
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, an allocator setting that changes
   memory layout, not arithmetic. No test, window, editor, feature or threshold changed;
   cert_window.py, editors_window.py and analyze_confirm.py are byte-identical to the
   registered hashes.

2. **2026-10-01 ~22:05: the allocator setting is not supported on Windows** ("expandable_segments
   not supported on this platform"), so it had no effect; ROME ran out of memory again on its
   first remaining record. The run was stopped before any further error row was kept.

3. **Audit-file mishap.** Re-running the first version of the split script overwrote
   `editors_window.oom-errors.jsonl` and its backup with an empty pass. The 27 OOM events remain
   recorded line by line in `logs/ROME.log`; the audit file was rebuilt from that log by
   `rebuild_oom_audit.py`. The split script now appends with a timestamp and never overwrites.
   The results file itself was unaffected: it held 660 good rows (BASE 600, ROME 60) before and
   after.

4. **2026-10-01 23:31: resumed with the whole GPU.** The other GPU process was paused; the run
   resumed from the 660 kept rows (BASE 600, ROME 60) and finished at 02:40 with no further errors. The 27
   records that had run out of memory were recomputed in this pass like any other pending record.

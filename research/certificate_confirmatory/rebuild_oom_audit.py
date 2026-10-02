"""Rebuild the audit of the 27 ROME OOM rows of the first pass from logs/ROME.log (the separate audit file was
overwritten by a second pass of the old split script; see DEVIATIONS.md)."""
import json, re
log = open(r'D:\HLM5-cert-confirm\baselines\confirm\logs\ROME.log', encoding='utf-8', errors='replace').read()
rows = [{'editor': 'ROME', 'case_id': int(m.group(2)), 'error': 'CUDA OOM (from log)', 'log_line': m.group(0)[:200], 'pass': 1}
        for m in re.finditer(r'\[ROME (\d+)/600\] case (\d+) ERROR CUDA OOM[^\n]*', log)]
seen, out = set(), []
for r in rows:
    if r['case_id'] in seen: continue
    seen.add(r['case_id']); out.append(r)
with open(r'D:\HLM5-cert-confirm\baselines\confirm\results\editors_window.oom-errors.jsonl', 'w', encoding='utf-8') as f:
    for r in out: f.write(json.dumps(r) + '\n')
print('OOM cases recorded in the log:', len(out), sorted(seen)[:5], '...')

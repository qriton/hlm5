"""Move OOM error rows out of the results so they are recomputed. Never overwrites: the backup and the audit file
are appended to, with a timestamp per pass."""
import json, time
p = r'D:\HLM5-cert-confirm\baselines\confirm\results\editors_window.jsonl'
stamp = time.strftime('%Y-%m-%dT%H:%M:%S')
lines = [l.rstrip('\n') for l in open(p, encoding='utf-8') if l.strip()]
keep = [l for l in lines if not ('error' in json.loads(l) and 'OOM' in json.loads(l)['error'])]
oom = [l for l in lines if l not in keep]
with open(p + '.backups', 'a', encoding='utf-8') as f:
    f.write(f'# pass {stamp}: {len(lines)} rows\n' + '\n'.join(lines) + '\n')
with open(p.replace('.jsonl', '.oom-errors.jsonl'), 'a', encoding='utf-8') as f:
    for l in oom: f.write(json.dumps({**json.loads(l), 'moved_at': stamp}) + '\n')
open(p, 'w', encoding='utf-8').write('\n'.join(keep) + '\n')
print('kept', len(keep), 'moved OOM rows', len(oom))

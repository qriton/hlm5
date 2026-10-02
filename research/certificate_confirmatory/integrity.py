"""Integrity of the confirmatory run: record counts, errors, restoration, registered hashes."""
import json, collections, hashlib, os
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'baselines', 'confirm')
rows = [json.loads(l) for l in open(os.path.join(R, 'results', 'editors_window.jsonl'), encoding='utf-8') if l.strip()]
c = collections.Counter((d['editor'], 'error' in d) for d in rows)
print('records (editor, error):', sorted(c.items()))
ids = {e: sorted(d['case_id'] for d in rows if d['editor'] == e and 'error' not in d) for e in ('BASE', 'ROME', 'MEMIT', 'FT')}
print('same 600 cases for every editor:', all(ids[e] == ids['BASE'] for e in ids), '| case ids', ids['BASE'][0], '..', ids['BASE'][-1], '| duplicates:', {e: len(v) - len(set(v)) for e, v in ids.items()})
print('errors:', [(d['editor'], d['case_id'], d['error'][:60]) for d in rows if 'error' in d][:5])
print('max restore deviation:', max(d.get('restore_max_dev', 0) for d in rows))
cert = [json.loads(l) for l in open(os.path.join(R, 'results', 'cert_window.jsonl'), encoding='utf-8') if l.strip()]
print('certificate rows:', len(cert), '| same cases as editors:', sorted(r['case_id'] for r in cert) == ids['BASE'])
agg = lambda e, k: sum(d[k] for d in rows if d['editor'] == e and 'error' not in d) / 600
for e in ('BASE', 'ROME', 'MEMIT', 'FT'):
    print(f"{e:6} efficacy {agg(e, 'efficacy_argmax'):.3f}  paraphrase {agg(e, 'paraphrase_argmax'):.3f}  neighbourhood {agg(e, 'neighborhood_score'):.3f}")
reg = dict((l.split()[1].replace('/', os.sep), l.split()[0]) for l in open(os.path.join(HERE, 'REGISTRATION.sha256')) if len(l.split()) == 2 and len(l.split()[0]) == 64)
for rel, h in reg.items():
    now = hashlib.sha256(open(os.path.join(HERE, rel), 'rb').read()).hexdigest()
    print('unchanged' if now == h else 'CHANGED  ', rel)

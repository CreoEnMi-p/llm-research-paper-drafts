"""Single prespecified source-only heuristic test on the locked V2 A+C pair split.
No model or hyperparameter selection is permitted after seeing this outcome.
"""
import json,hashlib,statistics,argparse
from pathlib import Path
from collections import defaultdict
from audit_chance_heuristics import heuristics
p=argparse.ArgumentParser()
p.add_argument('--frozen',default='frozen_protocol_v2/FROZEN_AC_TEST_256pairs.jsonl')
p.add_argument('--manifest',default='frozen_protocol_v2/PROTOCOL_MANIFEST.json')
p.add_argument('--out',default='frozen_v2_script_heuristics.json');a=p.parse_args()
m=json.loads(Path(a.manifest).read_text());content=Path(a.frozen).read_bytes()
assert hashlib.sha256(content).hexdigest()==m['frozen_test_sha256']
rows=[json.loads(z) for z in content.splitlines()]
by=defaultdict(list)
for row in rows:
    l=row['left'];r=row['right']
    lp,lc=heuristics(l['source']);rp,rc=heuristics(r['source'])
    assert len(set(lc))==len(set(rc))==4
    assert l['label'] in lc and r['label'] in rc
    q={'pair':row['pair_id'],'production':l['production'],'candidate_count':4,'literal_occurrences':4,
       'predictions':{name:{'left':lp[name],'right':rp[name],
           'acc':((lp[name]==l['label'])+(rp[name]==r['label']))/2,
           'both_correct':int(lp[name]==l['label'] and rp[name]==r['label']),
           'flip':int(lp[name]!=rp[name]),'invariant':int(lp[name]==rp[name])} for name in lp}}
    by[l['production']].append(q)

def summarize(items):
    if not items:return {}
    return {'n_pairs':len(items),'candidate_unique':4,'chance_uniform_single_given_four':.25,
        'chance_uniform_independent_pair':.0625,'chance_uniform_distinct_pair':1/12,
        'metrics':{name:{key:statistics.mean(p['predictions'][name][key] for p in items) for key in ['acc','both_correct','flip','invariant']} for name in items[0]['predictions']}}
res={'manifest_sha256':m['frozen_test_sha256'],'note':'Test used once after selected train-only protocol; heuristic methods fixed in old audit prior to V2 freeze.',
    'overall':summarize([z for items in by.values() for z in items]),
    'by_production':{k:summarize(v) for k,v in by.items()}}
Path(a.out).write_text(json.dumps(res,indent=2))
print('SCRIPT_V2_FINAL='+json.dumps(res['overall']))

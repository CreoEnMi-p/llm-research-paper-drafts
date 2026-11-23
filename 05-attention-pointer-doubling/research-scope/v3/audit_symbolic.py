"""Sealed-V3 structural ceiling and preregistered source-only heuristic audit.

The learned baseline policy is selected on AB/CD DEVELOPMENT pairs only.
V3 labels are read only after that selection; no posthoc test-based winner.
The restricted AST interpreter is a symbolic ceiling, not a neural baseline.
"""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
sys.path[:0] = [str(PARENT),str(PARENT/'repro')]
from ast_reference import predict
from grammar_compositions import make_dataset

lock=json.loads((ROOT/'PROTOCOL_LOCK.json').read_text())
manifest=json.loads((ROOT/'FROZEN_MANIFEST.json').read_text())
raw=(ROOT/'FROZEN_V3_448pairs.jsonl').read_bytes()
assert hashlib.sha256(raw).hexdigest()==manifest['test_sha256']==lock['frozen_test']['expected_sha256']
rows=[json.loads(r) for r in raw.splitlines()]
assert len(rows)==448

def digits(src):
    ds=[int(z) for z in re.findall(r'\b(?:2[0-9]|3[0-5])\b',src)]
    assert len(ds)==len(set(ds))==4,ds
    return ds

def query_kind(src):
    expr=src.rsplit('print(',1)[-1].strip()
    return 'call' if '()' in expr else 'direct'

def no_graph_baseline(src,policy):
    nums=digits(src)
    if policy in ('first','last','second','third'):
        return nums[{'first':0,'last':3,'second':1,'third':2}[policy]]
    if policy=='min':return min(nums)
    if policy=='max':return max(nums)
    if policy.startswith('callvs_'):
        a,b=map(int,policy.removeprefix('callvs_').split('_'))
        return nums[a if query_kind(src)=='call' else b]
    raise KeyError(policy)

DEV=[]
for combo,seed in [('AB',31071),('CD',41731)]:
    for a,b in make_dataset(combo,seed,64,True):
        DEV.append(({'source':a.source,'label':a.truth},
                    {'source':b.source,'label':b.truth}))
POLICIES=['first','last','second','third','min','max']
POLICIES += [f'callvs_{i}_{j}' for i in range(4) for j in range(4)]

def metrics(pairs,policy):
    good=0;single=0;invariant=0
    by=collections.defaultdict(lambda:[0,0])
    for pair in pairs:
        p=no_graph_baseline(pair[0]['source'],policy)
        q=no_graph_baseline(pair[1]['source'],policy)
        a=int(p==pair[0]['label']);b=int(q==pair[1]['label'])
        good+=a*b;single+=a+b;invariant+=int(p==q)
    n=len(pairs)
    return {'both':good/n,'single':single/(2*n),'invariant':invariant/n}

# Stable lexicographic tie-break, not selected using held-out V3 outcomes.
select=sorted(POLICIES,key=lambda k:(-metrics(DEV,k)['both'],k))[0]
dev_scored={k:metrics(DEV,k) for k in POLICIES}
print('V3_DEV_POLICY_PRESELECT',select,dev_scored[select],flush=True)

v3_pairs=[(z['left'],z['right']) for z in rows]
heur={p:metrics(v3_pairs,p) for p in POLICIES}
family=collections.defaultdict(list)
strata=collections.defaultdict(list)
for z in rows:
    family[z['family']].append((z['left'],z['right']))
    strata[z['stratum']].append((z['left'],z['right']))
ast_ok=0
for z in rows:
    assert z['left']['label']!=z['right']['label']
    assert z['left']['source'].rsplit('print(',1)[0]==z['right']['source'].rsplit('print(',1)[0]
    assert sorted(digits(z['left']['source']))==z['literal_candidates']
    a=predict(z['left']['source'])
    b=predict(z['right']['source'])
    assert (a,b)==(z['left']['label'],z['right']['label']), (z['pair_id'],a,b)
    ast_ok+=1
assert ast_ok==len(rows)
out={
 'protocol':lock['protocol'],'test_sha256':manifest['test_sha256'],
 'ast_symbolic_ceiling':{'both_correct':1.0,'single_correct':1.0,'n_pairs':ast_ok},
 'dev_policy_preselection':{'policy':select,'dev_scores':dev_scored[select],
                           'split':'AB/CD only, no V3 family labels used'},
 'all_predeclared_source_heuristics':heur,
 'selected_heuristic_v3':heur[select],
 'heuristic_by_family':{k:metrics(v,select) for k,v in family.items()},
 'heuristic_by_stratum':{k:metrics(v,select) for k,v in strata.items()},
 'max_test_source_heuristic_diagnostic':{
     'note':'Exploratory inspection ONLY; NEVER treat a best-of-test policy as preregistered baseline',
     'policy':max(heur,key=lambda k:heur[k]['both']),
     'score':max(z['both'] for z in heur.values())},
 'limitations':'All seven families are finite synthetic Python productions. Exact AST result is an explicit restricted symbolic interpreter; not a neural model.',
}
(ROOT/'V3_SYMBOLIC_HEURISTIC_AUDIT.json').write_text(json.dumps(out,indent=2))
print('V3_AST_CEILING',ast_ok,'/',len(rows),flush=True)
print('V3_SELECTED_SOURCE_HEURISTIC',select,heur[select],flush=True)
print('V3_HEURISTIC_FAMILY',json.dumps(out['heuristic_by_family']),flush=True)

"""Dev-only alias depth and unseen-primary-name extrapolation; never opens V3.

This executes the exact locked train script via runpy in --dev-only mode,
then reuses the trained network for newly constructed diagnostic pairs.
It does not alter preregistered V3 code, training, or frozen test sets.
"""
import argparse, hashlib, json, random, runpy, sys
from pathlib import Path
import ast

p=argparse.ArgumentParser()
p.add_argument('--seed',type=int,required=True)
p.add_argument('--arm',choices=['pointer','pointer-canonical','pointer-scrambled','candidate-classifier'],required=True)
p.add_argument('--out',required=True)
a=p.parse_args()
ROOT=Path(__file__).resolve().parent
script=ROOT/'train_locked.py'
text=script.read_text()
assert '--dev-only' in text and 'STEPS = 5000' in text
orig_argv=sys.argv
sys.argv=[str(script),'--arm',a.arm,'--seed',str(a.seed),'--dev-only','--out',str(ROOT/f'DEV_LOCKED_{a.arm}_s{a.seed}.json')]
try:
    state=runpy.run_path(str(script),run_name='__v3_dev_only__')
finally:
    sys.argv=orig_argv
from scope_program_benchmark import _tokenize
net=state['net']; evaluate=state['evaluate']
known=list('abcdefghijkl')
unseen=list('mnop')

def dataset(depth,heldout,seed,n_pairs=96):
    r=random.Random(seed)
    result=[]
    for z in range(n_pairs):
        x=r.choice(unseen if heldout else known)
        alias=r.sample([v for v in known if v!=x],depth)
        remaining=[v for v in known if v!=x and v not in alias]
        n0,n1=r.sample(remaining,2)
        v0,v1,v2,v3=r.sample(range(20,36),4)
        lines=[f'{x} = {v0}']
        last=x
        for aa in alias:
            lines.append(f'{aa} = {last}')
            last=aa
        lines.extend([f'{x} = {v1}',f'{n0} = {v2}',f'{n1} = {v3}'])
        if r.randrange(2):lines[-2:]=lines[-2:][::-1]
        prefix='\n'.join(lines)+'\n'
        variants=[(last,v0),(x,v1)]
        if r.randrange(2):variants.reverse()
        pair=[]
        for target,truth in variants:
            src=prefix+f'print({target})\n'
            raw,ids,dep=_tokenize(src)
            assert len(ids)<=110
            lits=[node.value for node in ast.walk(ast.parse(src)) if isinstance(node,ast.Constant) and type(node.value) is int]
            assert len(set(lits))==len(lits)==4
            pair.append({'source':src,'ids':ids,'depth':dep,'label':truth})
        result.append(tuple(pair))
    return result

results={}
for depth in range(1,7):
    for heldout in (False,True):
        pairs=dataset(depth,heldout,2026112400+depth*101+int(heldout)*10007,96)
        metrics,_=evaluate(net,pairs)
        results[f'depth{depth}|'+('unseen_main' if heldout else 'seen_names')]=metrics
        print('V3_ALIAS_DEV',a.arm,a.seed,depth,heldout,metrics,flush=True)
Path(a.out).write_text(json.dumps({'arm':a.arm,'seed':a.seed,
  'protocol':'exploratory AB alias-only OOD development diagnostics; exact locked training run dev-only; never read frozen V3',
  'n_pairs_per_cell':96,'metrics':results},indent=2))

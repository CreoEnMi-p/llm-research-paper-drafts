"""CPython-executed scope/binding/value benchmark, no oracle scope links in inputs.

All Python programs come from fixed, terminating safe templates. Every example
is verified with an independent CPython exec() (restricted builtins). A model
receives only tokenized Python source with INDENT/DEDENT/NEWLINE markers.
Def-use and scope labels are NEVER exposed to a trained model.
"""
import ast
import io
import json
import keyword
import random
import re
import tokenize
from dataclasses import dataclass
from typing import Dict

VARIABLES = list("abcdefghijklmnop")
NUMBERS = list(range(20,36))
TRAIN_KINDS = ["direct_last","copy_snapshot","copy_chain","local_global","local_result",
               "global_mutation","closure_late","closure_snapshot","param_global",
               "param_result","nested_local","nested_global"]
OOD_KINDS = ["nonlocal_mutation","nonlocal_untouched","closure_first","closure_second",
             "long_reassign","long_alias","nested3","global_from_nested"]
ALL_KINDS=TRAIN_KINDS+OOD_KINDS
KEYWORDS = ['def','return','global','nonlocal','print','pass']
PUNCT = ['=','(',')',':',',']
VOCAB=['<PAD>','<CLS>','<INDENT>','<DEDENT>','<NL>']+VARIABLES+['fun','outer','inner','mid','make','first','second']+KEYWORDS+PUNCT+[str(x) for x in NUMBERS]
ID={w:i for i,w in enumerate(VOCAB)}
NUM_START=20
CLS=ID['<CLS>'];PAD=ID['<PAD>'];NS=len(VOCAB)

@dataclass
class Program:
    source:str
    kind:str
    expect:int
    output:int
    tokens:list
    ids:list
    depth:list
    mask:list
    num_statements:int
    max_depth:int


def _source(kind, rng, value_pool=NUMBERS):
    a,b,c,d,e,f,g,h=rng.sample(VARIABLES[:8],8)
    v0,v1,v2,v3=rng.sample(value_pool,4)
    L=[]
    if kind=='direct_last':
        L=[f'{a} = {v0}',f'{a} = {v1}',f'print({a})'];truth=v1
    elif kind=='copy_snapshot':
        L=[f'{a} = {v0}',f'{b} = {a}',f'{a} = {v1}',f'print({b})'];truth=v0
    elif kind=='copy_chain':
        L=[f'{a} = {v0}',f'{b} = {a}',f'{a} = {v1}',f'{c} = {b}',f'{b} = {a}',f'print({c})'];truth=v0
    elif kind in ('local_global','local_result'):
        L=[f'{a} = {v0}', 'def fun():',f'    {a} = {v1}',f'    return {a}',f'{b} = fun()',f'print({a if kind=="local_global" else b})'];truth=v0 if kind=='local_global' else v1
    elif kind=='global_mutation':
        L=[f'{a} = {v0}','def fun():',f'    global {a}',f'    {a} = {v1}',f'fun()',f'print({a})'];truth=v1
    elif kind=='closure_late':
        L=[f'{a} = {v0}','def fun():',f'    return {a}',f'{a} = {v1}',f'print(fun())'];truth=v1
    elif kind=='closure_snapshot':
        L=[f'{a} = {v0}','def fun():',f'    return {a}',f'{b} = fun()',f'{a} = {v1}',f'print({b})'];truth=v0
    elif kind in ('param_global','param_result'):
        L=[f'{a} = {v0}',f'def fun({a}):',f'    return {a}',f'{b} = fun({v1})',f'print({a if kind=="param_global" else b})'];truth=v0 if kind=='param_global' else v1
    elif kind in ('nested_local','nested_global'):
        L=[f'{a} = {v0}','def outer():',f'    {a} = {v1}', '    def inner():',f'        return {a}', '    return inner()',f'{b} = outer()',f'print({b if kind=="nested_local" else a})'];truth=v1 if kind=='nested_local' else v0
    elif kind=='nonlocal_mutation':
        L=['def outer():',f'    {a} = {v0}','    def inner():',f'        nonlocal {a}',f'        {a} = {v1}','    inner()',f'    return {a}','print(outer())'];truth=v1
    elif kind=='nonlocal_untouched':
        L=['def outer():',f'    {a} = {v0}','    def inner():',f'        {a} = {v1}',f'        return {a}','    inner()',f'    return {a}','print(outer())'];truth=v0
    elif kind in ('closure_first','closure_second'):
        L=[f'def make({a}):','    def inner():',f'        return {a}', '    return inner',f'{b} = make({v0})',f'{c} = make({v1})',f'print({b if kind=="closure_first" else c}())'];truth=v0 if kind=='closure_first' else v1
    elif kind=='long_reassign':
        vals=rng.sample(value_pool,8)
        L=[f'{a} = {z}' for z in vals]+[f'print({a})'];truth=vals[-1]
    elif kind=='long_alias':
        # Explicit snapshots: old b persists after a overwrite; 8 effective assignments
        L=[f'{a} = {v0}',f'{b} = {a}',f'{a} = {v1}',f'{c} = {b}',f'{b} = {a}',f'{d} = {c}',f'{a} = {v2}',f'{e} = {d}',f'print({e})'];truth=v0
    elif kind=='nested3':
        L=[f'{a} = {v0}','def outer():',f'    {a} = {v1}','    def mid():',f'        {a} = {v2}','        def inner():',f'            return {a}', '        return inner()', '    return mid()','print(outer())'];truth=v2
    elif kind=='global_from_nested':
        L=[f'{a} = {v0}','def outer():','    def inner():',f'        global {a}',f'        {a} = {v1}','    inner()','outer()',f'print({a})'];truth=v1
    else: raise ValueError(kind)
    return '\n'.join(L)+'\n',truth


def _tokenize(source):
    toks=[];depth=[];level=0
    for t in tokenize.generate_tokens(io.StringIO(source).readline):
        if t.type==tokenize.ENCODING or t.type==tokenize.COMMENT:continue
        if t.type==tokenize.INDENT:
            level+=1;word='<INDENT>'
        elif t.type==tokenize.DEDENT:
            level-=1;word='<DEDENT>'
        elif t.type==tokenize.NEWLINE:
            word='<NL>'
        elif t.type in (tokenize.NL,tokenize.ENDMARKER):continue
        elif t.type in (tokenize.NAME,tokenize.NUMBER,tokenize.OP):word=t.string
        else:raise RuntimeError((t,source))
        assert word in ID, (word,t,source)
        toks.append(word);depth.append(level)
    toks.append('<CLS>');depth.append(0)
    return toks,[ID[t] for t in toks],depth


def sample(rng,kind):
    source,expected=_source(kind,rng)
    outputs=[]
    def safe_print(*args):
        assert len(args)==1 and type(args[0]) is int
        outputs.append(args[0])
    env={'__builtins__':{'print':safe_print}}
    exec(compile(source,'<verified_scope_program>','exec'),env)
    assert outputs==[expected],(kind,source,expected,outputs)
    toks,ids,depth=_tokenize(source)
    assert len(ids)<128,(kind,len(ids))
    ast_root=ast.parse(source)
    return Program(source,kind,expected,outputs[0],toks,ids,depth,
                   [t in NUMBERS for t in [int(w) if w.isdigit() else -1 for w in toks]],
                   sum(isinstance(z,(ast.Assign,ast.AnnAssign,ast.AugAssign)) for z in ast.walk(ast_root)),
                   max(depth))


def dataset(seed, n_per_kind, kinds=ALL_KINDS):
    rng=random.Random(seed)
    datasets={}
    for k in kinds:
        found=[];seen=set();tries=0
        while len(found)<n_per_kind:
            item=sample(rng,k)
            tries+=1
            if item.source not in seen:
                seen.add(item.source);found.append(item)
            assert tries<n_per_kind*100, (k, n_per_kind,tries)
        datasets[k]=found
    return datasets


def validate(n=128):
    data=dataset(20261117,n)
    for kind,ps in data.items():
        assert all(p.output==p.expect for p in ps)
        assert len({p.source for p in ps})==len(ps),(kind,'duplicate source')
    # These are causal counterexamples: same Python surface except global declaration
    a='a = 20\ndef fun():\n    a = 21\n    return a\nfun()\nprint(a)\n'
    b='a = 20\ndef fun():\n    global a\n    a = 21\n    return a\nfun()\nprint(a)\n'
    outputs=[]
    for code in (a,b):
        got=[];exec(code,{'__builtins__':{'print':lambda x:got.append(x)}})
        outputs.append(got[0])
    assert outputs==[20,21],outputs
    # Counterexample: compile-time Python local declaration overrides global value
    bad='a=20\ndef f():\n    b=a\n    a=21\n    return b\nf()\n'
    try:exec(bad,{'__builtins__':{}})
    except UnboundLocalError:pass
    else:raise AssertionError('Python static local must raise UnboundLocalError')
    return {'types':len(data),'per_type':n,'n_verified':sum(map(len,data.values())),
            'token_lengths':{k:[min(len(p.ids) for p in v),max(len(p.ids) for p in v)] for k,v in data.items()},
            'causal_global_pair':outputs,
            'unbound_local_semantics':'verified UnboundLocalError',
            'train':TRAIN_KINDS,'OOD':OOD_KINDS}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--per-kind',type=int,default=128);p.add_argument('--out',default='scope_benchmark_validation.json');a=p.parse_args()
    out=validate(a.per_kind)
    with open(a.out,'w') as f:json.dump(out,f,indent=2)
    print('SCOPE_BENCHMARK_VALIDATED='+json.dumps(out))

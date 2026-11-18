"""Hard executable counterfactuals beyond mere Python indentation *depth*.

Brother scopes can share indentation depth while having different lexical
binding IDs. Closures can share source code and lexical binding names but have
different runtime environment instances. Programs contain fixed terminating
source patterns and are checked against CPython, no oracle edges in inputs.
"""
import ast
import random
import json
from dataclasses import dataclass
from scope_program_benchmark import Program,VARIABLES,NUMBERS,_tokenize

HARD_KINDS=['sibling_fun','sibling_outer','sibling_closure_fun','sibling_closure_outer',
            'sibling_global','sibling_shadow_global','closure_many_first','closure_many_last',
            'nested_late','nested_snapshot']
HARD_PAIRS=[('sibling_fun','sibling_outer'),('sibling_closure_fun','sibling_closure_outer'),
            ('sibling_global','sibling_shadow_global'),('closure_many_first','closure_many_last'),
            ('nested_late','nested_snapshot')]

def source_for(kind,rng):
    a,b,c,d,e,f,g,h=rng.sample(VARIABLES[:8],8)
    v0,v1,v2,v3=rng.sample(NUMBERS,4)
    if kind in ('sibling_fun','sibling_outer'):
        lines=[f'{a} = {v0}',
            'def fun():',f'    {a} = {v1}',f'    return {a}',
            'def outer():',f'    {a} = {v2}',f'    return {a}',
            'print(fun())' if kind=='sibling_fun' else 'print(outer())']
        expected=v1 if kind=='sibling_fun' else v2
    elif kind in ('sibling_closure_fun','sibling_closure_outer'):
        lines=['def fun():',f'    {a} = {v0}',
            '    def inner():',f'        return {a}', '    return inner',
            'def outer():',f'    {a} = {v1}',
            '    def inner():',f'        return {a}', '    return inner',
            f'{b} = fun()',f'{c} = outer()',
            f'print({b if kind=="sibling_closure_fun" else c}())']
        expected=v0 if kind=='sibling_closure_fun' else v1
    elif kind in ('sibling_global','sibling_shadow_global'):
        lines=[f'{a} = {v0}', 'def fun():', f'    global {a}', f'    {a} = {v1}',
            'def outer():', f'    {a} = {v2}', f'    return {a}',
            'fun()',
            f'print({a})' if kind=='sibling_global' else 'print(outer())']
        expected=v1 if kind=='sibling_global' else v2
    elif kind in ('closure_many_first','closure_many_last'):
        vals=rng.sample(NUMBERS,6)
        lines=[f'def make({a}):', '    def inner():',f'        return {a}','    return inner']
        for name,value in zip([b,c,d,e,f,g],vals):lines.append(f'{name} = make({value})')
        lines.append(f'print({b if kind=="closure_many_first" else g}())')
        expected=vals[0] if kind=='closure_many_first' else vals[-1]
    elif kind in ('nested_late','nested_snapshot'):
        lines=[f'{a} = {v0}', 'def fun():', '    def inner():',f'        return {a}',
            '    return inner',f'{b} = fun()',
            f'{c} = {b}()' if kind=='nested_snapshot' else f'{a} = {v1}',
            f'{a} = {v1}' if kind=='nested_snapshot' else f'{c} = {b}()',f'print({c})']
        expected=v0 if kind=='nested_snapshot' else v1
    else:raise ValueError(kind)
    return '\n'.join(lines)+'\n',expected

def sample(rng,kind):
    source,expected=source_for(kind,rng)
    out=[]
    exec(compile(source,'<hard_python_scope_probe>','exec'),{'__builtins__':{'print':lambda x:out.append(x)}})
    assert out==[expected],(kind,source,expected,out)
    toks,ids,depth=_tokenize(source)
    assert len(ids)<=110,(kind,len(ids))
    parsed=ast.parse(source)
    return Program(source,kind,expected,out[0],toks,ids,depth,
        [tok.isdigit() for tok in toks],
        sum(isinstance(z,ast.Assign) for z in ast.walk(parsed)),max(depth))

def dataset(seed=95511,n=128):
    r=random.Random(seed)
    rows={}
    for kind in HARD_KINDS:
        values=[];seen=set()
        while len(values)<n:
            p=sample(r,kind)
            if p.source not in seen:
                seen.add(p.source);values.append(p)
        rows[kind]=values
    return rows

def pairs(seed=515166,n=64):
    r=random.Random(seed)
    out={}
    for k1,k2 in HARD_PAIRS:
        x=[];y=[]
        for i in range(n):
            state=r.getstate()
            p=sample(r,k1)
            r.setstate(state)
            q=sample(r,k2)
            assert p.output!=q.output
            x.append(p);y.append(q)
        out[k1+'__'+k2]=(x,y)
    return out

if __name__=='__main__':
    d=dataset();p=pairs()
    assert len(d)==10 and all(len(q)==128 for q in d.values())
    z={'types':len(d),'total':sum(map(len,d.values())),
       'pairs':{k:len(v[0]) for k,v in p.items()},
       'kinds':{k:{'min_len':min(len(q.ids) for q in v),
                    'max_len':max(len(q.ids) for q in v),
                    'max_depth':max(q.max_depth for q in v)} for k,v in d.items()},
       'exception_probe':'a=20; def f(): b=a; a=21; return b; f() => UnboundLocalError (covered in main audit)'}
    print('HARD_SCOPE_BENCHMARK_VALIDATED='+json.dumps(z))

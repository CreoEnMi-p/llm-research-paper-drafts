"""Compositional Python source grammar with CPython-execution oracle.

Mechanisms:
  A: assignment snapshot/aliasing
  B: rebinding by later writes
  C: distinct lexical scopes and shadowing
  D: closure runtime environment/late binding

Train only A+B, C+D; heldout (NEVER used for model selection): A+C.

No gold AST def-use, execution trace, name-binding id or scope identity in the
input. Syntax indentation is accessible (as in Python source). All classes
sample exactly four unique numeric literal candidates from 20..35, each
occurring once in the source; output always among those four.

This is a PROGRAM GRAMMAR, not a finite list of 20 fixed source templates:
case skeleton is formed by independently sampled production rules for variable
names, output expression, alias length, extra lexical scope, sibling functions,
noise statement placement, and function invocation. Finite grammar restrictions
remain; we do not claim coverage of arbitrary Python programs.
"""
from __future__ import annotations
import ast,hashlib,io,json,random,tokenize
from dataclasses import asdict,dataclass
from pathlib import Path
from typing import Optional
from scope_program_benchmark import VOCAB,ID,NUMBERS,VARIABLES,_tokenize

@dataclass
class Example:
    source:str
    composition:str
    selector:int
    truth:int
    source_literals:list[int]
    literal_occurrences:int
    unique_literal_candidates:int
    tokens:list[str]
    ids:list[int]
    depth:list[int]
    num_statements:int
    max_depth:int
    production:str
    pair_id:str

# Reduce implicit value position cue: randomly order distractors before/after
# the body where allowed. Every numeric value occurs once in every program.

def _build(rng:random.Random,combo:str,selector:int=0,settings:Optional[dict]=None):
    if settings is None:
        names=rng.sample(VARIABLES,12)
        vals=rng.sample(NUMBERS,4)
        spec={
          'names':names,'vals':vals,'alias_depth':rng.choice([1,2,3]),
          'noise_pos':rng.randrange(3),'production':rng.randrange(4),
          'return_indirect':bool(rng.randrange(2)),
          'noise_swap':bool(rng.randrange(2)),
        }
    else:spec=dict(settings)
    n=spec['names'];values=spec['vals'];v0,v1,v2,v3=values
    x,y,z,w,n0,n1,n2,n3,t0,t1,t2,t3=n
    noise=[f'{n0} = {v2}',f'{n1} = {v3}']
    if spec['noise_swap']:noise.reverse()
    # The four literal occurrences are deliberately always distinct.
    # Noise is inserted either before, inside (where legal), or after the
    # mechanism body, making first/last-value shortcuts non-universal.
    lines=[]
    def add_noise(where):
        if spec['noise_pos']==where:lines.extend(noise)
    production=''
    if combo=='AB':
        add_noise(0)
        # A: y = x snapshot, alias chain. B: x overwritten later.
        lines.append(f'{x} = {v0}')
        prev=x
        aliases=[y,z,w]
        for j in range(spec['alias_depth']):
            lines.append(f'{aliases[j]} = {prev}')
            prev=aliases[j]
        saved=prev
        add_noise(1)
        lines.append(f'{x} = {v1}')
        # Extra variable copies maintain semantics and vary dependency depth.
        if spec['return_indirect']:
            lines.append(f'{t0} = {x}')
            latest=t0
        else:latest=x
        add_noise(2)
        query=saved if selector==0 else latest
        lines.append(f'print({query})')
        truth=v0 if selector==0 else v1
        production='alias_snapshot_rebind'
    elif combo=='CD':
        # C: local x or parameter x shadows global x.
        # D: closure is invoked after creation, with active frame state.
        if spec['production'] % 2==0:
            add_noise(0)
            lines.append(f'{x} = {v0}')
            lines.append('def fun():')
            lines.append(f'    {x} = {v1}')
            lines.append('    def inner():')
            lines.append(f'        return {x}')
            if spec['return_indirect']:
                lines.append(f'    {t0} = inner')
                lines.append(f'    return {t0}')
            else:lines.append('    return inner')
            lines.append(f'{t1} = fun()')
            # v2,v3 appear once in external nuisance writes.
            add_noise(1)
            add_noise(2)
            query=f'{t1}()' if selector==0 else x
            lines.append(f'print({query})')
            truth=v1 if selector==0 else v0
            production='shadowed_closure_late_read'
        else:
            # Two closures with same source function but different invocation
            # frames. Constant v0 in module x; calls use v1 and v2; v3 noise.
            lines.append(f'{x} = {v0}')
            lines.append(f'def make({x}):')
            lines.append('    def inner():')
            lines.append(f'        return {x}')
            lines.append('    return inner')
            lines.append(f'{t1} = make({v1})')
            lines.append(f'{t2} = make({v2})')
            # noise v3 once (v2 already used), and don't call add_noise.
            lines.append(f'{n0} = {v3}')
            query=f'{t1}()' if selector==0 else f'{t2}()'
            lines.append(f'print({query})')
            truth=v1 if selector==0 else v2
            production='two_runtime_closure_instances'
    elif combo=='AC':
        # This combination is frozen and unseen in training.
        # A: an alias of outer x, C: a local x under lexical shadow.
        if spec['production']==0:
            add_noise(0)
            lines.append(f'{x} = {v0}')
            lines.append(f'{y} = {x}')
            lines.append('def fun():')
            lines.append(f'    {x} = {v1}')
            lines.append(f'    {z} = {y}')
            if spec['return_indirect']:
                lines.append(f'    {w} = {x}')
                lines.append(f'    return {w}')
            else:lines.append(f'    return {x}')
            lines.append(f'{t1} = fun()')
            add_noise(1)
            add_noise(2)
            query=y if selector==0 else t1
            lines.append(f'print({query})')
            truth=v0 if selector==0 else v1
            production='alias_crosses_local_shadow'
        elif spec['production']==1:
            # Identical indentation depth but distinct sibling lexical scopes.
            # Alias source y still points to the module x snapshot.
            lines.append(f'{x} = {v0}')
            lines.append(f'{y} = {x}')
            lines.append('def fun():')
            lines.append(f'    {x} = {v1}')
            lines.append(f'    return {x}')
            lines.append('def outer():')
            lines.append(f'    {x} = {v2}')
            lines.append(f'    return {x}')
            lines.append(f'{n0} = {v3}')
            query=y if selector==0 else ('fun()' if spec['return_indirect'] else 'outer()')
            lines.append(f'print({query})')
            truth=v0 if selector==0 else (v1 if spec['return_indirect'] else v2)
            production='sibling_scope_alias'
        elif spec['production']==2:
            # C: function parameter binding is distinct from same-name global.
            add_noise(0)
            lines.extend([f'{x} = {v0}',f'{y} = {x}',f'def fun({x}):'])
            if spec['return_indirect']:
                lines.extend([f'    {z} = {x}',f'    return {z}'])
            else:lines.append(f'    return {x}')
            lines.append(f'{t1} = fun({v1})')
            add_noise(1);add_noise(2)
            query=y if selector==0 else t1
            lines.append(f'print({query})')
            truth=v0 if selector==0 else v1
            production='alias_and_parameter_shadow'
        elif spec['production']==3:
            # A alias from module, C local x under two nested lexical scopes.
            add_noise(0)
            lines.extend([f'{x} = {v0}',f'{y} = {x}', 'def outer():',f'    {x} = {v1}', '    def inner():'])
            if spec['return_indirect']:
                lines.extend([f'        {z} = {x}',f'        return {z}'])
            else:lines.append(f'        return {x}')
            lines.extend(['    return inner()',f'{t1} = outer()'])
            add_noise(1);add_noise(2)
            query=y if selector==0 else t1
            lines.append(f'print({query})')
            truth=v0 if selector==0 else v1
            production='alias_nested_shadow'
    else:raise ValueError(combo)
    source='\n'.join(lines)+'\n'
    return source,truth,production,spec

def _safe_exec(src):
    outputs=[]
    def printer(*args):
        assert len(args)==1 and type(args[0]) is int
        outputs.append(args[0])
    builtins={'print':printer}
    exec(compile(src,'<composition_oracle>','exec'),{'__builtins__':builtins})
    assert len(outputs)==1
    return outputs[0]

def sample(rng,combo,selector=None,settings=None):
    if selector is None:selector=rng.randrange(2)
    src,expected,prod,settings=_build(rng,combo,selector,settings)
    actual=_safe_exec(src)
    assert actual==expected,(combo,selector,prod,src,expected,actual)
    tok,ids,depth=_tokenize(src)
    if len(ids)>110:raise ValueError(('Program too long for pilot model',len(ids),src))
    tree=ast.parse(src)
    literals=[int(q.value) for q in ast.walk(tree) if isinstance(q,ast.Constant) and type(q.value)==int and q.value in NUMBERS]
    assert len(literals)==len(set(literals))==4,(combo,prod,src,literals)
    assert expected in literals
    return Example(src,combo,selector,expected,sorted(literals),len(literals),len(set(literals)),tok,ids,depth,
                   sum(isinstance(z,ast.Assign) for z in ast.walk(tree)),max(depth),prod,
                   hashlib.sha256(src.rsplit('print(',1)[0].encode()).hexdigest()[:16])

def paired(rng,combo):
    state=rng.getstate()
    x=sample(rng,combo,selector=0)
    rng.setstate(state)
    y=sample(rng,combo,selector=1)
    assert x.truth!=y.truth,(x.source,y.source)
    assert x.source.rsplit('print(',1)[0]==y.source.rsplit('print(',1)[0]
    assert x.literal_occurrences==y.literal_occurrences==4
    identifier=hashlib.sha256(x.source.rsplit('print(',1)[0].encode()).hexdigest()[:16]
    x.pair_id=y.pair_id=identifier
    return x,y

def make_dataset(combo,seed,n,as_pairs=False):
    r=random.Random(seed)
    xs=[];seen=set();tries=0
    while len(xs)<n:
        tries+=1
        if tries>n*100:raise RuntimeError('insufficient unique grammar expansions')
        try:obj=paired(r,combo) if as_pairs else sample(r,combo)
        except ValueError as e:
            if 'Program too long' in str(e):continue
            raise
        key=obj[0].pair_id if as_pairs else obj.source
        if key in seen:continue
        xs.append(obj);seen.add(key)
    return xs

def describe(xs):
    flat=[y for x in xs for y in x] if xs and isinstance(xs[0],tuple) else xs
    from collections import Counter
    return {'n':len(flat),'candidates':dict(Counter(p.unique_literal_candidates for p in flat)),
      'source_len_min':min(len(p.ids) for p in flat),'source_len_max':max(len(p.ids) for p in flat),
      'production_types':dict(Counter(p.production for p in flat)),
      'answer_counts':dict(Counter(p.truth for p in flat)),
      'pair_count':len(xs) if xs and isinstance(xs[0],tuple) else 0}

if __name__=='__main__':
    for combo in ('AB','CD','AC'):
        for paired_flag in [False,True]:
            rows=make_dataset(combo,20261117,64,paired_flag)
            print(combo,'pairs' if paired_flag else 'single',describe(rows))

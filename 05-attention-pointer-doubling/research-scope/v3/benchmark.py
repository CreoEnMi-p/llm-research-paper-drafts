"""V3 held-out mechanism-composition grammar. CPython is used for labels ONLY.

Seven predeclared production families combine lexical scope, global/nonlocal,
late closure reads, assignment snapshots and distinct invocation frames.
Every paired intervention changes only the final print expression; the
pre-query source and four distinct numeric source literals are identical.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import io
import json
import random
import tokenize
from pathlib import Path

FAMILIES = (
    'sibling_scope_reads',
    'global_nonlocal_collision',
    'late_closure_snapshot',
    'double_frame_rebinding',
    'nonlocal_scope_snapshot',
    'nested_local_shadow',
    'global_late_snapshot',
)
STRATA = ('seen_names', 'unseen_main_name')
NUMBERS = tuple(range(20, 36))
KNOWN = tuple('abcdefghijkl')
UNSEEN = tuple('mnop')


def num_tokens(source):
    return sum(t.type not in (tokenize.ENCODING, tokenize.COMMENT, tokenize.NL,
                              tokenize.ENDMARKER) for t in
               tokenize.generate_tokens(io.StringIO(source).readline)) + 1


def _execute(source):
    output = []
    def printer(*args):
        assert len(args) == 1 and type(args[0]) is int
        output.append(args[0])
    exec(compile(source, '<frozen_v3_oracle>', 'exec'),
         {'__builtins__': {'print': printer}})
    assert len(output) == 1
    return output[0]


def _two(rng, family, stratum):
    main = rng.choice(UNSEEN if stratum == 'unseen_main_name' else KNOWN)
    others = rng.sample([k for k in KNOWN if k != main], 8)
    x, y, z, w, u, n0, n1, n2, n3 = (main, *others)
    v0, v1, v2, v3 = rng.sample(NUMBERS, 4)
    noise = f'{n0} = {v3}'
    lines = []
    def add_noise():
        if rng.randrange(2):
            lines.insert(0, noise)
        else:
            lines.append(noise)
    if family == 'sibling_scope_reads':
        lines = [f'{x} = {v0}']
        aa = ['def fun():', f'    {x} = {v1}', f'    {y} = {x}', f'    return {y}']
        bb = ['def outer():', f'    {x} = {v2}', f'    {y} = {x}', f'    return {y}']
        if rng.randrange(2): aa, bb = bb, aa
        lines += aa + bb
        calls = [f'{z} = fun()', f'{w} = outer()']
        if rng.randrange(2): calls.reverse()
        lines += calls
        add_noise()
        choices = ((z, v1), (w, v2))
    elif family == 'global_nonlocal_collision':
        lines = [f'{x} = {v0}']
        aa = ['def outer():', f'    {x} = {v1}', '    def inner():',
              f'        nonlocal {x}', f'        {x} = {v2}',
              '    inner()', f'    return {x}']
        bb = ['def fun():', f'    global {x}', f'    {x} = {v3}', f'    return {x}']
        if rng.randrange(2): aa, bb = bb, aa
        lines += aa + bb
        calls = [f'{z} = outer()', f'{w} = fun()']
        if rng.randrange(2): calls.reverse()
        lines += calls
        choices = ((z, v2), (x, v3))
    elif family == 'late_closure_snapshot':
        lines = [f'{x} = {v0}', f'{y} = {x}', 'def make():',
                 '    def inner():', f'        return {x}',
                 '    return inner', f'{z} = make()', f'{x} = {v1}',
                 f'{n0} = {v2}', f'{n1} = {v3}']
        if rng.randrange(2): lines[-2:] = lines[-2:][::-1]
        choices = ((f'{z}()', v1), (y, v0))
    elif family == 'double_frame_rebinding':
        lines = [f'{x} = {v0}', f'def make({x}):',
                 '    def inner():', f'        return {x}',
                 '    return inner', f'{y} = make({x})',
                 f'{x} = {v1}', f'{z} = make({x})',
                 f'{x} = {v2}', noise]
        choices = ((f'{y}()', v0), (f'{z}()', v1))
    elif family == 'nonlocal_scope_snapshot':
        lines = [f'{x} = {v2}', 'def outer():', f'    {x} = {v0}',
                 f'    {y} = {x}', '    def inner():', f'        nonlocal {x}',
                 f'        {x} = {v1}', '    inner()', f'    return {x}',
                 f'{z} = outer()', f'{w} = {x}']
        add_noise()
        choices = ((z, v1), (w, v2))
    elif family == 'nested_local_shadow':
        lines = [f'{x} = {v0}', f'{y} = {x}', 'def outer():',
                 f'    {x} = {v1}', '    def inner():', f'        {x} = {v2}',
                 f'        return {x}', '    return inner()',
                 f'{z} = outer()', f'{x} = {v3}']
        choices = ((y, v0), (z, v2))
    elif family == 'global_late_snapshot':
        lines = [f'{x} = {v0}', 'def make():', '    def inner():',
                 f'        return {x}', '    return inner', f'{y} = make()',
                 'def fun():', f'    global {x}', f'    {x} = {v1}',
                 'fun()', f'{z} = {x}', f'{x} = {v2}', noise]
        choices = ((f'{y}()', v2), (z, v1))
    else:
        raise ValueError(family)
    if rng.randrange(2): choices = choices[::-1]
    prefix = '\n'.join(lines) + '\n'
    left, right = (prefix + f'print({q})\n' for q, _ in choices)
    lval, rval = choices[0][1], choices[1][1]
    assert _execute(left) == lval and _execute(right) == rval
    assert lval != rval
    for src in (left, right):
        lits = [z.value for z in ast.walk(ast.parse(src))
                if isinstance(z, ast.Constant) and type(z.value) is int]
        assert len(lits) == len(set(lits)) == 4, (family, src, lits)
        assert max(num_tokens(src), 1) <= 192, (family, num_tokens(src))
    # Strict single-expression intervention, source body otherwise bit-identical.
    assert left.rsplit('print(', 1)[0] == right.rsplit('print(', 1)[0]
    return {'family': family, 'stratum': stratum,
            'pair_id': hashlib.sha256(prefix.encode()).hexdigest()[:24],
            'left': {'source': left, 'label': lval},
            'right': {'source': right, 'label': rval},
            'literal_candidates': sorted((v0, v1, v2, v3))}


def generate(seed, pairs_per_cell):
    rows = []
    seen = set()
    for fi, family in enumerate(FAMILIES):
        for si, stratum in enumerate(STRATA):
            rng = random.Random(seed + 1000003 * fi + 10007 * si)
            n = 0
            while n < pairs_per_cell:
                row = _two(rng, family, stratum)
                if row['pair_id'] in seen:
                    continue
                seen.add(row['pair_id'])
                rows.append(row)
                n += 1
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=2026112304)
    parser.add_argument('--per-cell', type=int, default=32)
    args = parser.parse_args()
    rows = generate(args.seed, args.per_cell)
    data = ''.join(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n'
                   for row in rows).encode()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(json.dumps({'n_pairs': len(rows), 'families': len(FAMILIES),
                      'strata': len(STRATA), 'sha256': hashlib.sha256(data).hexdigest(),
                      'max_source_tokens': max(num_tokens(r['left']['source']) for r in rows)}))


if __name__ == '__main__':
    main()

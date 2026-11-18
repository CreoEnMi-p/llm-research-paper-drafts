"""Source-only heuristic baselines; per-program candidate counts and exact chance.
No CPython execution is used to generate predictions. CPython already verified
labels in dataset. Random chance is conditional on *available unique literal
values*, distinct from uniform output-class 1/16.
"""
import argparse,ast,random,json,statistics,sys,math
from pathlib import Path
from collections import Counter
from scope_program_benchmark import dataset,NUMBERS
from scope_hard_probes import dataset as hard_dataset,pairs as hard_pairs

p=argparse.ArgumentParser()
p.add_argument('--n',type=int,default=128)
p.add_argument('--out',type=str,default='chance_heuristics_verified.json')
a=p.parse_args()

def get_nums(source):
    tree=ast.parse(source)
    occ=[int(n.value) for n in ast.walk(tree) if isinstance(n,ast.Constant) and type(n.value)==int and int(n.value) in NUMBERS]
    # lexical order (ast.walk order not guaranteed to reflect lexical sort for all AST node types)
    occ=[v for _,_,v in sorted((n.lineno,n.col_offset,int(n.value)) for n in ast.walk(tree) if isinstance(n,ast.Constant) and type(n.value)==int and int(n.value) in NUMBERS)]
    assert occ
    return occ

def heuristics(source):
    literals=get_nums(source)
    # Simple source AST walk, never follows scopes/calls/control flow.
    tree=ast.parse(source)
    assignments=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            assignments.append((n.lineno,n.targets[0].id,n.value))
    assignments.sort(key=lambda z:z[0])
    # last source-level print arg, not print value (no Python exec)
    prints=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='print']
    printname=prints[-1].args[0].id if prints and isinstance(prints[-1].args[0],ast.Name) else None
    last_literal_write=next((int(n.value) for _,_,n in reversed(assignments) if isinstance(n,ast.Constant) and type(n.value)==int),literals[-1])
    # source-ordered, depth-unaware last constant assignment on matching name, else fallback
    same_name=next((int(n.value) for _,name,n in reversed(assignments) if name==printname and isinstance(n,ast.Constant) and type(n.value)==int),last_literal_write)
    count=Counter(literals)
    most_common=sorted(count,key=lambda v:(-count[v],v))[0]
    return dict(first_literal=literals[0],last_literal=literals[-1],last_constant_assignment=last_literal_write,scope_blind_same_name=same_name,most_frequent_literal=most_common),literals

def analyze(name,items):
    preds=[];counts=[];uniques=[];truth=[]
    for z in items:
        pr,lit=heuristics(z.source)
        pred={k:int(v==z.output) for k,v in pr.items()}
        preds.append(pred);counts.append(len(lit));uniques.append(len(set(lit)));truth.append(z.output)
        assert z.output in lit,(name,z.source,z.output,lit)
    n=len(items)
    return {'n':n,'mean_candidate_occurrences':statistics.mean(counts),
            'min_candidate_occurrences':min(counts),'max_candidate_occurrences':max(counts),
            'mean_unique_candidates':statistics.mean(uniques),
            'min_unique_candidates':min(uniques),'max_unique_candidates':max(uniques),
            'candidate_occurrence_histogram':dict(sorted(Counter(counts).items())),
            'unique_candidate_histogram':dict(sorted(Counter(uniques).items())),
            'class_uniform_chance':1/len(NUMBERS),
            'unique_candidate_uniform_chance':statistics.mean(1/u for u in uniques),
            'occurrence_uniform_chance':statistics.mean(1/c for c in counts),
            'empirical_most_common_label_frequency':max(Counter(truth).values())/n,
            'heuristics':{k:sum(p[k] for p in preds)/n for k in preds[0]}}

train=dataset(884311,a.n)
hard=hard_dataset(95511,a.n)
rows={k:analyze(k,v) for k,v in {**train,**hard}.items()}
pairs=hard_pairs(515166,64)
pair_rows={}
for name,(left,right) in pairs.items():
    lhs=[heuristics(p.source) for p in left];rhs=[heuristics(p.source) for p in right]
    pa={};methods=lhs[0][0].keys()
    for method in methods:
        predictions=[(z[0][method],w[0][method]) for z,w in zip(lhs,rhs)]
        pa[method]={
            'both_correct':sum(p==x.output and q==y.output for (p,q),x,y in zip(predictions,left,right))/len(left),
            'prediction_flip':sum(p!=q for p,q in predictions)/len(left),
            'prediction_invariant':sum(p==q for p,q in predictions)/len(left),
        }
    pair_rows[name]={'n_pairs':len(left), 'labels_differ_fraction':sum(x.output!=y.output for x,y in zip(left,right))/len(left),
        'chance_independent_uniform_candidates_both':statistics.mean(1/(len(set(x[1]))*len(set(y[1]))) for x,y in zip(lhs,rhs)),
        'chance_class_uniform_independent_both':1/len(NUMBERS)**2,
        'heuristics':pa}
result={'protocol':'Old 20 hand-crafted CPython-executed templates, 10 additional hard templates, frozen preexisting validation seeds and CPython labels; source-only heuristics cannot call runtime exec, get gold kind, or use binding IDs.',
        'chances':{'class_uniform':1/16,'per_source_unique_literals':'E[1 / #distinct numeric candidates]','per_source_occurrence_uniform':'E[1 / #numeric occurrence positions] (different prior)',
             'paired_independent_unique_literals':'E[1/(unique candidates left * right)]','paired_all_class':1/256},
        'classes':rows,'paired':pair_rows}
Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2))
print('AUDIT_HEURISTICS='+json.dumps({'n_categories':len(rows),'n_hard_pairs':len(pair_rows),'kind_summary':{k:{'unique_candidates':round(v['mean_unique_candidates'],2),'candidate_chance':round(v['unique_candidate_uniform_chance'],4),'last_literal':v['heuristics']['last_literal']} for k,v in rows.items()}},ensure_ascii=False))

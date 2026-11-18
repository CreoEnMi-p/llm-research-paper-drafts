"""Post-hoc paired comparison of neural predictions vs prespecified first literal heuristic.
Do not use to tune architecture on consumed V2.
"""
from pathlib import Path
import random,json,ast,statistics
ROOT=Path(__file__).resolve().parent
pairs=[json.loads(line) for line in (ROOT/'frozen_protocol_v2/FROZEN_AC_TEST_256pairs.jsonl').read_text().splitlines()]
first={}
for r in pairs:
    a=r['left'];b=r['right']
    def first_num(source):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node,ast.Constant) and type(node.value) is int:
                return node.value
        raise AssertionError
    pa,pb=first_num(a['source']),first_num(b['source'])
    assert pa==pb
    first[r['pair_id']]={'production':a['production'],'first':pa,'baseline_acc':((pa==a['label'])+(pa==b['label']))/2,'both_correct':False}
assert abs(statistics.mean(x['baseline_acc'] for x in first.values())-.373046875)<1e-12
rows={}
for model in ('transformer','pointer-depth'):
    for seed in range(5):
        data=json.loads((ROOT/f'grammar_{model}_s{seed}.json').read_text())
        preds=data['frozen_test']['pair_predictions']
        for p in preds:
            assert p['pair_id'] in first
            b=first[p['pair_id']]
            v=(p['correct_left']+p['correct_right'])/2-b['baseline_acc']
            rows[(model,seed,p['pair_id'])]=(b['production'],v)

rng=random.Random(2026111807)
result={}
productions=sorted({x['production'] for x in first.values()})
for model in ('transformer','pointer-depth'):
    vals=[statistics.mean(v for (m,s,p),(k,v) in rows.items() if m==model and s==seed) for seed in range(5)]
    # cluster over independent grammar production families, then independent training seeds
    B=10000;boot=[]
    means_by_seed_prod={(s,k):statistics.mean(v for (m,sd,pid),(prod,v) in rows.items() if m==model and sd==s and prod==k) for s in range(5) for k in productions}
    for _ in range(B):
        seeds=rng.choices(range(5),k=5)
        prods=rng.choices(productions,k=4)
        seeds_means=[]
        for s in seeds:
            means=[]
            for prod in prods:
                means.append(means_by_seed_prod[(s,prod)])
            seeds_means.append(statistics.mean(means))
        boot.append(statistics.mean(seeds_means))
    boot.sort()
    result[model]={'seed_delta_vs_first_literal':vals,'seed_mean_delta':statistics.mean(vals),
                    'production_seed_bootstrap_95':[boot[int(.025*B)],boot[int(.975*B)]],
                    'n_seed':5,'n_production_clusters':len(productions)}
out={'protocol':'Post-hoc descriptive comparison, consumed frozen V2, first-literal script fixed before evaluation; do not tune using results',
     'model_delta_vs_first_literal':result}
(ROOT/'frozen_v2_posthoc_neural_vs_script.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))

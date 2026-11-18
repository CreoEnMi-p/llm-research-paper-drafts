"""Frozen grammar A+C results with 5-seed paired-metric statistics and
bootstrap resampling WHOLE production families, never token positions.

Reads predictions for both counterfactual members, not just pair accuracy.
Model settings were locked before V2 data was first inspected.
"""
import argparse,json,math,random,statistics
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',default='grammar_frozen_5seed_summary.json');p.add_argument('--reps',type=int,default=5000);a=p.parse_args()
root=Path(__file__).resolve().parent
models=('transformer','pointer-depth')
rows={}
for model in models:
    seeds=[]
    for seed in range(5):
        f=root/f'grammar_{model}_s{seed}.json'
        z=json.loads(f.read_text());zs=z['frozen_test']
        assert z['train_compositions']==['AB','CD'] and z['test_composition']=='AC'
        assert z['config']['steps']==1000 and z['config']['batch']==32
        assert z['frozen_test_sha256']==json.loads((root/'frozen_protocol_v2/PROTOCOL_MANIFEST.json').read_text())['frozen_test_sha256']
        seeds.append({'seed':seed,'train_seconds':z['train_seconds'],'params':z['params'],
            'by_production':zs['by_production'],'overall':zs['overall'],'predictions':zs['pair_predictions'],
            'dev':z['curve'][-1]['dev']})
    products=sorted(seeds[0]['by_production'])
    assert len(products)==4 and all(sorted(z['by_production'])==products for z in seeds)
    props=['acc','both_correct','pred_flip','pred_invariant']
    metrics={k:{'mean':statistics.mean(z['overall'][k] for z in seeds),
                'sd_across_seeds':statistics.stdev(z['overall'][k] for z in seeds),
                'seed_values':[z['overall'][k] for z in seeds]} for k in props}
    # Proper entire-template cluster bootstrap, cross with SEED for uncertainty.
    rng=random.Random(17711+(model=='pointer-depth'))
    boot={key:[] for key in props}
    for _ in range(a.reps):
        prod_sel=rng.choices(products,k=len(products))
        seed_sel=rng.choices(seeds,k=len(seeds))
        for key in props:
            boot[key].append(statistics.mean(statistics.mean(s['by_production'][kind][key] for kind in prod_sel) for s in seed_sel))
    for key in props:
        z=sorted(boot[key]);metrics[key]['cluster_seed_bootstrap_95']=[z[int(.025*a.reps)],z[int(.975*a.reps)]]
    groups={k:{t:{'mean':statistics.mean(s['by_production'][k][t] for s in seeds),
                  'seed_values':[s['by_production'][k][t] for s in seeds]} for t in props}
            for k in products}
    rows[model]={'n_seeds':5,'n_test_pairs_per_seed':256,'n_independent_production_clusters':4,
                 'metrics':metrics,'by_production':groups,'seeds':[{'seed':z['seed'],'train_seconds':z['train_seconds'],
                    'params':z['params'],'dev':z['dev'],'overall':z['overall']} for z in seeds]}
# Paired seed deltas; both models same eval sources and matching seed numbers
for key in ['acc','both_correct','pred_flip','pred_invariant']:
    d=[rows['pointer-depth']['metrics'][key]['seed_values'][i]-rows['transformer']['metrics'][key]['seed_values'][i] for i in range(5)]
    rows['pointer-depth']['metrics'][key]['delta_vs_transformer_by_seed']=d
    rows['pointer-depth']['metrics'][key]['delta_vs_transformer_mean']=statistics.mean(d)
script=json.loads((root/'frozen_v2_script_heuristics.json').read_text())
obj={'protocol':'Locked V2 A+C grammar composition, 4 productions, 256 paired programs, 5 seeds per model. Source had exactly four distinct literals (25% conditional uniform single value guess, 6.25% independent pair, 8.33% distinct-label-aware without replacement). The novel test used once per prespecified model. Bootstrap samples production clusters and seed; only FOUR independent grammar productions, confidence very wide.','test_sha256':json.loads((root/'frozen_protocol_v2/PROTOCOL_MANIFEST.json').read_text())['frozen_test_sha256'],
     'script_heuristics':script['overall'], 'results':rows,
     'warnings':['Old V1 was evaluated by a smoke run; retired before V2 freeze and model lock.','A+C heldout grammar is a finite compositional symbolic language, not production-source code; 4 production clusters are not enough for ICML generalization.','Only two pre-registered small architectures tested; optimized L5/RoPE, compiler-graph and pretrained code LM still absent.','Source literal pointer sees a restricted output candidate space, a powerful prior; do not compare to 16-class random chance.','Prediction invariance means model prediction did NOT change when query changed; for these pairs gold labels ALWAYS differ, so invariance is a failure diagnostic.','Conditioning on different gold labels, a uniformly selected distinct ordered pair has 1/(4*3)=8.33% chance both; independent draws 1/16=6.25%.']}
Path(a.out).write_text(json.dumps(obj,indent=2))
for model,z in rows.items():
 print(model,'accuracy',z['metrics']['acc'],'both',z['metrics']['both_correct'],'flip',z['metrics']['pred_flip'],'invariant',z['metrics']['pred_invariant'])
 print('CI', {k:v['cluster_seed_bootstrap_95'] for k,v in z['metrics'].items()})
print('WROTE',a.out)

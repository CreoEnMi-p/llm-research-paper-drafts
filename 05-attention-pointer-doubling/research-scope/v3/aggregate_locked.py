"""Preregistered V3 cluster×seed aggregation. Never fit models or select architectures."""
import collections
import hashlib
import json
import random
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent
lock=json.loads((ROOT/'PROTOCOL_LOCK.json').read_text())
manifest=json.loads((ROOT/'FROZEN_MANIFEST.json').read_text())
assert manifest['test_sha256']==lock['frozen_test']['expected_sha256']
assert hashlib.sha256((ROOT/'FROZEN_V3_448pairs.jsonl').read_bytes()).hexdigest()==manifest['test_sha256']
ARMS=lock['model_lock']['arms']
SEEDS=lock['model_lock']['seeds']
FAMS=lock['frozen_test']['families']
STRATA=lock['frozen_test']['name_strata']
data={}
for arm in ARMS:
    data[arm]={}
    for seed in SEEDS:
        path=ROOT/f'V3_MODEL_{arm}_s{seed}.json'
        d=json.loads(path.read_text())
        assert d['arm']==arm and d['seed']==seed and d['steps']==5000
        z=d['sealed_v3']
        assert z['test_sha256']==manifest['test_sha256']
        assert len(z['predictions'])==448
        assert len({r['pair_id'] for r in z['predictions']})==448
        data[arm][seed]=z['predictions']

def metric(items):
    return {
      'both':statistics.mean(r['both'] for r in items),
      'single':statistics.mean(r['acc'] for r in items),
      'flip':statistics.mean(r['flip'] for r in items),
      'correct_position_mass':(
        statistics.mean(r['correct_position_mass'] for r in items)
        if all(r['correct_position_mass'] is not None for r in items) else None)
    }

summary={}
by_fam={}
by_stratum={}
allby={}
for arm in ARMS:
    z=[r for seed in SEEDS for r in data[arm][seed]]
    summary[arm]=metric(z)
    by_fam[arm]={f:metric([r for r in z if r['family']==f]) for f in FAMS}
    by_stratum[arm]={s:metric([r for r in z if r['stratum']==s]) for s in STRATA}
    allby[arm]={seed:{f:statistics.mean(r['both'] for r in data[arm][seed]
                                    if r['family']==f) for f in FAMS}
                  for seed in SEEDS}

def boot_arm(arm,reps=5000):
    rand=random.Random(202611231)
    point=statistics.mean(allby[arm][s][f] for s in SEEDS for f in FAMS)
    draws=[]
    for i in range(reps):
        fams=rand.choices(FAMS,k=len(FAMS))
        seeds=rand.choices(SEEDS,k=len(SEEDS))
        draws.append(statistics.mean(allby[arm][s][f] for s in seeds for f in fams))
    return {'point':point,'percentile95':[sorted(draws)[int(.025*reps)],
                                           sorted(draws)[int(.975*reps)]]}

def boot_delta(a,b,reps=5000):
    rand=random.Random(202611232)
    d={(s,f):allby[a][s][f]-allby[b][s][f] for s in SEEDS for f in FAMS}
    point=statistics.mean(d.values())
    draws=[]
    for i in range(reps):
        fams=rand.choices(FAMS,k=len(FAMS))
        seeds=rand.choices(SEEDS,k=len(SEEDS))
        draws.append(statistics.mean(d[s,f] for s in seeds for f in fams))
    draws.sort()
    return {'point':point,'percentile95':[draws[int(.025*reps)],draws[int(.975*reps)]]}

intervals={arm:boot_arm(arm) for arm in ARMS}
diffs={other:boot_delta('pointer',other) for other in ARMS if other!='pointer'}

heur=json.loads((ROOT/'V3_SYMBOLIC_HEURISTIC_AUDIT.json').read_text())
assert heur['test_sha256']==manifest['test_sha256']
shortcuts=heur['selected_heuristic_v3']['both']
success=(summary['pointer']['both']>=.70
  and all(by_fam['pointer'][f]['both']>=.50 for f in FAMS)
  and all(diffs[arm]['percentile95'][0]>0 for arm in ('pointer-scrambled','candidate-classifier'))
  and summary['pointer']['both']>shortcuts)
out={
 'protocol':lock['protocol'],'test_sha256':manifest['test_sha256'],
 'n_test_pairs':448,'n_families':7,'n_training_seeds':5,
 'summaries':summary,'family_metrics':by_fam,'name_stratum_metrics':by_stratum,
 'cluster_seed_bootstrap':intervals,'paired_differences_vs_pointer':diffs,
 'symbolic_ceiling':heur['ast_symbolic_ceiling'],
 'preselected_source_heuristic':{'policy':heur['dev_policy_preselection']['policy'],
                                  'v3':heur['selected_heuristic_v3']},
 'research_go_precondition':success,
 'limitations':'7 family clusters, finite grammar; exact symbolic AST upper bound. No published graph/AST neural model or published symbol-invariant model has been reproduced here.'
}
(ROOT/'V3_FIVE_SEED_AGGREGATE.json').write_text(json.dumps(out,indent=2))
for arm in ARMS:
    print('V3_FINAL',arm,summary[arm],
          'bootstrap',intervals[arm],flush=True)
for f in FAMS:
    print('V3_FAMILY',f,
          {arm:round(by_fam[arm][f]['both'],5) for arm in ARMS},flush=True)
print('V3_PAIRED_DELTAS',json.dumps(diffs),flush=True)
print('V3_RESEARCH_GO',success,flush=True)

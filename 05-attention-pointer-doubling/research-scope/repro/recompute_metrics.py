"""Recompute old pilot tables with candidate-conditioned chance, source-only
heuristics, paired prediction INVARIANCE, and resampling whole grammar kinds.
Do not present old heldout templates as a fresh untouched test set.
"""
import argparse,json,random,statistics,math
from pathlib import Path
from scope_program_benchmark import TRAIN_KINDS,OOD_KINDS
from scope_hard_probes import HARD_KINDS
p=argparse.ArgumentParser();p.add_argument('--out',default='old_recomputed_tables.json');a=p.parse_args()
root=Path(__file__).resolve().parent
base=json.loads((root/'chance_heuristics_verified.json').read_text())
models=['transformer','transformer-depth','pointer','depth-pointer','pointer-depth-encoding-only']
rng=random.Random(20261118)
reps=5000

def basic(kinds):
    r={}
    for label,ks in [('train',TRAIN_KINDS),('ood',OOD_KINDS),('hard',HARD_KINDS)]:
        r[label]={'mean_unique_candidate_count':statistics.mean(kinds[k]['mean_unique_candidates'] for k in ks),
                  'class_uniform_chance':1/16,
                  'mean_uniform_unique_literal_chance':statistics.mean(kinds[k]['unique_candidate_uniform_chance'] for k in ks),
                  'mean_uniform_occurrence_chance':statistics.mean(kinds[k]['occurrence_uniform_chance'] for k in ks),
                  'heuristic_macro':{key:statistics.mean(kinds[k]['heuristics'][key] for k in ks) for key in kinds[ks[0]]['heuristics']}}
    return r

chances=basic(base['classes'])
# Select ONE source-only heuristic using TRAIN template means ONLY.
policy=max(chances['train']['heuristic_macro'],key=lambda x:chances['train']['heuristic_macro'][x])
chances['heuristic_policy_selected_only_on_training']={'name':policy, 'train':chances['train']['heuristic_macro'][policy],
   'ood':chances['ood']['heuristic_macro'][policy],'hard':chances['hard']['heuristic_macro'][policy]}

rows=[]
for model in models:
    seeds=[]
    for seed in range(5):
        raw=json.loads((root/f'scope_{model}_s{seed}.json').read_text());f=raw['curve'][-1]
        scores={k:v['acc'] for k,v in f['by_kind'].items()}
        pairs={name: {'both':v['both_correct'],'flip':v['pred_flip'],'invariant':1-v['pred_flip']} for name,v in f.get('paired',{}).items()}
        seeds.append({'seed':seed,'by_kind':scores,'pairs':pairs,'train_macro':statistics.mean(scores[k] for k in TRAIN_KINDS),
                      'ood_macro':statistics.mean(scores[k] for k in OOD_KINDS)})
    # Paired both/flip/invariant are truly observed ONLY for s1..s4, not five seeds.
    have=[v for v in seeds if v['pairs']]
    macro=statistics.mean(z['ood_macro'] for z in seeds)
    sd=statistics.stdev(z['ood_macro'] for z in seeds)
    boot=[]
    for _ in range(reps):
        ks=rng.choices(OOD_KINDS,k=len(OOD_KINDS))
        seed_sel=[rng.choice(seeds) for _ in range(5)]
        boot.append(statistics.mean(statistics.mean(s['by_kind'][k] for k in ks) for s in seed_sel))
    boot.sort()
    prs={}
    for pairname in sorted(have[0]['pairs']):
        vals=[x['pairs'][pairname] for x in have]
        prs[pairname]={'n_seed':len(vals),'mean_both':statistics.mean(v['both'] for v in vals),
                       'mean_flip':statistics.mean(v['flip'] for v in vals),
                       'mean_invariant':statistics.mean(v['invariant'] for v in vals),
                       'chance_independent_unique_candidates_both':base['paired'].get(pairname,{}).get('chance_independent_uniform_candidates_both'),
                       'warning':'the paired changes may not be single-token counterfactual for each legacy template'}
    rows.append({'model':model,'train_macro':statistics.mean(s['train_macro'] for s in seeds),
        'ood_macro':macro,'ood_sd_across_seeds':sd,
        'ood_joint_seed_template_bootstrap_percentile_95':[boot[int(.025*reps)],boot[int(.975*reps)]],
        'n_seeds_accuracy':5,'n_seeds_pairs':len(have),
        'paired_macro_both_correct':statistics.mean(prs[z]['mean_both'] for z in prs),
        'paired_macro_prediction_invariant':statistics.mean(prs[z]['mean_invariant'] for z in prs),
        'paired_by_type':prs,'per_seed':seeds})

hard={}
for name in ['transformer','pointer','depth-pointer','pointer-depth-encoding-only','transformer-depth']:
    for steps in [1000,3000]:
        f=root/f'hard_{name}_s0_n{steps}.json'
        if not f.exists():continue
        z=json.loads(f.read_text())['hard_ood']
        hard[f'{name}-{steps}']={'n_seeds':1,'hard_macro':z['macro'],
              'by_kind':{k:v['acc'] for k,v in z['by_kind'].items()},
              'pairs':{k:{'both':v['both_correct'],'flip':v['pred_flip'],'invariant':1-v['pred_flip']} for k,v in z['paired'].items()},
              'pair_macro_both':statistics.mean(x['both_correct'] for x in z['paired'].values()),
              'pair_macro_invariant':statistics.mean(1-x['pred_flip'] for x in z['paired'].values())}
res={'protocol':'Recomputed old exploratory 20 templates +10 hard; 5 seed accuracy, original pair metrics only 4 seeds (s1..4), hard paired only seed0. Bootstrap resamples whole OOD template families + training seed, not individual tokens. Source-only heuristic picks on TRAIN kinds, no test-kind selection.','chance':chances,'models':rows,'hard_one_seed':hard,
     'warnings':['Class-uniform 6.25% is NOT the appropriate conditional candidate-set chance on source-literal pointer tasks.','Per-class candidate counts are conditional on literal source candidates, usually 2 or 3, giving candidate-aware chance 50% or 33.3%.','Old OOD classes were repeatedly consulted during architecture design; not a frozen final test.','Original pair metrics missing for seed0, so any claim of 5-seed pairing from these logs is false.','Template-level bootstrap reflects finite 8 template family heterogeneity; source generation fixed heldout RNG; this does not establish broad real-code generalization.']}
Path(a.out).write_text(json.dumps(res,ensure_ascii=False,indent=2))
print('SELECTED_HEURISTIC',policy,chances['heuristic_policy_selected_only_on_training'])
for s in rows:print(s['model'],'OOD',round(s['ood_macro'],4),'CI',s['ood_joint_seed_template_bootstrap_percentile_95'],'paired invariance s1-4',round(s['paired_macro_prediction_invariant'],4))
print('CHANCES',chances)

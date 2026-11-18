"""One-shot immutable A+C test generation and manifest; never choose architecture based on this split.
Test records have CPython-validated target and randomized literal candidates.
Train/dev examples only from A+B and C+D, with disjoint seeds and grammar tags.
"""
from grammar_compositions import make_dataset,describe
from dataclasses import asdict
from pathlib import Path
import hashlib,json,argparse
p=argparse.ArgumentParser();p.add_argument('--outdir',default='frozen_protocol');p.add_argument('--test-n-pairs',type=int,default=256);p.add_argument('--test-seed',type=int,default=2026111704);a=p.parse_args()
out=Path(a.outdir);out.mkdir(exist_ok=True,parents=True)
SEEDS={'train_AB':11071,'train_CD':21731,'dev_AB':31071,'dev_CD':41731,'test_AC':a.test_seed}
# Never create or expose train+dev sample of combo AC; future training must import
# only generate(AB/CD) on separate fresh per-step RNGs.
test=make_dataset('AC',SEEDS['test_AC'],a.test_n_pairs,True)
rows=[]
for idx,(x,y) in enumerate(test):
    rows.append({'pair_index':idx,'pair_id':x.pair_id,'composition':'AC',
                 'left':{'source':x.source,'label':x.truth,'production':x.production,'literal_candidates':x.source_literals},
                 'right':{'source':y.source,'label':y.truth,'production':y.production,'literal_candidates':y.source_literals}})
assert all(row['left']['label']!=row['right']['label'] for row in rows)
assert all(len(row['left']['literal_candidates'])==len(row['right']['literal_candidates'])==4 for row in rows)
path=out/'FROZEN_AC_TEST_256pairs.jsonl'
content=''.join(json.dumps(z,sort_keys=True,separators=(',',':'))+'\n' for z in rows)
if path.exists():
    assert path.read_text()==content,'TEST FILE HAS CHANGED: MUST NOT OVERWRITE'
else:path.write_text(content)
sha=hashlib.sha256(content.encode()).hexdigest()
manifest={'protocol_version':'pbs-composition-20261117-v2','train_compositions':['AB','CD'],
          'development_compositions':['AB','CD'],'sealed_evaluation_composition':'AC',
          'seeds':SEEDS,'frozen_test_pairs':len(rows),'frozen_test_sha256':sha,
          'test_content_path':path.name,'test_generation_script_sha256':hashlib.sha256(Path('grammar_compositions.py').read_bytes()).hexdigest(),
          'per_example_literal_unique_candidates':4,'conditional_uniform_literal_chance':.25,
          'paired_independent_uniform_candidate_both':.0625,
          'paired_distinct_label_uniform_without_replacement_both':1/12,
          'strict_rule':'Use AB/CD dev only for architecture and hyperparameters. Commit config and freeze manifest BEFORE first AC inference; evaluate AC only once for each prespecified baseline. Do not change model after observing AC results. New followups require a new held-out seed, but should be marked as post hoc.',
          'do_not_conflate':'CPython labels on randomized limited syntax grammar, not actual open-source code; test on AC composition is OOD grammar with overlapping individual operators.'}
m=out/'PROTOCOL_MANIFEST.json';m.write_text(json.dumps(manifest,indent=2))
print('FROZEN_AC_SHA256='+sha)
print('FROZEN_SUMMARY='+json.dumps(describe(test)))
print('MANIFEST='+str(m))

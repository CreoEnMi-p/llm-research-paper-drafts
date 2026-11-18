"""Full independent restricted-AST symbolic ceiling on the sealed V2 A+C split.

This is a *post-hoc algorithmic ceiling* applied to a CONSUMED test; not a
fresh model evaluation, and not a trainable model. No gold labels are used
during prediction. Unlike CPython, Script.predict interprets only benchmark's
restricted AST fragment and returns the final print value.
"""
import hashlib,json,sys,random,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/"repro"))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from grammar_compositions import make_dataset
from ast_reference import predict
root=Path(__file__).resolve().parent
m=json.loads((root/'repro/frozen_protocol_v2/PROTOCOL_MANIFEST.json').read_text())
p=root/'repro/frozen_protocol_v2/FROZEN_AC_TEST_256pairs.jsonl'
source=p.read_bytes()
assert hashlib.sha256(source).hexdigest()==m['frozen_test_sha256']
rows=[json.loads(z) for z in source.splitlines()]
assert len(rows)==256
correct=0
by=collections.Counter()
for r in rows:
    assert r['left']['label']!=r['right']['label']
    a=predict(r['left']['source'])
    b=predict(r['right']['source'])
    assert a==r['left']['label'] and b==r['right']['label'],(r['pair_id'],a,b)
    by[r['left']['production']]+=1
    correct+=1
fresh={}
for combo in ('AB','CD','AC'):
    entries=make_dataset(combo,20261223,600,False)
    ok=sum(predict(x.source)==x.truth for x in entries)
    assert ok==len(entries),(combo,ok)
    fresh[combo]={'n':len(entries),'accuracy':ok/len(entries)}
result={'test_sha256':m['frozen_test_sha256'],'frozen_n_pairs':len(rows),
  'frozen_single_accuracy':1,'frozen_both_accuracy':correct/len(rows),
  'production_pairs':dict(by),'fresh_grammar':fresh,
  'disclosure':'Post hoc restricted-AST symbolic ceiling (no learned model, no gold labels at inference); V2 has already been used to evaluate and tune research hypotheses.'}
out=root/'repro/ast_ceiling_v2_verified.json'
out.write_text(json.dumps(result,indent=2))
print('AST_GRAMMAR_CEILING='+json.dumps(result),flush=True)

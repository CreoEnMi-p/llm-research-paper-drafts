"""Two exact-match algorithms differing ONLY in whether the pointer carries
the resolved occurrence POSITION. Uses pinned author's original Sanford k7
get_strings() and checks official hop labels.

The fixed-cutoff method is a deliberately simplified value-only surrogate,
NOT an exact implementation of neural P/Q streams. It tests whether nearest
matching tokens can be recursively selected correctly without updating the
causal cutoff. The position-aware method stores the actual occurrence index.
"""
import json,os,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str((Path(os.environ["SANFORD_REPO"])/"src").resolve()))
from tasks import InductionHopsFullSequenceTask

task=InductionHopsFullSequenceTask(seq_len=100,char_tokens=4,min_hops=7,max_hops=7,
                                  rng=np.random.RandomState(20261117))
n=4096
result={name:{k:0 for k in [1,2,3,4,5,6,7]} for name in ("position_aware","fixed_origin_cutoff")}
index_correct={name:{k:0 for k in [1,2,3,4,5,6,7]} for name in ("position_aware","fixed_origin_cutoff")}
ignored=0
for _ in range(n):
    inp, out, hopstrings, hopindices=task.get_strings(metadata=True,hops=7)
    chars=inp[1:]
    assert len(chars)==98
    if out[-1]=="~":
        ignored+=1
        continue
    last=len(chars)-1
    true_chars={k:hopstrings[k][last] for k in range(1,8)}
    true_indexes={k:hopindices[k][last] for k in range(1,8)}
    for name in result:
        position=last
        token=chars[position]
        for k in range(1,8):
            cutoff=position if name=="position_aware" else last
            j=chars[:cutoff].rfind(token)
            if j<0: raise AssertionError("expected valid 7-hop case")
            position=j+1
            token=chars[position]
            if token==true_chars[k]:
                result[name][k]+=1
            if position==true_indexes[k]:
                index_correct[name][k]+=1
total=n-ignored
assert total>0
assert all(result["position_aware"][k]==total for k in range(1,8))
assert all(index_correct["position_aware"][k]==total for k in range(1,8))
data={"n":total,"source_generator_commit":os.environ.get("SANFORD_SHA"),
      "original_task":"Sanford InductionHopsFullSequenceTask, char4 seq100, k7 last token",
      "value_only_fixed_cutoff_surrogate":{
          "char_accuracy_by_hop":{str(k):result["fixed_origin_cutoff"][k]/total for k in range(1,8)},
          "position_accuracy_by_hop":{str(k):index_correct["fixed_origin_cutoff"][k]/total for k in range(1,8)}
      },
      "position_aware_exact_oracle":{
          "char_accuracy_by_hop":{str(k):result["position_aware"][k]/total for k in range(1,8)},
          "position_accuracy_by_hop":{str(k):index_correct["position_aware"][k]/total for k in range(1,8)}
      },
      "warning":"Fixed-cutoff surrogate deliberately approximates a token-only state, NOT neural BLN. Position-aware is hard-coded exact newest-match and no trainable model."}
Path("sanford_positional_oracle.json").write_text(json.dumps(data,indent=2))
print("SANFORD_POSITION_ORACLE="+json.dumps(data),flush=True)

"""Profile exactly how Sanford's pinned official generator demands newest-match lookup.

No model or training. Uses get_strings() from source commit
5735bca9545545774098994e21392b51b285bcc5. Independently recomputes
all 7 hops for the LAST output token and asserts equality with author label.
We condition on non-null seven-hop cases, matching terminal7 neural protocol.
"""
import os,sys,json
from pathlib import Path
import numpy as np
src=Path(os.environ["SANFORD_REPO"])/"src"
sys.path.insert(0,str(src.resolve()))
from tasks import InductionHopsFullSequenceTask
task=InductionHopsFullSequenceTask(seq_len=100,char_tokens=4,min_hops=7,max_hops=7,
                                   rng=np.random.RandomState(981273))
valid=0;tries=0;ambiguous_first=0;count_by_stage=[[] for _ in range(7)]
oracle_product=[]
wrong_naive=[]
while valid<4096:
    inp,out,*_=task.get_strings(metadata=True,hops=7)
    tries+=1
    if out[-1]=="~": continue
    chars=inp[1:]                # official input excludes final 2 random source chars
    assert len(chars)==98 and len(out)==99
    pos=len(chars)-1
    counts=[]
    for hop in range(7):
        current=chars[pos]
        earlier=[j for j in range(pos) if chars[j]==current]
        assert earlier,(inp,out,hop,pos)
        counts.append(len(earlier))
        pos=earlier[-1]+1
    assert chars[pos]==out[-1],(inp,out,chars[pos])
    valid+=1
    for h,c in enumerate(counts):count_by_stage[h].append(c)
    ambiguous_first+=(counts[0]>1)
    oracle_product.append(float(np.prod([1/c for c in counts])))
    wrong_naive.append(counts)
result={"generator_commit":os.environ.get("SANFORD_SHA"),
    "task":"official InductionHopsFullSequenceTask; k=7, last token, conditioned on valid label",
    "valid_samples":valid,"generated_total":tries,
    "valid_fraction":valid/tries,
    "last_token_char_vocab":4,
    "first_hop_multiple_same_key_fraction":ambiguous_first/valid,
    "candidate_count_by_hop":[dict(hop=i+1,mean=float(np.mean(v)),
        median=float(np.median(v)),q10=float(np.quantile(v,0.1)),
        q90=float(np.quantile(v,0.9))) for i,v in enumerate(count_by_stage)],
    "uniform_selection_first_hop_expected_success":float(np.mean(1/np.array(count_by_stage[0]))),
    "uniform_independent_match_seven_hops_expected_path_success":float(np.mean(oracle_product)),
    "newest_exact_match_hard_oracle_success":1.0,
    "warning":"uniform complete path probability is not end-answer accuracy because wrong paths can collide to same character"}
Path("sanford_collision_profile.json").write_text(json.dumps(result,indent=2))
print("SANFORD_COLLISIONS="+json.dumps(result),flush=True)

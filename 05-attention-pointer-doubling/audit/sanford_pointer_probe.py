"""Direct mechanistic test: can an initialized pointer head select the *latest*
previous identical char in Stanford's pinned k7 sequence?

Unlike final 4-way char accuracy, this measures exact ATTENTION INDEX retrieval.
Head0 A0 at final query position should attend to value token immediately after
the last preceding occurrence of the same current char. This exact target
index is recomputed directly from the INPUT; label output is not used to pick.
"""
import argparse,importlib.util,os,sys,json
from pathlib import Path
import numpy as np,torch

p=argparse.ArgumentParser()
p.add_argument("--mode",choices=["none","rank","exact"],required=True)
p.add_argument("--seed",type=int,default=0)
p.add_argument("--n",type=int,default=256)
p.add_argument("--out",default="pointer_probe.json")
a=p.parse_args()
sys.path.insert(0,str(Path(os.environ["SANFORD_REPO"],"src").resolve()))
from tasks import InductionHopsFullSequenceTask
t=InductionHopsFullSequenceTask(seq_len=100,char_tokens=4,min_hops=7,max_hops=7,
                                 rng=np.random.RandomState(112233))
xx,yy=t.get_batch(a.n,hops=7)
noexist=t.token_map["~"]
assert (yy[:,-1]!=noexist).all()
input_ids=xx.long()
T=input_ids.shape[1]
assert T==99
# j indexes prior occurrence of the same char; a matching value is at j+1.
key_idx=torch.arange(1,T-1)
match=(input_ids[:,1:-1]==input_ids[:,-1,None])
assert match.any(-1).all()
latest_key=torch.where(match,key_idx[None,:],-1).amax(-1)
correct_value_pos=latest_key+1
first_counts=match.sum(-1)

sys.argv=["pdg_semantic_generated.py","--dev","cpu","--threads","2","--task","mqar",
          "--L","100","--V","256","--pairs","8","--width","64","--layers","4",
          "--heads","4","--seed",str(a.seed),"--sink","0","--alibi","0",
          "--eqh","2","--eqt","3","--ptr","11","--pnb","3",
          "--tie_recent",str(3.0 if a.mode!="none" else 0.0),
          "--tie_tol","0.04"]
modelpath=Path(__file__).resolve().parents[1]/"pdg_semantic_generated.py"
spec=importlib.util.spec_from_file_location("probe_model",modelpath)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
torch.set_num_threads(2);torch.manual_seed(a.seed)
m=mod.Model(t.num_tokens,64,4,4)
m.eval()
overall_hits=0;exact_mass=0.;selected_any_match=0;count=0
mass_by_candidates={}
with torch.no_grad():
    for i in range(0,a.n,16):
        ids=input_ids[i:i+16]
        wants=correct_value_pos[i:i+16]
        m.rec=True;m.ptrA=[]
        m(ids)
        assert len(m.ptrA)==4
        A=m.ptrA[0][:,-1,:]
        pred=A.argmax(-1)
        hits=(pred==wants)
        overall_hits+=int(hits.sum())
        exact_mass+=float(A.gather(1,wants[:,None]).sum())
        correct_idx=(pred-1).clamp(0,T-1)
        pred_is_candidate=ids.gather(1,correct_idx[:,None]).squeeze(1)==ids[:,-1]
        selected_any_match+=int(pred_is_candidate.sum())
        count+=len(ids)
        m.rec=False;m.ptrA=[]
result={"mode":a.mode,"seed":a.seed,"n":count,
        "task":"official Sanford k7, UNTRAINED first-pointer A0 attention index, query last",
        "correct_latest_predecessor_pos_top1":overall_hits/count,
        "correct_latest_predecessor_attention_mass":exact_mass/count,
        "top1_selects_any_matching_previous_char":selected_any_match/count,
        "mean_repeated_candidates":float(first_counts.float().mean()),
        "uniform_over_exact_candidates_top1_reference":float((1/first_counts.float()).mean()),
        "note":"This is first-pointer INDEX selection, not the 4-character final label accuracy. No gradient training."}
Path(a.out).write_text(json.dumps(result,indent=2))
print("SANFORD_POINTER_FIRST="+json.dumps(result),flush=True)

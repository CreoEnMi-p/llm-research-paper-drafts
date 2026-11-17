"""Read out the BLN Q POINTER STATE after layer 1, 2, 3, 4 on original Sanford k7.

Question: Even if A0 selects the exact nearest prior occurrence on hop 1,
does Q keep the information needed to perform subsequent context-dependent
hops? Set the task's k7 bit gate schedule (+8,+8,+8,-8) by hand. These
are RANDOM-INITIALIZED HEADS; final LM readout is not evaluated here.
Compare e(Q_after_l) to embedding of author's true hop1, hop3, hop7, hop7
output characters, not true *positions* (which cannot be decoded from token
embeddings alone). True source labels recomputed from official metadata.
"""
import argparse,importlib.util,os,sys,json
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F

ap=argparse.ArgumentParser()
ap.add_argument("--mode",choices=["none","rank","exact"],required=True)
ap.add_argument("--seed",type=int,default=0)
ap.add_argument("--pnb",type=int,choices=[0,3],default=3)
ap.add_argument("--n",type=int,default=128)
ap.add_argument("--out",default="qstate_probe.json")
a=ap.parse_args()
sys.path.insert(0,str(Path(os.environ["SANFORD_REPO"],"src").resolve()))
from tasks import InductionHopsFullSequenceTask

official=InductionHopsFullSequenceTask(seq_len=100,char_tokens=4,min_hops=7,max_hops=7,
                                      rng=np.random.RandomState(778123))
xx,yy,_,_,hop_strings,_=official.get_batch(a.n,metadata=True,hops=7)
assert (yy[:,-1]!=official.token_map["~"]).all()
# Direct source label for every intermediate hop; source location is the
# last position of the 98 source chars fed to model, i.e. index97.
stage_hops=[1,3,7,7]
labels=[[official.token_map[per_row[h][97]] for per_row in hop_strings]
        for h in stage_hops]
targets=torch.tensor(labels,dtype=torch.long)

# Patch only a generated model file to capture intermediate Q. Original
# model and other experiments untouched.
src=Path(__file__).resolve().parents[1]/"pdg_semantic_generated.py"
code=src.read_text()
old='        e0 = x; ptr = x; qq = x; rr = None; rq = torch.zeros_like(x); pprev = None; s.xb = None'
assert code.count(old)==1
code=code.replace(old,old+'\n        s.q_trace=[]',1)
old='                    if getattr(s, "rec", False): s.ptrA.append(A0.detach())'
assert code.count(old)==1
code=code.replace(old,'                    if getattr(s,"track_q",False) and a.ptr==11: s.q_trace.append(qq.detach())\n'+old,1)
dst=Path(__file__).resolve().parents[1]/"pdg_qtrace_generated.py"
dst.write_text(code)

sys.argv=["pdg_qtrace_generated.py","--dev","cpu","--threads","2","--task","mqar",
          "--L","100","--V","256","--pairs","8","--width","64","--layers","4",
          "--heads","4","--seed",str(a.seed),"--sink","0","--alibi","0",
          "--eqh","2","--eqt","3","--ptr","11","--pnb",str(a.pnb),
          "--tie_recent",str(3.0 if a.mode!="none" else 0.0),
          "--tie_tol","0.04"]
spec=importlib.util.spec_from_file_location("qtrace_model",dst)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
torch.set_num_threads(2);torch.manual_seed(a.seed)
m=mod.Model(official.num_tokens,64,4,4)
m.eval()
with torch.no_grad():
    m.S["pg"].copy_(torch.tensor([8.,8.,8.,-8.]))
n_correct=[0]*4
sims=[0.]*4
n=0
with torch.no_grad():
    for start in range(0,a.n,16):
        x=xx[start:start+16].long()
        m.track_q=True;m(x)
        assert len(m.q_trace)==4,len(m.q_trace)
        emb=F.normalize(m.emb.weight.detach(),dim=-1)
        for layer,state in enumerate(m.q_trace):
            q=F.normalize(state[:,-1,:],dim=-1)
            score=q @ emb.T
            pred=score.argmax(-1)
            truth=targets[layer,start:start+16]
            n_correct[layer]+=int((pred==truth).sum())
            sims[layer]+=float(score.gather(1,truth[:,None]).sum())
        n+=len(x)
        m.track_q=False
        m.q_trace=[]
result={"mode":a.mode,"pnb":a.pnb,"seed":a.seed,"n":n,
        "source":"Sanford original get_batch metadata k7, stage1/3/7/7 last label",
        "oracle_bit_gates":[8,8,8,-8],
        "model_trained":False,
        "Q_nearest_token_accuracy_by_layer":[z/n for z in n_correct],
        "Q_cosine_to_correct_char_embedding_by_layer":[z/n for z in sims],
        "warning":"Only character identity decoded; does NOT measure true predecessor position trajectory or trained model readout."}
Path(a.out).write_text(json.dumps(result,indent=2))
print("SANFORD_QSTATE="+json.dumps(result),flush=True)

"""Independent data-transfer benchmark using Sanford et al.'s ORIGINAL generator.

Inputs / targets are exactly those from the official InductionHopsFullSequenceTask
at github.com/chsanford/hop-induction-heads, pinned to a commit in the CI workflow.
Unlike the original paper's GPT2, the compared models use our own shared
small-Transformer implementation and optimizer. This is a transfer benchmark,
NOT a full 100,001-step reproduction of the published model.

Crucial: report conditional accuracy on valid non-null hops, because a majority
of long-hop targets may be '~', and full accuracy is misleading.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

p=argparse.ArgumentParser()
p.add_argument("--arm",choices=("rope","nope","bln","looped-nope"),required=True)
p.add_argument("--seed",type=int,default=0)
p.add_argument("--steps",type=int,default=1500)
p.add_argument("--batch",type=int,default=16)
p.add_argument("--max-hops",type=int,default=7)
p.add_argument("--layers",type=int,default=4)
p.add_argument("--width",type=int,default=64)
p.add_argument("--evaln",type=int,default=128)
p.add_argument("--eval-every",type=int,default=100)
p.add_argument("--lr",type=float,default=0.02)
p.add_argument("--out",default="sanford-audit.json")
a=p.parse_args()

# Preserve exactly the Sanford task code, imported from a pinned git checkout.
official=os.environ.get("SANFORD_REPO","")
assert official and Path(official,"src/tasks.py").exists(), "SANFORD_REPO checkout missing"
sys.path.insert(0,str(Path(official,"src").resolve()))
from tasks import InductionHopsFullSequenceTask
import numpy as np
import torch
import torch.nn.functional as F

sys.argv=["pdg_fair_generated.py",
          "--dev","cpu","--threads","2","--task","mqar",
          "--L","100","--V","2000","--pairs","8",
          "--width",str(a.width),"--layers",str(a.layers),
          "--seed",str(a.seed),"--init","0.5",
          "--sink","0","--alibi","0",
          "--lr",str(a.lr),"--elr","0.04","--nope",
          str(int(a.arm in ("nope","looped-nope"))),
          "--loop",str(int(a.arm=="looped-nope"))]
if a.arm == "bln":
    sys.argv+=["--eqh","2","--eqt","3","--ptr","11","--pnb","3"]
modpath=Path(__file__).resolve().parents[1]/"pdg_fair_generated.py"
assert modpath.exists(), "run audit/build_fair.py first"
spec=importlib.util.spec_from_file_location("pdg_fair_generated",modpath)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
assert m.DEV.type=="cpu"
torch.manual_seed(a.seed)
torch.set_num_threads(2)

task=InductionHopsFullSequenceTask(seq_len=100,char_tokens=4,min_hops=0,
                                   max_hops=a.max_hops,rng=np.random.RandomState(6000+a.seed))
valtask=InductionHopsFullSequenceTask(seq_len=100,char_tokens=4,min_hops=0,
                                      max_hops=a.max_hops,rng=np.random.RandomState(987123))
V=task.num_tokens
assert V == valtask.num_tokens
noexist=task.token_map[task.DOES_NOT_EXIST_CHAR]
blank=task.token_map[task.BLANK_CHAR]
val={}
for k in range(a.max_hops+1):
    x,y=valtask.get_batch(a.evaln,hops=k)
    val[k]=(x,y)

model=m.Model(V,a.width,a.layers,4).to(m.DEV)
scalars=[w for name,w in model.named_parameters() if name.startswith("S.")]
other=[w for name,w in model.named_parameters() if not name.startswith(("P.","S.","T."))]
optim_adam=torch.optim.Adam(
    [dict(params=other,lr=0.04)]+([dict(params=scalars,lr=0.04)] if scalars else []),
    betas=(0.9,0.95))
optim_muon=m.Muon(model,a.lr)
curve=[]
start=time.time()
def evaluate():
    model.eval()
    result={}
    with torch.no_grad():
        for k,(ins,lab) in val.items():
            correct=0;count=0;valid_correct=0;valid_total=0;empty_correct=0;empty_total=0;loss_sum=0
            for s in range(0,len(ins),16):
                xx=ins[s:s+16].to(m.DEV)
                yy=lab[s:s+16].to(m.DEV)
                out=model(xx)
                pr=out.argmax(-1)
                good=(pr==yy)
                valid=(yy!=noexist)&(yy!=blank)
                empty=(yy==noexist)
                correct+=int(good.sum())
                count+=yy.numel()
                valid_correct+=int((good&valid).sum())
                valid_total+=int(valid.sum())
                empty_correct+=int((good&empty).sum())
                empty_total+=int(empty.sum())
                loss_sum+=float(F.cross_entropy(out.float().reshape(-1,V),yy.reshape(-1),reduction="sum"))
            result[str(k)]={"accuracy":correct/count,
                    "nonnull_accuracy":valid_correct/max(1,valid_total),
                    "nonnull_fraction":valid_total/count,
                    "null_accuracy":empty_correct/max(1,empty_total),
                    "cross_entropy":loss_sum/count}
    model.train()
    return result

for i in range(a.steps):
    progress=min((i+1)/20,1.0,(a.steps-i)/max(1,0.3*a.steps))
    optim_muon.sc=progress
    for group in optim_adam.param_groups:
        group["lr"]=0.04*progress
    xx,yy=task.get_batch(a.batch)
    xx=xx.to(m.DEV);yy=yy.to(m.DEV)
    out=model(xx)
    loss=F.cross_entropy(out.float().reshape(-1,V),yy.reshape(-1))
    loss.backward()
    optim_muon.step()
    optim_adam.step()
    model.zero_grad(set_to_none=True)
    if (i+1)%a.eval_every==0 or (i+1)==a.steps:
        v=evaluate()
        curve.append({"step":i+1,"train_loss":float(loss),"per_hop":v,
                      "full_accuracy":float(np.mean([w["accuracy"] for w in v.values()])),
                      "nonnull_accuracy":float(np.mean([w["nonnull_accuracy"] for w in v.values()]))})
        print("SANFORD_PROGRESS",a.arm,a.seed,i+1,
              round(curve[-1]["full_accuracy"],4),round(curve[-1]["nonnull_accuracy"],4),flush=True)

res={"task":"OFFICIAL-InductionHopsFullSequenceTask","source_commit":os.environ.get("SANFORD_SHA"),
     "architecture":"original pdg_fair_generated Model, NOT Sanford GPT2",
     "arm":a.arm,"seed":a.seed,"max_hops":a.max_hops,"steps":a.steps,
     "width":a.width,"layers":a.layers,"train_seconds":time.time()-start,
     "parameters":sum(z.numel() for z in model.parameters()),
     "per_hop":curve[-1]["per_hop"],"curve":curve}
Path(a.out).write_text(json.dumps(res,indent=2))
print("SANFORD_FINAL="+json.dumps({k:res[k] for k in ("arm","seed","steps","parameters","train_seconds","per_hop")}),flush=True)

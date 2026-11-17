"""Terminal-only, non-null Sanford k=7 benchmark with train/eval isolation.

Uses EXACT pinned source generator's get_batch() (no custom semantics).
Only the LAST target token is supervised; train/eval exclude "~" samples.
Former full-sequence results are not comparable. Relative claims require
strong matched AdamW and Muon baselines, >=5 seeds and long training.
"""
import argparse, importlib.util, json, os, sys, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--arm",choices=["rope","nope","bln","onehop","loop-nope"],required=True)
p.add_argument("--seed",type=int,default=0)
p.add_argument("--steps",type=int,default=2000)
p.add_argument("--batch",type=int,default=32)
p.add_argument("--layers",type=int,default=4)
p.add_argument("--width",type=int,default=64)
p.add_argument("--evaln",type=int,default=256)
p.add_argument("--eval_every",type=int,default=100)
p.add_argument("--optimizer",choices=["adamw","muon_adam"],default="adamw")
p.add_argument("--lr",type=float,default=0.001)
p.add_argument("--weight_decay",type=float,default=0.01)
p.add_argument("--out",type=str,default="sanford-final.json")
a=p.parse_args()
import numpy as np, torch
import torch.nn.functional as F
official=Path(os.environ["SANFORD_REPO"])
assert (official/"src"/"tasks.py").exists()
sys.path.insert(0,str((official/"src").resolve()))
from tasks import InductionHopsFullSequenceTask

# Imported model code only, not its generated samples.
sys.argv=["pdg_reviewer_generated.py","--dev","cpu","--threads","2",
          "--task","mqar","--L","100","--pairs","8","--V","256",
          "--width",str(a.width),"--layers",str(a.layers),
          "--seed",str(a.seed),"--sink","0","--alibi","0",
          "--nope",str(int(a.arm in ("nope","loop-nope"))),
          "--loop",str(int(a.arm=="loop-nope"))]
if a.arm=="bln":sys.argv+=["--eqh","2","--eqt","3","--ptr","11","--pnb","3"]
if a.arm=="onehop":sys.argv+=["--eqh","1","--eqt","3","--ptr","1","--pnb","3"]
path=Path(__file__).resolve().parents[1]/"pdg_reviewer_generated.py"
spec=importlib.util.spec_from_file_location("pdg_final_import",path)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
torch.manual_seed(a.seed);torch.set_num_threads(2)
task_args=dict(seq_len=100,char_tokens=4,min_hops=7,max_hops=7)
train_task=InductionHopsFullSequenceTask(**task_args,rng=np.random.RandomState(813+a.seed))
val_task=InductionHopsFullSequenceTask(**task_args,rng=np.random.RandomState(918276))
# Official generator bug? min_hops=7 shifts count token to integer 0; every
# arm shares it; targets remain exactly k=7 with source generator.
V=train_task.num_tokens; noexist=train_task.token_map["~"]
def valid_batch(task,B):
    xs=[];ys=[];generated=0
    while sum(len(z) for z in xs)<B:
        x,y=task.get_batch(max(32,B),hops=7)
        valid=(y[:,-1]!=noexist)
        generated+=len(y)
        if valid.any():
            xs.append(x[valid]);ys.append(y[valid,-1])
        assert generated<=B*100, "Insufficient terminal non-null examples"
    return torch.cat(xs)[:B],torch.cat(ys)[:B],generated
vx,vy,vg=valid_batch(val_task,a.evaln)
assert len(vy)==a.evaln and not bool((vy==noexist).any())
model=mod.Model(V,a.width,a.layers,4).to(mod.DEV)
if a.optimizer=="adamw":
    adam=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay,betas=(0.9,0.95))
    muon=None
else:
    P=[v for k,v in model.named_parameters() if k.startswith("P.")]
    rest=[v for k,v in model.named_parameters() if not k.startswith("P.")]
    assert P and rest
    adam=torch.optim.Adam(rest,lr=a.lr,betas=(0.9,0.95))
    muon=mod.Muon(model,a.lr)
steps=a.steps;curve=[];start=time.time()
def evaluate():
    model.eval()
    with torch.no_grad():
        total_correct=0;loss_sum=0.0
        for off in range(0,a.evaln,32):
            x=vx[off:off+32].to(mod.DEV)
            y=vy[off:off+32].to(mod.DEV)
            logit=model(x)[:,-1]
            total_correct+=int((logit.argmax(-1)==y).sum())
            loss_sum+=float(F.cross_entropy(logit.float(),y,reduction="sum"))
    model.train()
    return total_correct/a.evaln,loss_sum/a.evaln
for step in range(steps):
    sc=min((step+1)/max(1,steps//50),1.0,(steps-step)/max(1,0.1*steps))
    if muon is not None:muon.sc=sc
    for group in adam.param_groups:group["lr"]=a.lr*sc
    xx,yy,_=valid_batch(train_task,a.batch)
    xx=xx.to(mod.DEV);yy=yy.to(mod.DEV)
    logits=model(xx)[:,-1]
    loss=F.cross_entropy(logits.float(),yy)
    loss.backward()
    if muon is not None:muon.step()
    adam.step()
    model.zero_grad(set_to_none=True)
    if (step+1)%a.eval_every==0 or step+1==steps:
        acc,valloss=evaluate()
        row=dict(step=step+1,acc=acc,val_loss=valloss,train_loss=float(loss.detach()))
        curve.append(row)
        print("FINAL7_PROGRESS",a.arm,a.seed,a.optimizer,step+1,round(acc,4),flush=True)
res=dict(arm=a.arm,seed=a.seed,steps=a.steps,batch=a.batch,layers=a.layers,
         optimizer=a.optimizer,lr=a.lr,weight_decay=a.weight_decay,
         task="pinned official Sanford generator k=7, FINAL VALID token ONLY",
         task_source=os.environ.get("SANFORD_SHA"),
         only_supervised_position=-1,exclude_null=True,val_n=a.evaln,
         initial_generated_for_val=vg,
         params=sum(x.numel() for x in model.parameters()),
         train_seconds=time.time()-start,final=curve[-1],
         peak=max(curve,key=lambda z:z["acc"]),curve=curve)
Path(a.out).write_text(json.dumps(res,indent=2))
print("FINAL7_RESULT="+json.dumps({k:res[k] for k in ["arm","seed","optimizer","steps","batch","layers","train_seconds","final","peak"]}),flush=True)

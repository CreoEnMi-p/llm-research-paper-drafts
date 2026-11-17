"""Position-preserving, sparse trainable pointer (experimental; NOT ICML-ready).

Critical distinction from original BLN:
  Q carries a distribution of *occurrence indices*, not pooled token vectors.
  One-hop keys read the immediately preceding character.
  A sparse transition table composes by either row-doubling (A^2, A^4) or
  k repeated single-hop applications; compare those two at equal conditions.

Q/K are learned, embeddings tied and orthogonally initialized, initialized
shared Q/K maps are identity. The log-distance recency coefficient is learned
from zero. Supervision is ONLY final k7 token, never intermediate positions.
Top-k indices have NO selection gradient, but selected weights differentiate.
This is a strongly structured algorithmic inductive bias, not raw language
understanding, and not a novel proof of logarithmic-depth pointer jumping.

This script uses Sanford's ORIGINAL pinned InductionHopsFullSequenceTask.
"""
import argparse, importlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

p=argparse.ArgumentParser()
p.add_argument("--mode",choices=["doubling","recurrent","dense"],required=True)
p.add_argument("--seed",type=int,default=0)
p.add_argument("--steps",type=int,default=1000)
p.add_argument("--batch",type=int,default=48)
p.add_argument("--k",type=int,default=4)
p.add_argument("--d",type=int,default=16)
p.add_argument("--lr",type=float,default=.01)
p.add_argument("--init-recency",type=float,default=0.)
p.add_argument("--freeze-recency",action="store_true")
p.add_argument("--init-temp",type=float,default=32.)
p.add_argument("--eval-every",type=int,default=200)
p.add_argument("--val-n",type=int,default=512)
p.add_argument("--out",default="sparse-occurrence.json")
a=p.parse_args()
torch.manual_seed(a.seed);torch.set_num_threads(2)
root=os.environ.get("SANFORD_REPO")
assert root and (Path(root)/"src"/"tasks.py").exists(),"Need pinned official task checkout"
sys.path.insert(0,str((Path(root)/"src").resolve()))
from tasks import InductionHopsFullSequenceTask

def official_batch(task,n,hops=7,metadata=False):
    if metadata:
        x,y,ins,outs,hop_strings,hop_indices=task.get_batch(n,metadata=True,hops=hops)
        source_index=task.seq_len-3
        positions=torch.tensor([path[hops][source_index]+1 for path in hop_indices],dtype=torch.long)
        truth=y[:,-1]
        valid=(truth!=task.token_map["~"])&(positions>0)
        x=x[valid];truth=truth[valid];positions=positions[valid]
        assert len(x)>0
        assert ((x[torch.arange(len(x)),positions])==truth).all()
        return x,truth,positions
    x,y=task.get_batch(n,hops=hops)
    truth=y[:,-1]
    good=truth!=task.token_map["~"]
    return x[good],truth[good]

class SparseOccurrence(nn.Module):
    def __init__(self,mode,k,d,char_tokens,init_recency,init_temp):
        super().__init__()
        self.mode=mode;self.K=k;self.chars=char_tokens
        self.emb=nn.Embedding(char_tokens+1,d)
        nn.init.orthogonal_(self.emb.weight)
        self.Wq=nn.Linear(d,d,bias=False)
        self.Wk=nn.Linear(d,d,bias=False)
        nn.init.eye_(self.Wq.weight);nn.init.eye_(self.Wk.weight)
        self.logtemp=nn.Parameter(torch.tensor(math.log(init_temp)))
        self.recency=nn.Parameter(torch.tensor(float(init_recency)))
        self.sink=nn.Parameter(torch.tensor(0.55))

    def logits(self,x):
        B,T=x.shape
        query=F.normalize(self.Wq(self.emb(x)),dim=-1)
        prev=F.pad(x,(1,0))[:,:-1]
        key=F.normalize(self.Wk(self.emb(prev)),dim=-1)
        temperature=self.logtemp.exp().clamp(max=100.)
        logits=temperature*torch.bmm(query,key.transpose(1,2))
        ix=torch.arange(T,device=x.device,dtype=logits.dtype)
        # Nonlinear distance prior; scalar strength LEARNED from final target.
        distance=(ix[:,None]-ix[None,:]).clamp(min=0)
        logits=logits-self.recency*torch.log1p(distance)[None]
        valid=(ix[None,:]<=ix[:,None])&(ix[None,:]>0)
        logits=logits.masked_fill(~valid[None],-1e4)
        # Learned self-loop threshold: state with no predecessor remains there.
        diag=torch.eye(T,device=x.device,dtype=torch.bool)[None]
        return torch.where(diag,torch.logaddexp(logits,(temperature*self.sink).expand(B,T,T)),logits)

    @staticmethod
    def compose(ind,weight,table_ind,table_weight,K):
        B,R,C=ind.shape
        T=table_ind.shape[1];K2=table_ind.shape[-1]
        flat=(torch.arange(B,device=ind.device)[:,None,None]*T+ind).reshape(-1)
        children=table_ind.reshape(B*T,K2)[flat].reshape(B,R,C,K2)
        vals=table_weight.reshape(B*T,K2)[flat].reshape(B,R,C,K2)
        joint=(weight[:,:,:,None]*vals).reshape(B,R,C*K2)
        paths=children.reshape(B,R,C*K2)
        top,choices=torch.topk(joint,k=min(K,C*K2),dim=-1)
        dest=paths.gather(-1,choices)
        return dest,top/(top.sum(-1,keepdim=True)+1e-9)

    def forward(self,x,hops=7):
        B,T=x.shape
        raw=self.logits(x)
        if self.mode=="dense":
            A=torch.softmax(raw,-1)
            q=torch.zeros(B,1,T,device=x.device);q[:,:,-1]=1
            power=A
            for bit in range(hops.bit_length()):
                if hops & (1<<bit):q=torch.bmm(q,power)
                if bit+1<hops.bit_length():power=torch.bmm(power,power)
            d=q[:,0,:]
            answer=torch.zeros(B,self.chars+1,device=x.device)
            answer.scatter_add_(1,x,d)
            return answer,d
        top,ind=torch.topk(raw,k=min(self.K,T),dim=-1)
        weight=F.softmax(top,dim=-1)
        qind=torch.full((B,1,1),T-1,device=x.device,dtype=torch.long)
        qw=torch.ones((B,1,1),device=x.device)
        if self.mode=="recurrent":
            for _ in range(hops):qind,qw=self.compose(qind,qw,ind,weight,self.K)
        else:
            powers=[(ind,weight)]
            for _ in range(1,hops.bit_length()):
                ii,ww=powers[-1]
                powers.append(self.compose(ii,ww,ii,ww,self.K))
            for bit,(ii,ww) in enumerate(powers):
                if hops & (1<<bit):qind,qw=self.compose(qind,qw,ii,ww,self.K)
        dest=qind[:,0,:];mass=qw[:,0,:]
        symbols=x.gather(1,dest)
        answer=torch.zeros((B,self.chars+1),device=x.device)
        answer.scatter_add_(1,symbols,mass)
        return answer,(dest,mass)

def evaluate(model,length,hops,n,seed):
    task=InductionHopsFullSequenceTask(seq_len=length,char_tokens=model.chars,
        min_hops=hops,max_hops=hops,rng=np.random.RandomState(seed))
    total=0;corr=0;position=0;logloss=0.;meanmass=0.
    model.eval()
    with torch.no_grad():
        for start in range(0,n,64):
            x,y,pidx=official_batch(task,min(64,n-start),hops,metadata=True)
            pred,dist=model(x,hops)
            correct=pred.argmax(-1)==y
            corr+=int(correct.sum());total+=len(y)
            logloss+=float(-torch.log(pred.gather(1,y[:,None])+1e-12).sum())
            if model.mode=="dense":
                position+=int((dist.argmax(-1)==pidx).sum())
                meanmass+=float(dist.gather(1,pidx[:,None]).sum())
            else:
                idx,w=dist
                p=idx.gather(1,w.argmax(-1)[:,None]).squeeze(1)
                position+=int((p==pidx).sum())
                meanmass+=float((w*(idx==pidx[:,None])).sum())
    model.train()
    assert total>0
    return {"n":total,"char_accuracy":corr/total,"position_accuracy":position/total,
        "true_position_mass":meanmass/total,"nll":logloss/total}

def run():
    m=SparseOccurrence(a.mode,a.k,a.d,4,a.init_recency,a.init_temp)
    m.recency.requires_grad_(not a.freeze_recency)
    optim=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=a.lr,weight_decay=0)
    train=InductionHopsFullSequenceTask(seq_len=100,char_tokens=4,min_hops=7,max_hops=7,
                                       rng=np.random.RandomState(71+a.seed))
    eval_specs=[("k7-L100",100,7,a.val_n,12345),
                ("k7-L200",200,7,min(256,a.val_n),53455),
                ("k7-L400",400,7,128,98375),
                ("k15-L200",200,15,min(256,a.val_n),86543)]
    t0=time.perf_counter();curve=[]
    for step in range(a.steps):
        x,y=official_batch(train,a.batch,7)
        probs,_=m(x,7)
        loss=-torch.log(probs.gather(1,y[:,None])+1e-9).mean()
        optim.zero_grad(set_to_none=True);loss.backward();optim.step()
        if (step+1)%a.eval_every==0 or step==0 or step+1==a.steps:
            row={"step":step+1,"train_loss":float(loss.detach()),
                 "recency":float(m.recency.detach()),"temp":float(m.logtemp.detach().exp()),
                 "sink_threshold":float(m.sink.detach())}
            row["eval"]={name:evaluate(m,L,h,n,s) for name,L,h,n,s in eval_specs}
            curve.append(row)
            print("OCCURRENCE_PROGRESS",json.dumps(dict(arm=a.mode,seed=a.seed,**row)),flush=True)
    results={"config":vars(a),"task_source_sha":os.environ.get("SANFORD_SHA"),
             "train_target":"official last char k7 ONLY, no position supervision, no auxiliary hops",
             "train_source":"official InductionHopsFullSequenceTask, random 4-char no-adj-repeat",
             "model_init":"orthogonal tied embeddings, identity Q/K maps, log-distance recency initialized as specified; this is a strong handcrafted prior",
             "topk_gradient_caveat":"hard topK indices nondifferentiable; selected softmax weights differentiate",
             "score":"occurrence position Top1 AND final character; fixed heldout RNG; no best-checkpoint selection",
             "parameter_count":sum(z.numel() for z in m.parameters()),
             "train_seconds":time.perf_counter()-t0,"curve":curve}
    Path(a.out).write_text(json.dumps(results,indent=2))
    print("OCCURRENCE_FINAL="+json.dumps({
        "mode":a.mode,"seed":a.seed,"steps":a.steps,"K":a.k,
        "parameters":results["parameter_count"],"train_seconds":results["train_seconds"],
        "recency":curve[-1]["recency"],"val":curve[-1]["eval"]}),flush=True)
if __name__=="__main__":run()

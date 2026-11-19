"""Development-only prerequisite audit of AB/CD query sensitivity.

Isolate whether A+B failed because 1000 optimizer steps were insufficient.
Extended 5000-step matched-candidate classifier baseline; no frozen A+C sample is accessed. Compare AB-only versus AB/CD mixed training,
with independent or paired query sampling at fixed program count and compute. The frozen A+C set is
NEVER opened or evaluated by this script. All model inputs are
Python tokenize lexical IDs and optionally INDENT-derived depth, never gold
scope links/def-use edges/CPython runtime state.
"""
import argparse,json,hashlib,math,random,time,statistics
from pathlib import Path
import numpy as np,torch
from torch import nn
import torch.nn.functional as F
from grammar_compositions import make_dataset,sample,paired
from scope_program_benchmark import ID,NS,PAD,NUMBERS

P=argparse.ArgumentParser()
P.add_argument('--model',choices=['transformer-candidate-depth'],required=True)
P.add_argument('--seed',type=int,required=True)
P.add_argument('--train-mode',choices=['single','paired'],required=True,help='Only training batching differs; never accesses sealed A+C.')
P.add_argument('--train-combo',choices=['AB_only','mixed'],default='AB_only',help='AB-only capacity/optimizer prerequisite control.')
P.add_argument('--steps',type=int,default=1000)
P.add_argument('--batch',type=int,default=32)
P.add_argument('--lr',type=float,default=.001)
P.add_argument('--d',type=int,default=64)
P.add_argument('--layers',type=int,default=2)
P.add_argument('--val-pairs',type=int,default=64)
P.add_argument('--eval-every',type=int,default=250)
P.add_argument('--out',type=str,default='composition.json')
a=P.parse_args()
if a.steps != 5000 or a.batch != 32 or a.d != 64 or a.layers != 2 or a.lr != .001:
    raise ValueError('Extended-budget AB-only development protocol: fixed steps5000, batch32, d64, layers2, LR.001')
random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(2)
CID={k:ID[str(k)] for k in NUMBERS};MAXLEN=110
IDS_TO_CLASS=torch.full((NS,),-1,dtype=torch.long)
for k,v in CID.items():IDS_TO_CLASS[v]=k-20
class Net(nn.Module):
    def __init__(self):
        super().__init__();d=a.d
        self.word=nn.Embedding(NS,d);self.pos=nn.Embedding(MAXLEN,d)
        if a.model in ('pointer-depth','transformer-candidate-depth'):self.depth=nn.Embedding(5,d)
        enc=nn.TransformerEncoderLayer(d_model=d,nhead=4,dim_feedforward=d*2,dropout=0,batch_first=True,norm_first=True)
        self.enc=nn.TransformerEncoder(enc,num_layers=a.layers,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(d)
        if a.model in ('transformer','transformer-candidate-depth'):self.out=nn.Linear(d,16)
        else:self.q=nn.Linear(d,d);self.k=nn.Linear(d,d);self.recency=nn.Parameter(torch.tensor(0.))
    def forward(self,x,depth):
        B,T=x.shape;t=torch.arange(T)
        h=self.word(x)+self.pos(t)[None]
        if a.model in ('pointer-depth','transformer-candidate-depth'):h=h+self.depth(depth.clamp(max=4))
        h=self.norm(self.enc(h,src_key_padding_mask=x==PAD))
        last=(x!=PAD).sum(-1)-1
        query=h[torch.arange(B),last]
        if a.model in ('transformer','transformer-candidate-depth'):
            logits=self.out(query)
            if a.model=='transformer-candidate-depth':
                # Exact same legal source-literal candidate set the pointer uses.
                # This is a public grammar property, NOT a gold binding or answer.
                literal_class=IDS_TO_CLASS[x]
                valid=(literal_class>=0)
                counts=torch.zeros(B,16,dtype=torch.long,device=x.device)
                counts.scatter_add_(1,literal_class.clamp(min=0),valid.long())
                assert (counts>0).sum(-1).min().item()==4
                logits=logits.masked_fill(counts==0,-1e5)
            return logits
        s=(self.q(query)[:,None]*self.k(h)).sum(-1)/math.sqrt(a.d)
        values=IDS_TO_CLASS[x];s=s+self.recency*t[None]/max(1,T-1)
        s=s.masked_fill(values<0,-1e5)
        mass=F.softmax(s,dim=-1)
        result=torch.zeros(B,16)
        result.scatter_add_(1,values.clamp(min=0),mass)
        return torch.log(result+1e-9)

def batch(ps):
    T=max(len(z.ids) for z in ps);B=len(ps)
    assert T<=MAXLEN,(T,len(ps))
    x=torch.full((B,T),PAD,dtype=torch.long)
    dep=torch.zeros((B,T),dtype=torch.long)
    y=torch.empty(B,dtype=torch.long)
    for i,z in enumerate(ps):
        x[i,:len(z.ids)]=torch.tensor(z.ids)
        dep[i,:len(z.depth)]=torch.tensor(z.depth)
        y[i]=z.truth-20
    return x,dep,y

# Development-only experiment: the frozen A+C file is neither imported nor read.
# Seeds match the historical AB/CD development/training protocol.
manifest={'seeds':{'dev_AB':31071,'dev_CD':41731,'train_AB':11071,'train_CD':21731}}
dev={combo:make_dataset(combo,manifest['seeds']['dev_'+combo],a.val_pairs,True) for combo in ('AB','CD')}
models=Net();optimizer=torch.optim.AdamW(models.parameters(),lr=a.lr,weight_decay=.01)
rand={combo:random.Random(manifest['seeds']['train_'+combo]+a.seed*7919) for combo in ('AB','CD')}

def predict_many(items):
    preds=[]
    models.eval()
    with torch.no_grad():
        for k in range(0,len(items),32):
            x,d,y=batch(items[k:k+32]);pr=models(x,d)
            pred=pr.argmax(-1).tolist()
            preds.extend([z+20 for z in pred])
    models.train()
    return preds

def dev_eval():
    z={}
    for combo, pairs in dev.items():
        left=[x for x,y in pairs];right=[y for x,y in pairs]
        pred_l=predict_many(left);pred_r=predict_many(right)
        z[combo]={'n':len(pairs),
                  'acc':sum((p==o.truth)+(q==v.truth) for p,q,o,v in zip(pred_l,pred_r,left,right))/(2*len(pairs)),
                  'both':sum(p==o.truth and q==v.truth for p,q,o,v in zip(pred_l,pred_r,left,right))/len(pairs),
                  'flip':sum(p!=q for p,q in zip(pred_l,pred_r))/len(pairs),
                  'invariant':sum(p==q for p,q in zip(pred_l,pred_r))/len(pairs)}
    return z

start=time.perf_counter();curve=[]
for step in range(a.steps):
    xs=[]
    if a.train_mode=='single':
        for i in range(a.batch):
            combo='AB' if a.train_combo=='AB_only' else ('AB' if (step*a.batch+i)%2==0 else 'CD')
            xs.append(sample(rand[combo],combo))
    else:
        for i in range(a.batch//2):
            combo='AB' if a.train_combo=='AB_only' else ('AB' if (step*(a.batch//2)+i)%2==0 else 'CD')
            left,right=paired(rand[combo],combo)
            xs.extend((left,right))
    assert len(xs)==a.batch
    x,d,y=batch(xs)
    logits=models(x,d)
    loss=F.cross_entropy(logits,y) if a.model in ('transformer','transformer-candidate-depth') else F.nll_loss(logits,y)
    optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(models.parameters(),1);optimizer.step()
    if (step+1)%a.eval_every==0 or step+1==a.steps:
        row={'step':step+1,'train_loss':float(loss.detach()),'dev':dev_eval()}
        curve.append(row);print('COMPOSITION_DEV='+json.dumps({'model':a.model,'seed':a.seed,**row}),flush=True)
out={'config':vars(a),'params':sum(p.numel() for p in models.parameters()),
     'train_seconds':time.perf_counter()-start,
     'train_compositions':(['AB'] if a.train_combo=='AB_only' else ['AB','CD']),'test_compositions':['AB','CD'],
     'no_AC_access':True,'curve':curve,'dev':curve[-1]['dev'],
     'description':'Exploratory follow-up on AB/CD ONLY; no V2 A+C test access or evaluation'}
Path(a.out).write_text(json.dumps(out,indent=2))
print('DEV_FINAL='+json.dumps({'train_mode':a.train_mode,'train_combo':a.train_combo,'model':a.model,'seed':a.seed,
    'params':out['params'],'seconds':out['train_seconds'],'dev':out['dev']}),flush=True)

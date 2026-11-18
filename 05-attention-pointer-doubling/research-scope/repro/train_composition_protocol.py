"""LOCKED evaluation protocol for compositional scope task.

Source-only train AB+CD; develop AB/CD on independent RNG; eval AC once AFTER
model configuration is locked. Never train / tune on frozen AC labels.
Return predictions for EVERY frozen pair, allowing paired invariance analysis,
paired full-correct, and production-cluster bootstrap. All model inputs are
Python tokenize lexical IDs and optionally INDENT-derived depth, never gold
scope links/def-use edges/CPython runtime state.
"""
import argparse,json,hashlib,math,random,time,statistics
from pathlib import Path
import numpy as np,torch
from torch import nn
import torch.nn.functional as F
from grammar_compositions import make_dataset,sample
from scope_program_benchmark import ID,NS,PAD,NUMBERS

P=argparse.ArgumentParser()
P.add_argument('--model',choices=['transformer','pointer-depth'],required=True)
P.add_argument('--seed',type=int,required=True)
P.add_argument('--steps',type=int,default=1000)
P.add_argument('--batch',type=int,default=32)
P.add_argument('--lr',type=float,default=.001)
P.add_argument('--d',type=int,default=64)
P.add_argument('--layers',type=int,default=2)
P.add_argument('--val-pairs',type=int,default=64)
P.add_argument('--eval-every',type=int,default=250)
P.add_argument('--out',type=str,default='composition.json')
P.add_argument('--sealed-path',type=str,default='frozen_protocol_v2/FROZEN_AC_TEST_256pairs.jsonl')
P.add_argument('--manifest',type=str,default='frozen_protocol_v2/PROTOCOL_MANIFEST.json')
a=P.parse_args()
if a.steps != 1000 or a.batch != 32 or a.d != 64 or a.layers != 2 or a.lr != .001:
    raise ValueError('LOCKED protocol: fixed steps1000, batch32, d64, layers2, LR.001; development smoke must never access V2 test')
random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(2)
CID={k:ID[str(k)] for k in NUMBERS};MAXLEN=110
IDS_TO_CLASS=torch.full((NS,),-1,dtype=torch.long)
for k,v in CID.items():IDS_TO_CLASS[v]=k-20
class Net(nn.Module):
    def __init__(self):
        super().__init__();d=a.d
        self.word=nn.Embedding(NS,d);self.pos=nn.Embedding(MAXLEN,d)
        if a.model=='pointer-depth':self.depth=nn.Embedding(5,d)
        enc=nn.TransformerEncoderLayer(d_model=d,nhead=4,dim_feedforward=d*2,dropout=0,batch_first=True,norm_first=True)
        self.enc=nn.TransformerEncoder(enc,num_layers=a.layers,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(d)
        if a.model=='transformer':self.out=nn.Linear(d,16)
        else:self.q=nn.Linear(d,d);self.k=nn.Linear(d,d);self.recency=nn.Parameter(torch.tensor(0.))
    def forward(self,x,depth):
        B,T=x.shape;t=torch.arange(T)
        h=self.word(x)+self.pos(t)[None]
        if a.model=='pointer-depth':h=h+self.depth(depth.clamp(max=4))
        h=self.norm(self.enc(h,src_key_padding_mask=x==PAD))
        last=(x!=PAD).sum(-1)-1
        query=h[torch.arange(B),last]
        if a.model=='transformer':return self.out(query)
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

def from_sealed(r):
    from scope_program_benchmark import _tokenize
    from grammar_compositions import Example
    source=r['source']
    ids,raw,depth=None,None,None
    raw,ids,depth=_tokenize(source)
    return Example(source,'AC',-1,r['label'],r['literal_candidates'],4,4,raw,ids,depth,0,max(depth),r['production'],'')

manifest=json.loads(Path(a.manifest).read_text())
content=Path(a.sealed_path).read_bytes()
assert hashlib.sha256(content).hexdigest()==manifest['frozen_test_sha256']
assert manifest['train_compositions']==['AB','CD'] and manifest['sealed_evaluation_composition']=='AC'
# FIXED DEV: contains only training combinations, never AC. Dev may be used
# for observation, but protocol chooses final model checkpoint regardless.
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

def sealed_eval_once():
    pairs=[json.loads(line) for line in content.splitlines() if line]
    left=[from_sealed(z['left']) for z in pairs]
    right=[from_sealed(z['right']) for z in pairs]
    pl=predict_many(left);pr=predict_many(right)
    out=[]
    for row,a1,b1 in zip(pairs,pl,pr):
        out.append({'pair_id':row['pair_id'],'production':row['left']['production'],
                    'pred_left':a1,'pred_right':b1,
                    'gold_left':row['left']['label'],'gold_right':row['right']['label'],
                    'correct_left':int(a1==row['left']['label']),
                    'correct_right':int(b1==row['right']['label']),
                    'both_correct':int(a1==row['left']['label'] and b1==row['right']['label']),
                    'pred_flip':int(a1!=b1),'pred_invariant':int(a1==b1),
                    'literal_candidates':len(row['left']['literal_candidates'])})
    from collections import defaultdict
    groups=defaultdict(list)
    for r in out:groups[r['production']].append(r)
    def metric(rs):
        return {'n_pairs':len(rs),'both_correct':statistics.mean(x['both_correct'] for x in rs),
                'pred_flip':statistics.mean(x['pred_flip'] for x in rs),
                'pred_invariant':statistics.mean(x['pred_invariant'] for x in rs),
                'acc':statistics.mean((x['correct_left']+x['correct_right'])/2 for x in rs),
                'literal_candidates_unique':4,'candidate_uniform_single_chance':.25,
                'candidate_uniform_pair_independent_both_chance':.0625,'distinct_labels_without_replacement_both_chance':1/12}
    return {'overall':metric(out),'by_production':{k:metric(v) for k,v in groups.items()},
            'pair_predictions':out}

start=time.perf_counter();curve=[]
for step in range(a.steps):
    xs=[]
    for i in range(a.batch):
        combo='AB' if (step*a.batch+i)%2==0 else 'CD'
        xs.append(sample(rand[combo],combo))
    x,d,y=batch(xs)
    logits=models(x,d)
    loss=F.cross_entropy(logits,y) if a.model=='transformer' else F.nll_loss(logits,y)
    optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(models.parameters(),1);optimizer.step()
    if (step+1)%a.eval_every==0 or step+1==a.steps:
        row={'step':step+1,'train_loss':float(loss.detach()),'dev':dev_eval()}
        curve.append(row);print('COMPOSITION_DEV='+json.dumps({'model':a.model,'seed':a.seed,**row}),flush=True)
# FIRST AND ONLY A+C evaluation for this locked configuration.
sealed=sealed_eval_once()
out={'config':vars(a),'params':sum(p.numel() for p in models.parameters()),
    'train_seconds':time.perf_counter()-start,'frozen_test_sha256':manifest['frozen_test_sha256'],
    'train_compositions':['AB','CD'],'test_composition':'AC','protocol':'frozen AC evaluated once after final fixed 1000 step checkpoint; no tuning on AC labels',
    'curve':curve,'frozen_test':sealed}
Path(a.out).write_text(json.dumps(out,indent=2))
print('COMPOSITION_FINAL='+json.dumps({'model':a.model,'seed':a.seed,'params':out['params'],
    'seconds':out['train_seconds'],'dev':curve[-1]['dev'],'sealed_overall':sealed['overall'],
    'sealed_by_production':sealed['by_production']}),flush=True)

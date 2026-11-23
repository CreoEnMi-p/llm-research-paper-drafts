"""Precommitted V3 model protocol. Never inspect sealed programs before training.

All arms are 2-layer d64 encoders, 5k AdamW steps with paired AB/CD data;
the source-location pointer and address-scramble control have identical
trainable tensors and the same attention, softmax, and scatter-add operations.
Canonical arm uses a fixed renaming-normalization preprocessor; it is NOT the
published Symbol-Invariant Transformer.
"""
import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parent
REPRO = ROOT.parent / 'repro'
sys.path.insert(0, str(REPRO))
import grammar_compositions as gram
from scope_program_benchmark import ID, NS, NUMBERS, PAD, VARIABLES, _tokenize

p = argparse.ArgumentParser()
p.add_argument('--arm', choices=['pointer','pointer-scrambled','pointer-canonical','candidate-classifier'], required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--out', required=True)
p.add_argument('--dev-only', action='store_true')
a = p.parse_args()
STEPS = 5000
BATCH = 32
WIDTH = 64
LAYERS = 2
MAXLEN = 110
LR = 0.001
TRAIN_NAMES = list('abcdefghijkl')
random.seed(a.seed)
np.random.seed(a.seed)
torch.manual_seed(a.seed)
torch.set_num_threads(2)
CID = {v: ID[str(v)] for v in NUMBERS}
IDS_TO_CLASS = torch.full((NS,), -1, dtype=torch.long)
for v, tok in CID.items(): IDS_TO_CLASS[tok] = v - 20
WORD_IDS = {ID[x] for x in VARIABLES}
CANON = [ID[x] for x in VARIABLES]

# This modifies only the AB/CD training generator's identifier vocabulary.
# The V3 test and lexer still recognize the original full 16-name vocabulary.
gram.VARIABLES = TRAIN_NAMES


def make_batch(items, arm):
    T = max(len(z['ids']) for z in items)
    assert T <= MAXLEN, T
    x = torch.full((len(items), T), PAD, dtype=torch.long)
    dep = torch.zeros((len(items), T), dtype=torch.long)
    label = torch.empty(len(items), dtype=torch.long)
    address = torch.full((len(items), T), -1, dtype=torch.long)
    for i, z in enumerate(items):
        ids = list(z['ids'])
        if arm == 'pointer-canonical':
            mapping = {}
            for j, token in enumerate(ids):
                if token in WORD_IDS:
                    if token not in mapping:
                        mapping[token] = CANON[len(mapping)]
                    ids[j] = mapping[token]
        x[i,:len(ids)] = torch.tensor(ids)
        dep[i,:len(ids)] = torch.tensor(z['depth'])
        label[i] = z['label'] - 20
        lit_positions = [j for j, token in enumerate(ids) if IDS_TO_CLASS[token] >= 0]
        assert len(lit_positions) == 4
        original = [int(IDS_TO_CLASS[ids[j]]) for j in lit_positions]
        # Stable per program BODY, shared by counterfactual print interventions.
        body = z['source'].rsplit('print(', 1)[0]
        perm = list(range(4))
        r = random.Random(int(hashlib.sha256(body.encode()).hexdigest()[:16], 16))
        r.shuffle(perm)
        # Both pointer arms execute the same Python/matrix operations and draw
        # the same permutation. Only address-to-value correspondence differs.
        use = perm if arm == 'pointer-scrambled' else sorted(perm)
        for j, target in enumerate(use):
            address[i,lit_positions[j]] = original[target]
    return x, dep, label, address


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        d = WIDTH
        self.word = nn.Embedding(NS, d)
        self.pos = nn.Embedding(MAXLEN, d)
        self.depth = nn.Embedding(5, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4,
            dim_feedforward=d*2, dropout=0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=LAYERS,
                                          enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        if a.arm == 'candidate-classifier':
            self.out = nn.Linear(d, 16)
        else:
            self.q = nn.Linear(d, d)
            self.k = nn.Linear(d, d)
            self.recency = nn.Parameter(torch.tensor(0.))

    def forward(self, x, depth, address, probe=False):
        B,T = x.shape
        tim = torch.arange(T, device=x.device)
        h = self.word(x) + self.pos(tim)[None] + self.depth(depth.clamp(max=4))
        h = self.norm(self.enc(h,src_key_padding_mask=x==PAD))
        last = (x!=PAD).sum(-1)-1
        query = h[torch.arange(B, device=x.device),last]
        if a.arm == 'candidate-classifier':
            logits = self.out(query)
            raw = IDS_TO_CLASS[x]
            counts = torch.zeros(B,16, dtype=torch.long, device=x.device)
            counts.scatter_add_(1,raw.clamp(min=0),(raw>=0).long())
            assert (counts>0).sum(-1).min().item() == 4
            return (logits.masked_fill(counts==0,-1e5), None) if probe else logits.masked_fill(counts==0,-1e5)
        score = (self.q(query)[:,None]*self.k(h)).sum(-1)/math.sqrt(WIDTH)
        score = score + self.recency*tim[None]/max(1,T-1)
        score = score.masked_fill(address<0,-1e5)
        mass = F.softmax(score, dim=-1)
        result = torch.zeros(B,16, device=x.device)
        result.scatter_add_(1,address.clamp(min=0),mass)
        logits = torch.log(result+1e-9)
        return (logits,mass) if probe else logits


def convert(ex):
    return {'source':ex.source,'ids':ex.ids,'depth':ex.depth,'label':ex.truth}


def dev_pairs(combo, seed, n=64):
    return [(convert(x),convert(y)) for x,y in gram.make_dataset(combo,seed,n,True)]


def v3_pairs(path):
    rows=[json.loads(z) for z in Path(path).read_text().splitlines() if z]
    assert len(rows)==448
    out=[]
    for z in rows:
        pair=[]
        for side in ('left','right'):
            item=z[side]
            tok,ids,depth=_tokenize(item['source'])
            pair.append({'source':item['source'],'ids':ids,'depth':depth,'label':item['label']})
        out.append((pair[0],pair[1]))
    return rows,out


def evaluate(net,pairs):
    predictions=[[],[]]
    probes=[[],[]]
    net.eval()
    with torch.no_grad():
        for side in (0,1):
            for k in range(0,len(pairs),32):
                subset=[pair[side] for pair in pairs[k:k+32]]
                x,d,y,addr=make_batch(subset,a.arm)
                logits,mass=net(x,d,addr,probe=True)
                predictions[side].extend([int(v)+20 for v in logits.argmax(-1).tolist()])
                if mass is not None:
                    # Answer labels used only for POST-PREDICTION probes.
                    for bi,target in enumerate(y.tolist()):
                        positions=(addr[bi]==target)
                        probes[side].append(float(mass[bi,positions].sum()))
    net.train()
    r=[]
    for i,(left,right) in enumerate(pairs):
        pl,pr=predictions[0][i],predictions[1][i]
        r.append({'pred_left':pl,'pred_right':pr,
                  'gold_left':left['label'],'gold_right':right['label'],
                  'both':int(pl==left['label'] and pr==right['label']),
                  'flip':int(pl!=pr),
                  'acc':((pl==left['label'])+(pr==right['label']))/2,
                  'correct_position_mass':((probes[0][i]+probes[1][i])/2 if probes[0] else None)})
    metric={key:statistics.mean(z[key] for z in r) for key in ('both','flip','acc')}
    if probes[0]: metric['correct_position_mass']=statistics.mean(z['correct_position_mass'] for z in r)
    return metric,r


# Predeclared fixed AB/CD training/development seeds, no V3 test access before
# the final 5000-step checkpoint. The V3 evaluation is one designated run.
dev={c:dev_pairs(c, {'AB':31071,'CD':41731}[c]) for c in ('AB','CD')}
trainrng={c:random.Random({'AB':11071,'CD':21731}[c]+a.seed*7919) for c in ('AB','CD')}
net=Net()
opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=.01)
start=time.perf_counter()
curve=[]
for step in range(STEPS):
    items=[]
    for i in range(BATCH//2):
        c='AB' if (step*(BATCH//2)+i)%2==0 else 'CD'
        left,right=gram.paired(trainrng[c],c)
        items.extend((convert(left),convert(right)))
    x,d,y,addr=make_batch(items,a.arm)
    logits=net(x,d,addr)
    loss=F.cross_entropy(logits,y) if a.arm=='candidate-classifier' else F.nll_loss(logits,y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(),1)
    opt.step()
    if (step+1) in (1000,2500,5000):
        z={c:evaluate(net,dev[c])[0] for c in ('AB','CD')}
        curve.append({'step':step+1,'dev':z})
        print('V3_DEV='+json.dumps({'arm':a.arm,'seed':a.seed,'step':step+1,'dev':z}),flush=True)

out={'arm':a.arm,'seed':a.seed,'params':sum(p.numel() for p in net.parameters()),
     'train_seconds':time.perf_counter()-start,'steps':STEPS,'batch':BATCH,
     'width':WIDTH,'layers':LAYERS,'lr':LR,'train_combos':['AB','CD'],
     'train_name_pool':''.join(TRAIN_NAMES),'development':curve,
     'mechanism_note':'Never interpret source literal mass alone as proof of learned lexical binding.'}
if not a.dev_only:
    manifest=json.loads((ROOT/'FROZEN_MANIFEST.json').read_text())
    data_path=ROOT/'FROZEN_V3_448pairs.jsonl'
    raw=data_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==manifest['test_sha256']
    assert manifest['n_pairs']==448 and manifest['model_lock']['steps']==STEPS
    rows,pairs=v3_pairs(data_path)
    assert len(rows)==len(pairs)==448
    metrics,pred=evaluate(net,pairs)
    for z,r in zip(rows,pred):
        r.update({'pair_id':z['pair_id'],'family':z['family'],'stratum':z['stratum']})
    out['sealed_v3']={'test_sha256':manifest['test_sha256'],'metrics':metrics,'predictions':pred}
    print('V3_SEALED='+json.dumps({'arm':a.arm,'seed':a.seed,'metrics':metrics}),flush=True)
Path(a.out).write_text(json.dumps(out,indent=2))
print('V3_DONE='+json.dumps({'arm':a.arm,'seed':a.seed,'params':out['params'],
       'dev':curve[-1]['dev'],'has_sealed':not a.dev_only}),flush=True)

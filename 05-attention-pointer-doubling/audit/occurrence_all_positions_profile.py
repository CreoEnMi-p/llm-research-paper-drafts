"""Measure the all-query regime where binary lifting might actually be useful.

On the official Sanford full-sequence task (every source position queried),
compare k serial sparse pointer lookups against log(k) table doubling. Same
trainable parameters, same attention key/recency and same gold labels; this is
an *inference protocol and fwd/bwd compute* comparison, NOT a new trained
model. We use strong fixed recency=6 for both to isolate path composition.

No answers are inserted into the input: official source string has only a
leading hop-count token. Scores include all non-null gold positions; null
answers are masked. CPU timing is not GPU kernel performance.
"""
import importlib.util,json,os,statistics,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import profile,ProfilerActivity

sys.argv=["train_sparse_occurrence.py","--mode","doubling","--steps","1"]
spec=importlib.util.spec_from_file_location("official_sparse",Path(__file__).with_name("train_sparse_occurrence.py"))
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
torch.set_num_threads(2)
def predict_all(m,x,hops,mode):
    B,T=x.shape
    score=m.logits(x)
    vals,idx=torch.topk(score,k=m.K,dim=-1)
    weights=F.softmax(vals,dim=-1)
    qidx=torch.arange(T,device=x.device)[None,:,None].expand(B,T,1)
    qw=torch.ones(B,T,1,device=x.device)
    if mode=="doubling":
        powers=[(idx,weights)]
        for bit in range(1,hops.bit_length()):
            i,w=powers[-1]
            powers.append(m.compose(i,w,i,w,m.K))
        for bit,(i,w) in enumerate(powers):
            if hops&(1<<bit):qidx,qw=m.compose(qidx,qw,i,w,m.K)
    else:
        for _ in range(hops):qidx,qw=m.compose(qidx,qw,idx,weights,m.K)
    ix=qidx
    symbol=x.gather(1,ix.reshape(B,-1)).reshape(ix.shape)
    out=torch.zeros((B,T,m.chars+1),device=x.device)
    out.scatter_add_(2,symbol,qw)
    return out,ix,qw

out=[]
for length in (100,200,400):
    for hops in (7,15):
        task=v.InductionHopsFullSequenceTask(seq_len=length,char_tokens=4,
            min_hops=hops,max_hops=hops,rng=np.random.RandomState(435+length+hops))
        x,y=task.get_batch(8,hops=hops)
        good=(y>0)&(y<=4)
        assert good.any()
        for mode in ("doubling","recurrent"):
            torch.manual_seed(0)
            m=v.SparseOccurrence(mode,4,16,4,6.,32.,"identity")
            def step():
                prob,ix,qw=predict_all(m,x,hops,mode)
                selected=prob.gather(2,y.clamp(0,4)[:,:,None]).squeeze(-1)
                loss=-(torch.log(selected+1e-9)*good).sum()/good.sum()
                loss.backward()
                m.zero_grad(set_to_none=True)
                return prob,ix,qw
            # fixed matches: compare numeric outputs on all valid queries
            with torch.no_grad():
                p,i,w=predict_all(m,x,hops,mode)
                acc=float((((p.argmax(-1)==y)&good).sum()/good.sum()).item())
            for _ in range(3):step()
            times=[]
            for _ in range(8):
                tic=time.perf_counter()
                step()
                times.append((time.perf_counter()-tic)*1000)
            with profile(activities=[ProfilerActivity.CPU],with_flops=True) as pf:
                step()
            row=dict(length=length,hops=hops,mode=mode,batch=len(x),
                n_valid=int(good.sum()),valid_label_accuracy=acc,
                median_fwd_bwd_ms=statistics.median(times),
                mean_fwd_bwd_ms=statistics.mean(times),
                profile_matmul_flops=sum(int(z.flops or 0) for z in pf.key_averages()))
            out.append(row)
            print("ALL_QUERY_COST="+json.dumps(row),flush=True)
Path("all_query_cost.json").write_text(json.dumps({
    "protocol":"Official Sanford generator, all non-null hop_k outputs scored, fixed strong nearest-match prior; 2-thread CPU, 8 timed fwd/bwd steps; same matched model parameters; all-query evaluation unlike terminal-only training.",
    "remarks":["No training in this profile (both arms use same fixed position-biased initialized matcher).","All-query recurrent must follow every position for k steps; doubled table is shared across all positions.","These timings are CPU indexing operations, not GPU fusion or total FLOPs."],
    "rows":out},indent=2))

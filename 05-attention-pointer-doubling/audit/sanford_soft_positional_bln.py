"""Differentiable-position-state pointer doubling proof of concept on PINNED
Sanford task. Random tied token embeddings, NO model training, NO gold labels
used in prediction. The STATE is a distribution over occurrence INDICES, not
a pooled token embedding; each lookup is causal relative to the pointed index.

Base transition A[i,j]: soft score between token at i and token immediately
before value position j, with causal constraint j<=i. Optional near-match
recent-position slope. Square transition matrices to produce A^2, A^4, and
compose query delta @ A @ A^2 @ A^4 = A^7. Outputs are both exact
occurrence-index and four-character accuracies against author metadata.

This O(T^3 log k) **dense reference prototype is NOT efficient**, not a
novel parallel pointer-jumping algorithm, and not a trained neural model.
"""
import json,os,sys,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
sys.path.insert(0,str((Path(os.environ["SANFORD_REPO"])/"src").resolve()))
from tasks import InductionHopsFullSequenceTask
torch.manual_seed(55);torch.set_num_threads(2)
outputs=[]
for seq_len,B in [(100,256),(200,64)]:
    task=InductionHopsFullSequenceTask(seq_len=seq_len,char_tokens=4,min_hops=7,max_hops=7,
                                     rng=np.random.RandomState(19823+seq_len))
    xx,yy,_,_,hopstrings,hopindices=task.get_batch(B,metadata=True,hops=7)
    assert (yy[:,-1]!=task.token_map["~"]).all()
    x=xx.long()
    _,T=x.shape
    assert T==seq_len-1
    source_at=seq_len-3
    targets={}
    for k in (1,3,7):
        targets[k]={
            "pos":torch.tensor([hopindices[b][k][source_at]+1 for b in range(B)],dtype=torch.long),
            "char":torch.tensor([task.token_map[hopstrings[b][k][source_at]] for b in range(B)],dtype=torch.long)
        }
    emb=F.normalize(torch.randn(task.num_tokens,32),dim=-1)
    q=emb[x]
    previous_ids=F.pad(x,(1,0),value=0)[:,:-1]
    keys=emb[previous_ids]
    cos=torch.bmm(q,keys.transpose(1,2))
    causal=(torch.arange(T)[None,:]<=torch.arange(T)[:,None]) & (torch.arange(T)[None,:]>0)
    causal=causal[None].expand(B,-1,-1)
    same=(x[:,:,None]==previous_ids[:,None,:])
    out={}
    for mode in ("soft_unbiased","soft_recency","exact_unbiased","exact_recency"):
        t0=time.perf_counter()
        rec=200. if mode.endswith("recency") else 0.
        logits=400*cos+(rec*torch.arange(T,dtype=torch.float32)/T)[None,None,:]
        if mode.startswith("exact"):
            mask=causal&same
        else:
            mask=causal
        # Unreachable earliest indices have a self-loop to make A stochastic.
        no=~mask.any(-1)
        mask=mask|(no[:,:,None]&torch.eye(T,dtype=torch.bool)[None,:,:])
        A=torch.softmax(logits.masked_fill(~mask,-1e9),dim=-1)
        A2=torch.bmm(A,A)
        A4=torch.bmm(A2,A2)
        stages={
            1:A[:,-1,:],
            3:torch.bmm(A[:,-1:,:],A2).squeeze(1),
        }
        stages[7]=torch.bmm(stages[3][:,None,:],A4).squeeze(1)
        metrics={}
        for k,d in stages.items():
            pos_pred=d.argmax(-1)
            dist_char=torch.zeros(B,task.num_tokens)
            dist_char.scatter_add_(1,x,d)
            char_pred=dist_char.argmax(-1)
            metrics[str(k)]={
                "occurrence_pos_top1":float((pos_pred==targets[k]["pos"]).float().mean()),
                "correct_occurrence_mass":float(d.gather(1,targets[k]["pos"][:,None]).mean()),
                "output_char_top1":float((char_pred==targets[k]["char"]).float().mean()),
                "true_char_probability":float(dist_char.gather(1,targets[k]["char"][:,None]).mean())
            }
        out[mode]={"by_hop":metrics,"forward_seconds":time.perf_counter()-t0}
    outputs.append({"length":seq_len,"n":B,"class_count":4,
                    "experiment":"position-distribution Q state, random tied token embeddings, oracle-free forward",
                    "scores":out})
res={"official_source_commit":os.environ.get("SANFORD_SHA"),
     "setup":"Temperature400 and recency coefficient200 over normalized position, not learned; 4-mode controlled k7, exact variant uses token-ID equality hard mask; no gold label used to compute predicted pointers",
     "complexity":"dense quadratic memory, cubic matmul time per doubling round",
     "outputs":outputs}
Path("sanford_soft_position_pointer.json").write_text(json.dumps(res,indent=2))
print("SOFT_POSITION_POINTER="+json.dumps(res),flush=True)

"""Group-wise stream metrics and independent answer-only AR rollout.

Do NOT run model comparisons until a strong baseline learns all categories.
All category masks and stale old-value labels are independently recomputed
from visible left-to-right write events (the model does not receive them).
"""
import numpy as np
import torch
import torch.nn.functional as F
from audit_stream_task import oracle_groups, specs


def eval_stream_tf(model,heldout,device,a):
    seq,mask=heldout
    groups,stale,events=oracle_groups(seq,mask,a)
    was=model.training; model.eval()
    with torch.inference_mode():
        x=torch.as_tensor(seq,device=device,dtype=torch.long)
        out=model(x[:,:-1])
        prediction=out.argmax(-1)
        scores={}
        for name,sel in groups.items():
            if name=="other":continue
            mk=torch.as_tensor(sel,device=device)
            n=int(mk.sum())
            if n==0:
                scores[name]={"n":0,"acc":None,"nll":None}
            else:
                tgt=x[:,1:][mk]
                qlogits=out[mk].float()
                scores[name]={"n":n,
                 "acc":float((prediction[mk]==tgt).float().mean()),
                 "nll":float(F.cross_entropy(qlogits,tgt))}
        old=torch.as_tensor(stale,device=device)
        oldmask=old!=-1
        stale_rate=(prediction[oldmask]==old[oldmask]).float().mean().item() if oldmask.any() else None
    model.train(was)
    return {"scores":scores,"stale_answer_rate":stale_rate,"event_counts":events}


def passes_learning_gate(stats):
    """Decision locked *before* looking at any pilot results."""
    scores=stats["scores"]
    min_groups=("old_untouched","after_overwrite","new_key")
    return (scores["all"]["acc"] is not None and scores["all"]["acc"]>=.82
        and all(scores[n]["n"]>=100 and scores[n]["acc"]>=.75 for n in min_groups))


def eval_stream_ar(model,heldout,device,a):
    seq,mask=heldout
    groups,stale,events=oracle_groups(seq,mask,a)
    B,T=mask.shape; assert B<=128
    was=model.training;model.eval()
    with torch.inference_mode():
        x=torch.as_tensor(seq,device=device,dtype=torch.long)
        rollout=x.clone()
        predictions=torch.full((B,T),-1,dtype=torch.long,device=device)
        for r in range(T//3):
            ipos=3*r+1
            active=torch.as_tensor(mask[:,ipos],dtype=torch.bool,device=device)
            if not bool(active.any()):
                continue
            # The model sees only the prefix through the query key.
            lg=model(rollout[:,:ipos+1])
            guess=lg[:,-1,:].argmax(-1)
            predictions[active,ipos]=guess[active]
            # Replacing ONLY read answers, never external writes.
            rollout[active,ipos+1]=guess[active]
        changed=torch.as_tensor(mask,device=device)
        assert bool((predictions[changed]>=0).all())
        # Preserve ALL non-query facts, markers and keys verbatim.
        answer_positions=torch.zeros_like(x,dtype=torch.bool)
        answer_positions[:,1:]=torch.as_tensor(mask,device=device)
        assert torch.equal(rollout[~answer_positions],x[~answer_positions])
        for pos in range(T):
            active=torch.as_tensor(mask[:,pos],device=device)
            if bool(active.any()):
                assert torch.equal(rollout[~active,pos+1],x[~active,pos+1])
        scores={}
        y=x[:,1:]
        for name,g in groups.items():
            if name=="other":continue
            mk=torch.as_tensor(g,device=device)
            n=int(mk.sum())
            scores[name]={"n":n,"acc":float((predictions[mk]==y[mk]).float().mean()) if n else None}
        old=torch.as_tensor(stale,device=device)
        om=old!=-1
        stale_rate=float((predictions[om]==old[om]).float().mean()) if bool(om.any()) else None
    model.train(was)
    return {"scores":scores,"stale_answer_rate":stale_rate,"n_rows":B,
            "event_counts":events}

"""Rank-based near-tie recency ablation (only changes near-tied candidates).

Call build_semantic.py first; this replaces the generated file in a CI
checkout, *not* checked-in original model. Tied positions are defined by
the attention logit maximum within --tie_tol, NOT exact symbol identity.
The most recent near-tied candidate receives a larger monotone rank bonus.
This is a strong architectural prior and must be compared at equal compute.
"""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name("build_semantic.py")))
f=Path(__file__).resolve().parents[1]/"pdg_semantic_generated.py"
s=f.read_text()
old='''        time_idx=torch.arange(t,device=logits.device,dtype=logits.dtype)/max(1,t-1)
        return logits+near.to(logits.dtype)*a.tie_recent*time_idx[None,None,:]'''
new='''        rank=near.to(logits.dtype).cumsum(-1)
        return logits+near.to(logits.dtype)*a.tie_recent*rank'''
assert s.count(old)==1,s.count(old)
f.write_text(s.replace(old,new,1))
print("generated recency RANK prior")

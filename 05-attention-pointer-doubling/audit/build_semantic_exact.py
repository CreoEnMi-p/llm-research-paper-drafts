"""A *hard lexical* prior: recency only among near-tied exact-ID candidates.

This is distinct from the old near-logit rank control. At every pointer stage,
decode continuous pointer embedding to closest learned token embedding (hard
nearest neighbor). Compare this decoded token ID against PREVIOUS position ID
for each candidate value. Add a monotone recent-rank bonus *only* when >=2
exact-ID candidates have matching scores within tie_tol.

This injects token identity via a non-differentiable decoder. It is not a
parameter-free generally useful attention trick, and may fail on mixed states.
It does not implement scope masking (a genuinely different requirement).
For this small experiment, vocabulary has 31 tokens; complexity grows O(TV).
"""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name("build_semantic.py")))
path=Path(__file__).resolve().parents[1]/"pdg_semantic_generated.py"
s=path.read_text()
def patch(old,new):
    global s
    assert s.count(old)==1,(old[:100],s.count(old))
    s=s.replace(old,new,1)

patch('    def body_params(s):',
'''    def exact_id_tie_recency(s,logits,query_vec,ids):
        if a.tie_recent <= 0: return logits
        # Hard embedding->token decoding and exact candidate-key filtering;
        # this does not backpropagate through the discrete token assignment.
        with torch.no_grad():
            q=F.normalize(query_vec.detach(),dim=-1)
            vocab=F.normalize(s.emb.weight.detach(),dim=-1)
            decoded=(q @ vocab.transpose(-1,-2)).argmax(-1)  # (batch, t)
            prev=F.pad(ids,(1,0),value=-1)[:,:-1]           # key at value pos j is ids[j-1]
            same=(decoded[:,:,None]==prev[:,None,:])&torch.isfinite(logits)
            exact_logits=logits.masked_fill(~same,float("-inf"))
            best=exact_logits.amax(-1,keepdim=True)
            near=same&(exact_logits>=best-a.tie_tol)
            multiple=(near.sum(-1,keepdim=True)>1)
            eligible=near&multiple
            rank=eligible.to(logits.dtype).cumsum(-1)
            bias=rank*eligible.to(logits.dtype)*a.tie_recent
        return logits+bias

    def body_params(s):''')
patch('lg0=s.near_tie(lg0)','lg0=s.exact_id_tie_recency(lg0,ptr,ids)')
patch('lg1=s.near_tie(lg1)','lg1=s.exact_id_tie_recency(lg1,qq,ids)')
path.write_text(s)
print("generated hard embedding-ID exact-candidate recency")

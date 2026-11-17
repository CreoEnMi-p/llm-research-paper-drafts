"""Build a tie-only-near-match recency control, no changes to baseline file.

The bias is applied *only among positions whose ATTENTION LOGITS are within
--tie_tol of that query's maximum*, after causal masking. This is a soft-score
tie heuristic, **not** an exact token-ID oracle and not true scope awareness.
"""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).with_name("build_reviewer.py")))
path=Path(__file__).resolve().parents[1]/"pdg_reviewer_generated.py"
s=path.read_text()
def patch(x,y):
    global s
    assert s.count(x)==1,(x[:95],s.count(x))
    s=s.replace(x,y,1)
patch('p.add_argument("--seed", type=int, default=0)',
'''p.add_argument("--tie_recent",type=float,default=0.,help="bonus (0 off) among attention top-score near ties")
p.add_argument("--tie_tol",type=float,default=0.03,help="logit proximity defining approximate ties")
p.add_argument("--seed", type=int, default=0)''')
patch('    def body_params(s):',
'''    def near_tie(s,logits):
        if a.tie_recent <= 0: return logits
        t=logits.shape[-1]
        best=logits.amax(-1,keepdim=True)
        near=(logits>=best-a.tie_tol)&torch.isfinite(logits)
        time_idx=torch.arange(t,device=logits.device,dtype=logits.dtype)/max(1,t-1)
        return logits+near.to(logits.dtype)*a.tie_recent*time_idx[None,None,:]

    def body_params(s):''')
patch('                    A0 = torch.softmax(lg0, -1)',
'''                    lg0=s.near_tie(lg0)
                    A0 = torch.softmax(lg0, -1)''')
patch('                        A1_ = torch.softmax(lg1, -1); g_ = torch.sigmoid(s.S["pg"][i])',
'''                        lg1=s.near_tie(lg1)
                        A1_ = torch.softmax(lg1, -1); g_ = torch.sigmoid(s.S["pg"][i])''')
out=Path(__file__).resolve().parents[1]/"pdg_semantic_generated.py"
out.write_text(s)
print("wrote",out)

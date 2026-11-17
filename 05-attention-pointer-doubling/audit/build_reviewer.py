"""Reviewer-grade benchmark adapter: first-query/position metrics and AdamW.

Generates pdg_reviewer_generated.py from the original paper's pdg.py via the
pre-existing build_fair.py; original code and previous results are untouched.
"""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name("build_fair.py")))
out=Path(__file__).resolve().parents[1]/"pdg_fair_generated.py"
src=out.read_text()
def patch(old,new):
    global src
    n=src.count(old)
    if n!=1:
        raise RuntimeError(f"expected exactly one match got {n}: {old[:110]}")
    src=src.replace(old,new,1)

patch('p.add_argument("--seed", type=int, default=0)',
'''p.add_argument("--optimizer", choices=["muon_adam","adamw"],default="muon_adam")
p.add_argument("--adam_lr",type=float,default=0.001)
p.add_argument("--weight_decay",type=float,default=0.01)
p.add_argument("--first_only",type=int,default=0,help="train loss only on first question")
p.add_argument("--index_every",type=int,default=0,help="log per-index eval each N steps")
p.add_argument("--seed", type=int, default=0)''')

patch('    muon = Muon(m, a.lr); adam = torch.optim.Adam(groups, betas=(a.ab1, a.ab2), eps=a.eps)',
'''    if a.optimizer=="adamw":
        muon = None
        adam = torch.optim.AdamW(m.parameters(),lr=a.adam_lr,betas=(0.9,0.95),weight_decay=a.weight_decay)
    else:
        muon = Muon(m,a.lr)
        adam = torch.optim.Adam(groups, betas=(a.ab1,a.ab2), eps=a.eps)''')
patch('        sc = min((step + 1) / max(1, a.warm), 1.0, (a.steps - step) / max(1e-9, a.cool * a.steps)); muon.sc = sc',
'''        sc = min((step + 1) / max(1, a.warm), 1.0, (a.steps - step) / max(1e-9, a.cool * a.steps))
        if muon is not None: muon.sc=sc''')
patch('        mf = mk.float(); loss = (F.cross_entropy',
'''        if a.first_only:
            assert a.task=="hopx"
            first_idx=a.hops*a.pairs*(2+a.fill)
            assert mk[:,first_idx].all()
            mk[:,:first_idx]=False
            mk[:,first_idx+1:]=False
        mf = mk.float(); loss = (F.cross_entropy''')
patch('        muon.step(); adam.step(); m.zero_grad(set_to_none=True)',
'''        if muon is not None: muon.step()
        adam.step(); m.zero_grad(set_to_none=True)''')
patch('    f = torch.compile(m) if a.compile else m; t0 = time.time()',
'''    f = torch.compile(m) if a.compile else m
    t0 = time.time()
    index_curve=[]''')
patch('            l, acc = evaluate(m); curve.append(round(acc, 3))',
'''            l, acc = evaluate(m); curve.append(round(acc, 3))
            if a.index_every and (step+1)%a.index_every==0:
                index_curve.append(dict(step=step+1,by_index=evaluate_by_query(m)))''')
patch('    res["full_curve"] = full_curve',
'''    res["full_curve"] = full_curve
    res["index_curve"] = index_curve
    res["by_index"] = evaluate_by_query(m) if a.task=="hopx" else None
    res["first_acc"] = res["by_index"][0] if res["by_index"] else None
    res["chance_uniform_vocab"] = 1.0 / V
    res["chance_elimination_by_query"] = [1.0/(a.pairs-j) for j in range(a.pairs)] if a.task=="hopx" else None
    res["chance_elimination_mean"] = sum(res["chance_elimination_by_query"])/a.pairs if a.task=="hopx" else None
    res["optimizer"] = a.optimizer''')
# Define a separate evaluator, prevents any previous answer from leaking into
# first-query accuracy. Count rows, not correlated answer slots.
needle='\ndef main():\n'
assert src.count(needle)==1
add='''

def evaluate_by_query(m,E=None):
    E=EV if E is None else E
    assert a.task=="hopx", "By-index only supported for hopx"
    seq=np.asarray(E[0]); B=len(seq)
    offset=a.hops*a.pairs*(2+a.fill)
    count=min(a.pairs,(seq.shape[1]-offset)//2)
    assert count>=1
    pos=torch.arange(offset,offset+2*count,2,device=DEV)
    hits=np.zeros(count,dtype=np.int64)
    m.eval()
    with torch.no_grad():
        for start in range(0,B,16):
            xs=torch.tensor(seq[start:start+16],device=DEV)
            preds=m(xs[:,:-1])[:,pos].argmax(-1)
            answer=xs[:,pos+1]
            hits+=(preds==answer).sum(0).cpu().numpy()
    m.train()
    return [float(z)/B for z in hits]
'''
src=src.replace(needle,add+needle)
dest=Path(__file__).resolve().parents[1]/"pdg_reviewer_generated.py"
dest.write_text(src)
print(f"generated {dest.name}; source byte count={len(src)}")

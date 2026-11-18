"""Same CPU runner, same task, compare compute for L+2 vs doubling vs deep onehop.

Train-step profiler (forward/backward), not GPU latency. Torch profiler counts
partial matmul/conv FLOPs only; full total FLOPs are not certified.
"""
import importlib.util,json,statistics,sys,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch.profiler import profile,ProfilerActivity
HERE=Path(__file__).resolve().parents[1]
src=HERE/"pdg_reviewer_generated.py"
cases=[
 ("rope-7",7,5,[]),
 ("nope-7",7,5,["--nope","1"]),
 ("bln-7",7,3,["--eqh","2","--eqt","3","--ptr","11","--pnb","3"]),
 ("onehop-7",7,6,["--eqh","1","--eqt","3","--ptr","1","--pnb","3"]),
 ("rope-15",15,6,[]),
 ("nope-15",15,6,["--nope","1"]),
 ("bln-15",15,4,["--eqh","2","--eqt","3","--ptr","11","--pnb","3"]),
 ("onehop-15",15,8,["--eqh","1","--eqt","3","--ptr","1","--pnb","3"]),
]
rows=[]
for label,hops,layers,flags in cases:
    sys.argv=["pdg_reviewer_generated.py","--dev","cpu","--threads","2",
        "--task","hopx","--hops",str(hops),"--pairs","8","--rev","1",
        "--L","0","--layers",str(layers),"--V","2000","--seed","0",
        "--sink","0","--alibi","0","--width","64",
        "--init","0.5","--batch","8"]+flags
    spec=importlib.util.spec_from_file_location("audit_profile_"+label.replace("-","_"),src)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    torch.manual_seed(0);torch.set_num_threads(2)
    m=mod.Model(mod.V,mod.a.width,mod.a.layers,mod.a.heads).to(mod.DEV)
    x0,mk=mod.make(8,np.random.default_rng(0))
    x=torch.tensor(x0,device=mod.DEV)
    mask=torch.tensor(mk,device=mod.DEV).float()
    def step():
        out=m(x[:,:-1])
        y=x[:,1:]
        loss=(F.cross_entropy(out.float().reshape(-1,mod.V),y.reshape(-1),reduction="none")*mask.reshape(-1)).sum()/mask.sum()
        loss.backward();m.zero_grad(set_to_none=True)
    for _ in range(4):step()
    ms=[]
    for _ in range(12):
        t=time.perf_counter();step();ms.append((time.perf_counter()-t)*1000)
    with profile(activities=[ProfilerActivity.CPU],with_flops=True) as prof:
        step()
    flops=sum(int(row.flops or 0) for row in prof.key_averages())
    rows.append(dict(arm=label,hops=hops,layers=layers,parameters=sum(p.numel() for p in m.parameters()),
        median_step_ms=statistics.median(ms),mean_step_ms=statistics.mean(ms),
        partial_torch_profiler_flops=flops,context=x.shape[1]-1,batch=8))
print("MATCHED_COST_RESULT="+json.dumps(rows),flush=True)
(HERE/"audit"/"cost_matched_onehop.json").write_text(json.dumps(rows,indent=2))

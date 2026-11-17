"""Same-runner CPU training-step cost audit, plus torch.profiler matmul FLOPs.

This is *not* hardware-independent total FLOPs: PyTorch profiles only selected
matmul/conv ops; normalization, softmax, pointer similarity and dispatch
overhead are not exhaustively counted. Use wall time plus reported partial FLOPs.
"""
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

HERE=Path(__file__).resolve().parents[1]
path=HERE/"pdg_fair_generated.py"
assert path.exists()
cases=[
 ("rope-15", ["--sink","0","--alibi","0","--layers","4"]),
 ("nope-15", ["--sink","0","--alibi","0","--nope","1","--layers","4"]),
 ("doubling-15", ["--sink","0","--alibi","0","--eqh","2","--eqt","3","--ptr","11","--pnb","3","--layers","4"]),
 ("looped-nope-15",["--sink","0","--alibi","0","--nope","1","--loop","1","--layers","8"])
]
res=[]
for name,flags in cases:
    sys.argv=["pdg_fair_generated.py","--dev","cpu","--threads","2",
            "--task","hopx","--hops","15","--pairs","8","--rev","1",
            "--V","2000","--L","0","--width","64","--seed","0",
            "--init","0.5","--batch","16","--lr","0.02"]+flags
    spec=importlib.util.spec_from_file_location("pdg_cost_"+name.replace("-","_"),path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    torch.manual_seed(0)
    m=mod.Model(mod.V,mod.a.width,mod.a.layers,mod.a.heads).to(mod.DEV)
    sq,mk=mod.make(16,np.random.default_rng(0));x=torch.tensor(sq,device=mod.DEV)
    mask=torch.tensor(mk,device=mod.DEV).float()
    def step():
        pred=m(x[:,:-1])
        loss=(F.cross_entropy(pred.float().reshape(-1,mod.V),
              x[:,1:].reshape(-1),reduction="none")*mask.reshape(-1)).sum()/mask.sum()
        loss.backward();m.zero_grad(set_to_none=True)
        return loss
    for _ in range(4):step()
    times=[]
    for _ in range(10):
        t=time.perf_counter();step();times.append(time.perf_counter()-t)
    with profile(activities=[ProfilerActivity.CPU],record_shapes=True,with_flops=True) as prof:
        step()
    matmul_flops=sum((e.flops or 0) for e in prof.key_averages())
    res.append(dict(name=name,median_fb_ms=statistics.median(times)*1000,
                    mean_fb_ms=statistics.mean(times)*1000,
                    max_fb_ms=max(times)*1000,
                    parameters=sum(p.numel() for p in m.parameters()),
                    effective_layers=mod.a.layers, matmul_profile_flops=matmul_flops,
                    flops_warning="partial CPU matmul/conv profile; not full exact training FLOPs"))
print("COST_AUDIT_FINAL="+json.dumps(res),flush=True)
(HERE/"audit"/"cost_profile.json").write_text(json.dumps(res,indent=2))

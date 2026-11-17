"""Same-runner benchmark comparing k-hop sparse recurrent vs doubled index tables.

One final query per sequence (official Sanford terminal protocol). The same
tied embedding / QK / learned-recency coefficients and source inputs are
shared. This is fwd + backward wall time, CPU only; torch profiler FLOPs
does NOT count all index ops and softmax kernels. Do not equate kernel count.
"""
import importlib.util,json,os,statistics,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.profiler import profile,ProfilerActivity
import numpy as np

sys.argv=["train_sparse_occurrence.py","--mode","doubling","--steps","1"]
path=Path(__file__).with_name("train_sparse_occurrence.py")
spec=importlib.util.spec_from_file_location("train_sparse_occurrence",path)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
torch.set_num_threads(2)
rows=[]
for length in (100,200,400):
    for hops in (7,15):
        task=mod.InductionHopsFullSequenceTask(seq_len=length,char_tokens=4,
            min_hops=hops,max_hops=hops,rng=np.random.RandomState(23+length+hops))
        x,y=mod.official_batch(task,16,hops)
        for mode in ("doubling","recurrent","dense"):
            torch.manual_seed(0)
            m=mod.SparseOccurrence(mode,4,16,4,4.0,32.0,"identity")
            # identical init, no training; time only compares algorithms
            def step():
                prob,_=m(x,hops)
                loss=-torch.log(prob.gather(1,y[:,None])+1e-9).mean()
                loss.backward()
                m.zero_grad(set_to_none=True)
            for _ in range(4):step()
            timings=[]
            for _ in range(10):
                tic=time.perf_counter();step()
                timings.append(1000*(time.perf_counter()-tic))
            with profile(activities=[ProfilerActivity.CPU],with_flops=True) as p:
                step()
            flops=sum(int(z.flops or 0) for z in p.key_averages())
            r=dict(mode=mode,hops=hops,seq_len=length,batch=len(x),K=4,
                median_fwd_bwd_ms=statistics.median(timings),
                mean_fwd_bwd_ms=statistics.mean(timings),
                partial_profiled_flops=flops,
                params=sum(p.numel() for p in m.parameters()))
            print("OCCURRENCE_COST="+json.dumps(r),flush=True)
            rows.append(r)
Path("occurrence_cost.json").write_text(json.dumps({
    "protocol":"Same CPU runner, torch2 threads, 4 warmups+10 timed fwd/bwd, batch16 official Sanford inputs, same model parameters/init; dense is O(T^3), sparse doubler O(T² + TK²log k), recurrent single-query O(T² + kK²)",
    "warnings":["partial matmul profiler flops omit index gathering, softmax and topk; walltime CPU only","single final query favors recurrent, all-query may favor doubling but not measured"],"rows":rows
},indent=2))

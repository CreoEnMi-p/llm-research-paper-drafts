"""Replay *identical* seed/flags to diagnose cross-Action-run variability.
No hypothesis test unless bitwise stability verified.
"""
import json,os,platform,subprocess,sys
from pathlib import Path
import torch
ROOT=Path(__file__).parent
print(json.dumps({"torch":torch.__version__,"platform":platform.platform(),
                  "processor":platform.processor(),"GITHUB_SHA":os.environ.get("GITHUB_SHA")}),flush=True)
BASE=["--width","32","--layers","2","--heads","2","--L","64",
"--batch","32","--steps","300","--pairs","16","--lpsX","1","--lpsC","64",
"--dev","cpu","--task","mqar","--lr","0.05","--elr","0.04",
"--init","0.5","--alibi","0","--sink","0","--mix","g",
"--lps","0.25","--lpsI","1","--lpsM","0.2"]
res=[]
for seed,threads,rep in ((1,2,0),(1,2,1),(2,2,0),(2,2,1),
                         (1,1,0),(1,1,1),(2,1,0),(2,1,1)):
  tag=f"replay:s{seed}:t{threads}:r{rep}"
  proc=subprocess.run([sys.executable,"rc.py",*BASE,"--seed",str(seed),
                     "--threads",str(threads),"--tag",tag],
                     cwd=ROOT,capture_output=True,text=True,timeout=600)
  if proc.returncode:
     print("ERROR",tag,proc.stdout[-2000:],proc.stderr[-2000:],flush=True)
     raise RuntimeError(tag)
  d=json.loads((ROOT/"results.jsonl").read_text().splitlines()[-1])
  assert d["tag"]==tag
  r=dict(seed=seed,threads=threads,rep=rep,loss=d["val"],acc=d["acc"],
         train_s=d["train_s"],curve=d["curve"],S_l=d.get("S_l"))
  res.append(r)
  (ROOT/"audit_replay.json").write_text(json.dumps(res,indent=2))
  print(json.dumps(r),flush=True)
print("REPLAY COMPLETE",flush=True)

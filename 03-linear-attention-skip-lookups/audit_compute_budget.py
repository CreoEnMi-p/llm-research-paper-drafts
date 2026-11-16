"""Compute-budget sanity control for the 300-step sequential exact gate.

CPU wall-time (not FLOPs) matching: baseline g C64 with 1000 steps
instead of gated g with 300 steps. Original baseline 300-step and exact
gate 300-step observations are in audit_gate_matrix.py.
"""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).parent
BASE=["--width","32","--layers","2","--heads","2",
"--L","64","--batch","32","--pairs","16","--lpsX","1","--lpsC","64",
"--dev","cpu","--threads","2","--task","mqar","--lr","0.05",
"--elr","0.04","--init","0.5","--alibi","0","--sink","0","--mix","g"]
rows=[]
for seed in (0,1,2):
  for steps in (1000,):
    tag=f"budget_g1000:s{seed}"
    proc=subprocess.run([sys.executable,"rc.py",*BASE,
       "--steps",str(steps),"--seed",str(seed),"--tag",tag],
       cwd=ROOT,capture_output=True,text=True,timeout=900)
    if proc.returncode:
      print("FAILED",tag,proc.stdout[-2000:],proc.stderr[-2000:],flush=True)
      raise RuntimeError(tag)
    data=json.loads((ROOT/"results.jsonl").read_text().splitlines()[-1])
    assert data["tag"]==tag
    result=dict(seed=seed,steps=steps,loss=data["val"],acc=data["acc"],
                train_s=data["train_s"],steps90=data["steps90"],curve=data["curve"])
    rows.append(result)
    (ROOT/"audit_budget_g1000.json").write_text(json.dumps(rows,indent=2))
    print(json.dumps(result),flush=True)
print("COMPLETE wall-time-matched CPU baseline (hardware-specific)",flush=True)

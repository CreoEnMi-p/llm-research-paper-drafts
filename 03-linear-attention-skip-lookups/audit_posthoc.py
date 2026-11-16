"""Frozen-checkpoint causal intervention sanity audit. 5 paired seeds,
same training schedule and fixed heldout MQAR evaluation. Compares targeted
answer updates against equal-count query-key suppression and complete query
phase suppression. This checks intervention *at inference*, not learning.
"""
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).parent
COMMON = [
    "--width","32","--layers","2","--heads","2",
    "--L","64","--batch","32","--steps","300",
    "--pairs","16","--lpsX","1","--lpsC","16",
    "--dev","cpu","--threads","2","--task","mqar",
    "--lr","0.04","--elr","0.04","--init","0.5",
    "--alibi","0","--sink","0","--posthoc","1",
]
out=[]
for seed in range(5):
    for arm in ("g","D"):
        tag=f"posthoc:seed{seed}:{arm}"
        args=[sys.executable,"rc.py",*COMMON,"--seed",str(seed),
              "--mix",arm,"--tag",tag]
        ret=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=360)
        if ret.returncode:
            print(tag,ret.stdout[-2500:],ret.stderr[-2500:],flush=True)
            raise RuntimeError(f"Failed: {tag}")
        last=(ROOT/"results.jsonl").read_text().splitlines()[-1]
        d=json.loads(last)
        assert d["tag"]==tag
        record={"seed":seed,"arm":arm,"base":{"loss":d["val"],"accuracy":d["acc"]},
                "interventions":d["posthoc"]}
        out.append(record)
        print(json.dumps(record),flush=True)
(ROOT/"audit_posthoc_summary.json").write_text(json.dumps(out,indent=2))
print("Summary written; interventions applied to already-trained model")

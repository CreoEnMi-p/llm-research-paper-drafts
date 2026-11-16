"""Small *single-seed* paired-initialization smoke benchmark, not paper evidence.

Fixed original rc.py seed=0. Three arms; ONLY recurrence varies:
g baseline, old chunk-start predicted-skip, self-consistent exact gate.
No detach/margin yet (ablate these only after causal correctness).
"""
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).parent
COMMON=[
    "--width","32","--layers","2","--heads","2","--L","64",
    "--batch","32","--steps","300","--pairs","16",
    "--lpsX","1","--lpsC","64","--dev","cpu","--threads","2",
    "--task","mqar","--lr","0.05","--elr","0.04",
    "--init","0.5","--alibi","0","--sink","0","--mix","g",
]
ARMS=[
    ("baseline",[]),
    ("chunkstart_gate",["--lps","0.25"]),
    ("exact_gate",["--lps","0.25","--lpsI","1"]),
]
out=[]
for label,extra in ARMS:
    tag="gemini_audit:"+label
    r=subprocess.run([sys.executable,"rc.py",*COMMON,*extra,"--tag",tag],
                     cwd=ROOT,capture_output=True,text=True,timeout=900)
    if r.returncode:
        print(label,r.stderr[-3000:],r.stdout[-2000:],flush=True)
        raise RuntimeError(label)
    lines=(ROOT/"results.jsonl").read_text().splitlines()
    record=json.loads(lines[-1])
    assert record["tag"]==tag,(label,record.get("tag"))
    row=dict(arm=label,seed=0,loss=record["val"],acc=record["acc"],
             steps90=record["steps90"],train_s=record["train_s"],
             gst=record.get("gst"))
    out.append(row)
    print(json.dumps(row),flush=True)
(ROOT/"audit_gate_pilot.json").write_text(json.dumps(out,indent=2))
print("Caution: one fixed-seed synthetic MQAR run only; no inference of significance.",flush=True)

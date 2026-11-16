"""Test Gemini's stability-collapse claims under the precise original settings.
Separate D/M and write-only vs full-update erasure. Two paired seeds.
"""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument("--case",choices=("delta","long"),required=True);cfg=p.parse_args()
COMMON=["--width","32","--layers","2","--heads","2","--steps","300","--pairs","16",
        "--lpsX","1","--lpsC","16","--dev","cpu","--threads","2",
        "--task","mqar","--lr","0.05","--elr","0.04","--init","0.5",
        "--alibi","0","--sink","0"]
cases={
 "delta":["--L","64","--batch","32","--mix","d"],
 "long":["--L","256","--batch","8","--mix","g"],
}
arms={
 "base":[],
 "chunkstart":["--lps","0.25"],
 "exact":["--lps","0.25","--lpsI","1"],
 "margin":["--lps","0.25","--lpsI","1","--lpsM","0.2"],
 "detach_margin":["--lps","0.25","--lpsI","1","--lpsM","0.2","--lpsD","1"],
 "margin_full_update":["--lps","0.25","--lpsI","1","--lpsM","0.2","--lpsE","1"],
}
plan=("base","chunkstart","exact","margin","detach_margin","margin_full_update") if cfg.case=="delta" else ("base","chunkstart","exact","margin","detach_margin")
out=[]
for seed in (0,1):
 for arm in plan:
    tag=f"gemini_stability:{cfg.case}:s{seed}:{arm}"
    cmd=[sys.executable,"rc.py",*COMMON,*cases[cfg.case],
         "--seed",str(seed),*arms[arm],"--tag",tag]
    res=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=1000)
    if res.returncode:
       print("FAILED",tag,res.stdout[-1000:],res.stderr[-2500:],flush=True)
       raise RuntimeError(tag)
    d=json.loads((ROOT/"results.jsonl").read_text().splitlines()[-1])
    assert d["tag"]==tag
    rec=dict(case=cfg.case,seed=seed,arm=arm,loss=d["val"],acc=d["acc"],
             steps90=d["steps90"],train_s=d["train_s"],gst=d.get("gst"))
    out.append(rec)
    (ROOT/f"audit_stability_{cfg.case}.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(rec),flush=True)
print("COMPLETE",cfg.case,flush=True)

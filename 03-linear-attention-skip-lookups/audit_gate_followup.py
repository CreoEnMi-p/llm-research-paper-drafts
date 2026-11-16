"""Independent follow-up to Gemini audit: stronger controls and true phase diagnostics.

Usage: python audit_gate_followup.py --suite mqar2|p4_hop2|margin
Results are only synthetic-task observations; no novelty/causal claims.
"""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--suite",choices=["mqar2","p4_hop2","margin"],required=True)
a=p.parse_args()
ROOT=Path(__file__).parent
MODEL=["--width","32","--layers","2","--heads","2","--dev","cpu","--threads","2",
       "--lr","0.05","--elr","0.04","--init","0.5","--alibi","0","--sink","0",
       "--mix","g","--lpsX","1"]
SETTINGS={
 "mqar2":["--L","128","--batch","16","--steps","450","--pairs","16","--lpsC","16","--task","mqar2"],
 "p4":["--L","64","--batch","32","--steps","300","--pairs","4","--lpsC","16","--task","mqar"],
 "hop2":["--L","128","--batch","16","--steps","450","--pairs","16","--lpsC","16","--task","hop2"],
 "c64":["--L","64","--batch","32","--steps","300","--pairs","16","--lpsC","64","--task","mqar"],
}
GATES={
 "base":[],
 "prev":["--lps","0.25","--lpsI","1"],
 "key":["--lps","0.25","--lpsI","1","--lpsR","2"],
 "rowshuffle":["--lps","0.25","--lpsI","1","--lpsc","1"],
 "margin":["--lps","0.25","--lpsI","1","--lpsM","0.2"],
 "detach":["--lps","0.25","--lpsI","1","--lpsD","1"],
 "both":["--lps","0.25","--lpsI","1","--lpsD","1","--lpsM","0.2"],
}
if a.suite=="mqar2":
    plan=[
      (0,"mqar2","prev"),(0,"mqar2","key"),(0,"mqar2","rowshuffle"),
      *[(s,"mqar2",arm) for s in (1,2) for arm in ("base","prev","key")],
    ]
elif a.suite=="p4_hop2":
    plan=[
      *[(s,"p4",arm) for s in (1,2) for arm in ("base","prev","key")],
      *[(s,"hop2",arm) for s in (0,1) for arm in ("base","prev","key")],
    ]
else:
    plan=[(s,"c64",arm) for s in (1,2) for arm in ("margin","detach","both")]
out=[]
for seed,case,arm in plan:
    tag=f"gate_followup:{a.suite}:{case}:s{seed}:{arm}"
    r=subprocess.run([sys.executable,"rc.py",*MODEL,*SETTINGS[case],
                    "--seed",str(seed),*GATES[arm],"--tag",tag],
                     cwd=ROOT,capture_output=True,text=True,timeout=900)
    if r.returncode:
        print("FAILED:",tag,r.stdout[-1500:],r.stderr[-2500:],flush=True)
        raise RuntimeError(tag)
    rows=(ROOT/"results.jsonl").read_text().splitlines()
    d=json.loads(rows[-1]);assert d["tag"]==tag
    rec={"seed":seed,"case":case,"arm":arm,"loss":d["val"],"acc":d["acc"],
         "steps90":d["steps90"],"train_s":d["train_s"],"gst":d.get("gst")}
    out.append(rec)
    (ROOT/f"audit_followup_{a.suite}.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(rec),flush=True)
print("SUITE COMPLETE",a.suite,flush=True)

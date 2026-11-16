"""Apply the *same* cosine margin to previous-query, current-key, and shuffled-value gates.
Compare with no-margin results in the first paired matrix; no new baseline claim.
"""
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).parent
COMMON=["--width","32","--layers","2","--heads","2",
        "--L","64","--batch","32","--steps","300","--pairs","16",
        "--lpsX","1","--lpsC","64","--dev","cpu","--threads","2",
        "--task","mqar","--lr","0.05","--elr","0.04","--init","0.5",
        "--alibi","0","--sink","0","--mix","g",
        "--lps","0.25","--lpsI","1","--lpsM","0.2"]
ARMS=[("prev_margin",[]),("key_margin",["--lpsR","2"]),
      ("shuffle_margin",["--lpsc","1"])]
results=[]
for seed in (0,1,2):
    for name,extra in ARMS:
        tag=f"key_margin:s{seed}:{name}"
        res=subprocess.run([sys.executable,"rc.py",*COMMON,"--seed",str(seed),
                            *extra,"--tag",tag],cwd=ROOT,
                           capture_output=True,text=True,timeout=900)
        if res.returncode:
            print("FAILED",tag,res.stderr[-3000:],flush=True)
            raise RuntimeError(tag)
        row=json.loads((ROOT/"results.jsonl").read_text().splitlines()[-1])
        assert row["tag"]==tag
        item=dict(seed=seed,arm=name,loss=row["val"],acc=row["acc"],
                  train_s=row["train_s"],gst=row.get("gst"))
        results.append(item)
        (ROOT/"audit_key_margin.json").write_text(json.dumps(results,indent=2))
        print(json.dumps(item),flush=True)
print("COMPLETE same-margin current-key fairness matrix",flush=True)

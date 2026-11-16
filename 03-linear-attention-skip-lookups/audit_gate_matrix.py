"""Small paired-seed audit matrix. Results are exploratory, not conference evidence."""
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).parent
MODEL=["--width","32","--layers","2","--heads","2","--dev","cpu",
       "--threads","2","--lr","0.05","--elr","0.04","--init","0.5",
       "--alibi","0","--sink","0","--mix","g","--lpsX","1"]
BASE={
 "c64":["--L","64","--batch","32","--steps","300","--pairs","16",
        "--lpsC","64","--task","mqar"],
 "p4":["--L","64","--batch","32","--steps","300","--pairs","4",
       "--lpsC","16","--task","mqar"],
 "mqar2":["--L","128","--batch","16","--steps","450","--pairs","16",
          "--lpsC","16","--task","mqar2"],
}
ARMS={
 "base":[],
 "prev_exact":["--lps","0.25","--lpsI","1"],
 "current_key_exact":["--lps","0.25","--lpsI","1","--lpsR","2"],
 "row_shuffle_exact":["--lps","0.25","--lpsI","1","--lpsc","1"],
 "prev_detach":["--lps","0.25","--lpsI","1","--lpsD","1"],
 "prev_margin":["--lps","0.25","--lpsI","1","--lpsM","0.2"],
 "prev_detach_margin":["--lps","0.25","--lpsI","1","--lpsD","1","--lpsM","0.2"],
}
PLAN=(
  [(0,"c64",arm) for arm in ("base","prev_exact","current_key_exact","row_shuffle_exact")] +
  [(1,"c64",arm) for arm in ("base","prev_exact","current_key_exact","row_shuffle_exact")] +
  [(2,"c64",arm) for arm in ("base","prev_exact","current_key_exact")] +
  [(0,"p4",arm) for arm in ("base","prev_exact")] +
  [(0,"mqar2",arm) for arm in ("base","prev_exact")] +
  [(0,"c64",arm) for arm in ("prev_detach","prev_margin","prev_detach_margin")]
)
output=[]
for seed,case,arm in PLAN:
    tag=f"gate_matrix:{case}:s{seed}:{arm}"
    cmd=[sys.executable,"rc.py",*MODEL,*BASE[case],
         "--seed",str(seed),*ARMS[arm],"--tag",tag]
    run=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=900)
    if run.returncode:
        print(tag,"FAILED",run.stdout[-1500:],run.stderr[-3000:],flush=True)
        raise RuntimeError(tag)
    row=json.loads((ROOT/"results.jsonl").read_text().splitlines()[-1])
    assert row["tag"]==tag
    rec=dict(seed=seed,case=case,arm=arm,loss=row["val"],acc=row["acc"],
             steps90=row["steps90"],train_s=row["train_s"],gst=row.get("gst"))
    output.append(rec)
    (ROOT/"audit_gate_matrix.json").write_text(json.dumps(output,indent=2))
    print(json.dumps(rec),flush=True)
print("All comparisons require repeated seeds and correct strong baselines before publication.",flush=True)

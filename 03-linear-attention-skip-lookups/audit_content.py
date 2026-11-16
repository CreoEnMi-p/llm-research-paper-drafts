"""Pre-registered counterexample controls for Predicted-Skip on MQAR.
Matched 300-step budget, 5 paired seeds (train/init), independent fixed eval.
All tested variants have one scalar per head except the original no-extra-gate
baseline. Stop the content-specific claim if the matched uniform or shuffled
content control matches predicted-skip. Outputs are exploratory, not
multiple-hypothesis-corrected conference evidence.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
COMMON = ["--width","32","--layers","2","--heads","2",
          "--L","64","--batch","32","--steps","300","--pairs","16",
          "--lpsX","1","--lpsC","16","--dev","cpu","--threads","2",
          "--task","mqar","--lr","0.04","--elr","0.04",
          "--init","0.5","--alibi","0","--sink","0","--mix","g"]
ARMS = [
    ("no_extra_gate",[]),
    ("predicted_skip",["--lps","0.25"]),
    ("shuffled_value",["--lps","0.25","--lpsc","1"]),
    ("uniform_write_suppress",["--lps","0.25","--lpsU","1"]),
    ("fixed_cosine_gamma3",["--lps","3","--lpsf","1"]),
    ("current_key_read",["--lps","0.25","--lpsR","2"]),
]
records=[]
for seed in range(5):
    for name, flags in ARMS:
        tag=f"control:seed{seed}:{name}"
        run=subprocess.run([sys.executable,"rc.py",*COMMON,
                            "--seed",str(seed),"--tag",tag,*flags],
                           cwd=ROOT,capture_output=True,text=True,timeout=360)
        if run.returncode:
            print(tag,run.stdout[-2000:],run.stderr[-2000:],flush=True)
            raise RuntimeError(tag)
        item=json.loads((ROOT/"results.jsonl").read_text().splitlines()[-1])
        assert item["tag"]==tag
        record=dict(seed=seed,variant=name,loss=item["val"],
                    accuracy=item["acc"],train_seconds=item["train_s"])
        records.append(record)
        print(json.dumps(record),flush=True)
(ROOT/"audit_content_summary.json").write_text(json.dumps(records,indent=2))
print("PRE-REGISTERED DECISION: predicted-skip must outperform shuffled and uniform controls on >=4/5 seeds; otherwise stop the read-value content story.")
print("Wrote audit_content_summary.json")

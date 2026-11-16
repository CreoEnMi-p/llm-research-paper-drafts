"""Fairer local feature-only write gate vs prior-read on memory updates.

The local control has no access to the recurrent state in its multiplicative
write factor. It is a generous O(H*d) extra-parameter input-role predictor;
read-based exact gate has only O(H) gamma scalars (plus shared model).
The phase oracle knows task-specific answer positions and is explicitly
non-deployable, intended solely as an upper bound and causal mechanism test.

Keep train and heldout seeds fixed. Compute controls:
 local650 (equal token) and local2500 (approx equal wall time to exact650).
"""
import argparse, subprocess, sys, json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--task",choices=["mqar_rewrite","mqar_alias"],required=True)
p.add_argument("--seed",type=int,required=True)
a=p.parse_args()
ROOT=Path(__file__).parent
COMMON=[
 "--width","32","--layers","2","--heads","2","--L","96",
 "--batch","24","--pairs","12","--lpsX","1","--lpsC","16",
 "--dev","cpu","--threads","2","--task",a.task,
 "--lr","0.05","--elr","0.04","--init","0.5",
 "--alibi","0","--sink","0","--mix","g",
 "--update_eval","1","--update_heldout_seed","92741",
]
ARMS=[
 ("local_input_gate_650",["--localS","1","--steps","650"]),
 ("local_input_gate_2500",["--localS","1","--steps","2500"]),
 ("phase_answer_oracle_650",["--phaseOracle","1","--steps","650"]),
]
out=[]
for arm,flags in ARMS:
    tag=f"local_read_falsifier:{a.task}:s{a.seed}:{arm}"
    r=subprocess.run([sys.executable,"rc.py",*COMMON,*flags,
                     "--seed",str(a.seed),"--tag",tag],
                     cwd=ROOT,capture_output=True,text=True,timeout=1050)
    if r.returncode:
        print("FAILED",tag,r.stdout[-2500:],r.stderr[-3500:],flush=True)
        raise RuntimeError(tag)
    rec=json.loads((ROOT/"results.jsonl").read_text().splitlines()[-1])
    assert rec["tag"]==tag
    ar=rec["update_eval"]["autoregressive"]
    item={"task":a.task,"seed":a.seed,"arm":arm,"train_s":rec["train_s"],
          "steps":rec["args"]["steps"],"total_params":rec["total_params"],
          "initial_AR":ar["query_A"]["accuracy"],
          "changed_AR":ar["query_B_changed"]["accuracy"],
          "untouched_AR":ar["query_B_unchanged"]["accuracy"],
          "overall_AR":ar["all_queries"]["accuracy"],
          "stale_old_AR":rec["update_eval"].get("old_value_prediction_rate_changed",{}).get("autoregressive")}
    out.append(item)
    (ROOT/f"local_control_{a.task}_{a.seed}.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(item),flush=True)
print("DONE",a.task,a.seed)

"""Test channel-wise independent erase/write/decay recurrence as an expressivity control.

This **does not** claim an official NVIDIA GDN2 implementation, speed,
convergence or beta-guard fidelity. It tests a mathematical recurrence
using the same toy embedding/block/qkv pipeline and 650-step token budget.
Compare against previous PR #5/#6 logs; do not sweep hyperparameters on
the test set.
"""
import argparse,subprocess,sys,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--task",choices=["mqar_rewrite","mqar_alias"],required=True)
p.add_argument("--seed",type=int,required=True)
a=p.parse_args()
ROOT=Path(__file__).parent
COMMON=["--width","32","--layers","2","--heads","2","--L","96",
        "--batch","24","--steps","650","--pairs","12",
        "--lpsX","1","--lpsC","16","--dev","cpu","--threads","2",
        "--task",a.task,"--lr","0.05","--elr","0.04",
        "--init","0.5","--alibi","0","--sink","0",
        "--update_eval","1","--update_heldout_seed","92741"]
tag=f"gdn2_equation:{a.task}:s{a.seed}"
cmd=[sys.executable,"rc.py",*COMMON,"--mix","Z","--seed",str(a.seed),"--tag",tag]
r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=1200)
if r.returncode:
    print("FAIL",tag,r.stdout[-3500:],r.stderr[-5500:],flush=True)
    raise RuntimeError(tag)
rec=json.loads((ROOT/"results.jsonl").read_text().splitlines()[-1]);assert rec["tag"]==tag
u=rec["update_eval"];arr=u["autoregressive"]
row={"task":a.task,"seed":a.seed,"arm":"gdn2_equation_reference",
     "train_seconds":rec["train_s"],"total_parameters":rec["total_params"],
     "heldout_seed":u["heldout_seed"],"val_loss":rec["val"],
     "AR_initial":arr["query_A"]["accuracy"],
     "AR_new_or_updated":arr["query_B_changed"]["accuracy"],
     "AR_unchanged":arr["query_B_unchanged"]["accuracy"],
     "AR_overall":arr["all_queries"]["accuracy"],
     "stale_old":u.get("old_value_prediction_rate_changed",{}).get("autoregressive")}
(ROOT/f"gdn2_equation_{a.task}_{a.seed}.json").write_text(json.dumps(row,indent=2))
print(json.dumps(row),flush=True)

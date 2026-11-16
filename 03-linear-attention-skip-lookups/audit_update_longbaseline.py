"""Compute-time calibrated controls for the preregistered update falsifiers.

The slow exact gates took about 4x ordinary GLA training time at 650
steps. These controls give the ordinary non-gated architectures about the
same time (GLA 2500 steps, Delta 2000 steps), holding model width,
architecture, optimizer family, batch size, task, train-data RNG seed,
and heldout seed fixed.

All values are controlled BEFORE running longer baselines; use task-by-task
and seed-by-seed comparison against PR #5's 650-step gated arms.
Token budgets intentionally differ because wall time, not tokens, is held
approximately fixed. No exact budget equality claim without measured times.
"""
import argparse,subprocess,sys,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--task",choices=["mqar_rewrite","mqar_alias"],required=True)
p.add_argument("--seed",type=int,required=True)
a=p.parse_args()
ROOT=Path(__file__).parent
COMMON=["--width","32","--layers","2","--heads","2","--L","96",
        "--batch","24","--pairs","12","--lpsX","1","--lpsC","16",
        "--dev","cpu","--threads","2","--task",a.task,"--lr","0.05",
        "--elr","0.04","--init","0.5","--alibi","0","--sink","0",
        "--update_eval","1","--update_heldout_seed","92741"]
ARMS=[
    ("g_base_2500",["--mix","g","--steps","2500"]),
    ("D_delta_2000",["--mix","D","--steps","2000"]),
]
out=[]
for name,extra in ARMS:
    tag=f"update_compute:{a.task}:s{a.seed}:{name}"
    cmd=[sys.executable,"rc.py",*COMMON,*extra,"--seed",str(a.seed),"--tag",tag]
    run=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=950)
    if run.returncode:
        print("FAILED",tag,run.stdout[-3000:],run.stderr[-4000:],flush=True)
        raise RuntimeError(tag)
    record=json.loads((ROOT/"results.jsonl").read_text().splitlines()[-1])
    assert record["tag"]==tag
    u=record["update_eval"]
    arr=u["autoregressive"]
    row={"task":a.task,"seed":a.seed,"arm":name,"steps":record["args"]["steps"],
         "train_s":record["train_s"],"val_loss":record["val"],
         "ar_changed":arr["query_B_changed"]["accuracy"],
         "ar_unchanged":arr["query_B_unchanged"]["accuracy"],
         "ar_queryA":arr["query_A"]["accuracy"],
         "ar_overall":arr["all_queries"]["accuracy"],
         "tf_changed":u["teacher_forced"]["query_B_changed"]["accuracy"],
         "stale_old":u.get("old_value_prediction_rate_changed",{}).get("autoregressive")}
    out.append(row)
    (ROOT/f"update_compute_{a.task}_{a.seed}.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(row),flush=True)
print("DONE",a.task,a.seed)

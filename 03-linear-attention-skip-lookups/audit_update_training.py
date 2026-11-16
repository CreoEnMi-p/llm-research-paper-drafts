"""Precommitted answer-memory update / alias falsification matrix.

2 tasks x 3 paired seeds x 4 arms. No hyperparameter search. Fresh heldout
rows are fixed across arms. Evaluate teacher forcing AND autoregressive
self-generated answers. This is NOT an official GDN2 benchmark.

Do not infer Gate failure if ordinary gated-linear cannot learn both changed
and unchanged Q2 categories (AR accuracy >=70% for each, in >=2/3 seeds).
If previous-read fails to improve update/new-key accuracy vs g in >=2/3
seeds, reject general-purpose usefulness of previous-read suppression.
Current-key equivalence rules out previous-read signal uniqueness.
"""
import argparse, json, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--task",choices=["mqar_rewrite","mqar_alias"],required=True)
p.add_argument("--seed",type=int,required=True)
a=p.parse_args()
ROOT=Path(__file__).parent
COMMON=[
    "--width","32","--layers","2","--heads","2",
    "--L","96","--batch","24","--steps","650","--pairs","12",
    "--lpsX","1","--lpsC","16","--dev","cpu","--threads","2",
    "--task",a.task,"--lr","0.05","--elr","0.04","--init","0.5",
    "--alibi","0","--sink","0",
    "--update_eval","1","--update_heldout_seed","92741",
]
ARMS=[
    ("g_base",["--mix","g"]),
    ("g_prev_margin",["--mix","g","--lps","0.25","--lpsI","1","--lpsM","0.2"]),
    ("g_key_margin",["--mix","g","--lps","0.25","--lpsI","1","--lpsR","2","--lpsM","0.2"]),
    ("D_delta",["--mix","D"]),
]
records=[]
for name,flags in ARMS:
    tag=f"memory_updates:{a.task}:seed{a.seed}:{name}"
    run=subprocess.run([sys.executable,"rc.py",*COMMON,*flags,
                        "--seed",str(a.seed),"--tag",tag],
                       cwd=ROOT,capture_output=True,text=True,timeout=900)
    if run.returncode:
        print("FAILED",tag,run.stdout[-4000:],run.stderr[-4500:],flush=True)
        raise RuntimeError(tag)
    rec=json.loads((ROOT/"results.jsonl").read_text().splitlines()[-1])
    assert rec["tag"]==tag
    u=rec["update_eval"]
    row={"task":a.task,"seed":a.seed,"arm":name,"train_s":rec["train_s"],
         "val_loss":rec["val"],"val_accuracy":rec["acc"],"update_eval":u}
    records.append(row)
    (ROOT/f"update_{a.task}_{a.seed}.json").write_text(json.dumps(records,indent=2))
    tf=u["teacher_forced"];ar=u["autoregressive"]
    print(json.dumps({
        "task":a.task,"seed":a.seed,"arm":name,"train_s":rec["train_s"],
        "heldout_seed":u["heldout_seed"],
        "AR_A":ar["query_A"]["accuracy"],
        "AR_changed":ar["query_B_changed"]["accuracy"],
        "AR_unchanged":ar["query_B_unchanged"]["accuracy"],
        "TF_A":tf["query_A"]["accuracy"],
        "TF_changed":tf["query_B_changed"]["accuracy"],
        "TF_unchanged":tf["query_B_unchanged"]["accuracy"],
        "AR_stale_old":u.get("old_value_prediction_rate_changed",{}).get("autoregressive"),
    }),flush=True)
print("FINISHED",a.task,a.seed,flush=True)

"""Fresh heldout AR falsification of Gemini's EXISTING margin=0.2 candidate.

Precommitted before new heldout results are observed:
- Fixed margin .2, gamma initial .25, no hyperparameter search.
- Three paired seeds, same task/model/training protocol except baseline
  gets both 300-step token-matched and 1500-step approximate CPU-wall-time arms.
- Eval split is new rng seed 91731, never used for train/periodic validation.
- An AR improvement claim for the margin previous-read variant REQUIRES
  beating the 1500-step g baseline in >=2/3 seeds, AND positive mean.
- No 'previous-read unique' claim if matched current-key is competitive.
- Even if this passes, still NO standalone method until real GDN2 and
  overwrite/alias tests; the sequential reference is not a fast kernel.
"""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
a=p.parse_args()
ROOT=Path(__file__).parent
MODEL=[
 "--width","32","--layers","2","--heads","2",
 "--L","64","--batch","32","--pairs","16",
 "--lpsX","1","--lpsC","64","--dev","cpu","--threads","2",
 "--task","mqar","--lr","0.05","--elr","0.04",
 "--init","0.5","--alibi","0","--sink","0",
 "--ar_eval","1","--ar_heldout_seed","91731",
]
ARMS=[
 ("g_300",["--steps","300","--mix","g"]),
 ("g_1500",["--steps","1500","--mix","g"]),
 ("prev_margin_300",["--steps","300","--mix","g","--lps","0.25","--lpsI","1","--lpsM","0.2"]),
 ("key_margin_300",["--steps","300","--mix","g","--lps","0.25","--lpsI","1","--lpsM","0.2","--lpsR","2"]),
]
out=[]
for name,extra in ARMS:
  tag=f"ar_margin_fresh:seed{a.seed}:{name}"
  proc=subprocess.run([sys.executable,"rc.py",*MODEL,"--seed",str(a.seed),
                       "--tag",tag,*extra],cwd=ROOT,
                      capture_output=True,text=True,timeout=900)
  if proc.returncode:
    print("FAILED",tag,proc.stdout[-3000:],proc.stderr[-5000:],flush=True)
    raise RuntimeError(tag)
  d=json.loads((ROOT/"results.jsonl").read_text().splitlines()[-1])
  assert d["tag"]==tag
  x=d["ar_eval"]["no_intervention"]
  iv=d["ar_eval"]["suppress_answer_writes"]
  rec=dict(seed=a.seed,arm=name,full_val_loss=d["val"],
           full_val_acc=d["acc"],fresh_tf_acc=x["tf_acc"],
           fresh_ar_acc=x["ar_acc"],fresh_ar_full_seq_acc=x["ar_full_sequence_acc"],
           fresh_ar_by_query=x["ar_by_query"],oracle_ar_acc=iv["ar_acc"] if iv else None,
           train_s=d["train_s"])
  out.append(rec)
  (ROOT/f"audit_ar_margin_seed{a.seed}.json").write_text(json.dumps(out,indent=2))
  print(json.dumps(rec),flush=True)
print("FRESH HOLDOUT COMPLETE; compare exact paired seeds and wall-time budget.",flush=True)

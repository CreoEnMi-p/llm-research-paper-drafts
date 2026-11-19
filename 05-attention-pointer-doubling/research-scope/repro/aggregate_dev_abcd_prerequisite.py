"""Aggregate AB/CD development-only prerequisite experiment; no AC data read."""
import json,statistics
from pathlib import Path

rows=[]
for mode in ("single","paired"):
  for model in ("transformer","pointer-depth"):
    for seed in range(5):
      path=Path(f"dev_{mode}_{model}_s{seed}.json")
      d=json.loads(path.read_text())
      cfg=d["config"]
      assert cfg["train_mode"]==mode and cfg["model"]==model and cfg["seed"]==seed
      assert d["no_AC_access"] is True and d["test_compositions"]==["AB","CD"]
      for comp in ("AB","CD"):
        assert d["dev"][comp]["n"]==64
        rows.append({"mode":mode,"model":model,"seed":seed,"composition":comp,
                     **{k:d["dev"][comp][k] for k in ("acc","both","flip","invariant")}})
out={"protocol":"AB/CD development only: unchanged models, 1000 steps, batch32, 5 seeds, source-only no oracle links, no AC read",
     "rows":rows,"summary":{},"seed_paired_deltas":{}}
for model in ("transformer","pointer-depth"):
  for comp in ("AB","CD"):
    for mode in ("single","paired"):
      group=[r for r in rows if r["model"]==model and r["composition"]==comp and r["mode"]==mode]
      out["summary"][f"{model}|{comp}|{mode}"]={k:{"mean":statistics.mean(r[k] for r in group),
                 "seed_values":[r[k] for r in group]} for k in ("acc","both","flip","invariant")}
    delta=[]
    for seed in range(5):
      a=next(r for r in rows if r["model"]==model and r["composition"]==comp and r["mode"]=="single" and r["seed"]==seed)
      b=next(r for r in rows if r["model"]==model and r["composition"]==comp and r["mode"]=="paired" and r["seed"]==seed)
      delta.append(b["both"]-a["both"])
    out["seed_paired_deltas"][f"{model}|{comp}"]={"deltas":delta,"mean":statistics.mean(delta)}
Path("dev_abcd_prerequisite_summary.json").write_text(json.dumps(out,indent=2))
for k,v in out["summary"].items():
  print("DEV_PREREQUISITE",k,
        "single",round(v["acc"]["mean"],4),"both",round(v["both"]["mean"],4),
        "flip",round(v["flip"]["mean"],4))
for k,v in out["seed_paired_deltas"].items():
  print("DEV_PAIR_DELTA",k,round(v["mean"],4),"per_seed",v["deltas"])

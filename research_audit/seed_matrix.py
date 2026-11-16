"""Independent reproducibility screen; uses patched temporary copies to vary train seeds.
The original repository files remain unchanged. This script needs only torch/numpy.
"""
import argparse, json, pathlib, re, subprocess, sys, tempfile

ROOT=pathlib.Path(__file__).resolve().parents[1]
P=argparse.ArgumentParser()
P.add_argument("--paper",choices=["03","05"],required=True)
P.add_argument("--seeds",type=int,default=4)
args=P.parse_args()
dirpath=ROOT/("03-linear-attention-skip-lookups" if args.paper=="03" else "05-attention-pointer-doubling")
original=(dirpath/("rc.py" if args.paper=="03" else "pdg.py")).read_text()
assert "torch.manual_seed(0)" in original
assert "rng = np.random.default_rng(0)" in original
all_rows=[]

def go(seed, label, flags):
    text=original.replace("torch.manual_seed(0)",f"torch.manual_seed({seed})")
    text=text.replace("rng = np.random.default_rng(0)",f"rng = np.random.default_rng({seed})")
    path=dirpath/f"_audit_seed_{seed}.py"
    path.write_text(text)
    tag=f"IA-{args.paper}-{seed}-{label}"
    cmd=[sys.executable,str(path),*flags.split(),"--tag",tag]
    p=subprocess.run(cmd,cwd=dirpath,capture_output=True,text=True,timeout=600)
    if p.returncode:
        print(json.dumps({"tag":tag,"error":p.stderr[-1300:],"out":p.stdout[-400:]}),flush=True)
        raise RuntimeError(f"Failed {tag}")
    rows=[json.loads(line) for line in (dirpath/"results.jsonl").read_text().splitlines() if line.strip()]
    r=next(row for row in reversed(rows) if row.get("tag")==tag)
    rec={"tag":tag,"seed":seed,"condition":label,
         "acc":r.get("acc"),"val":r.get("val"),
         "steps90":r.get("steps90"),"train_s":r.get("train_s")}
    print(json.dumps(rec),flush=True)
    all_rows.append(rec)
    path.unlink()

if args.paper=="03":
    common="--width 32 --layers 2 --heads 2 --L 64 --batch 32 --steps 300 --pairs 16 --lpsX 1 --lpsC 16 --dev cpu --threads 2 --task mqar --lr 0.06 --elr 0.04 --init 0.5 --alibi 0 --sink 0 --mix g"
    for seed in range(args.seeds):
        go(seed,"base",common)
        go(seed,"predicted",common+" --lps 0.25")
        go(seed,"content-shuffle",common+" --lps 0.25 --lpsc 1")
        go(seed,"key-read",common+" --lps 0.25 --lpsR 2")
    go(0,"empty-read",common+" --lps 0.25 --lpsC 64")
else:
    common="--task hopx --pairs 8 --batch 16 --dev cpu --threads 2 --init 0.5 --stop 0.99 --L 0 --nev 256"
    BLN="--sink 0 --alibi 0 --eqh 2 --eqt 3 --ptr 11 --pnb 3 --lr 0.02"
    PRD="--sink 0 --alibi 0 --eqh 2 --eqt 3 --ptr 6 --lr 0.02"
    for seed in range(args.seeds):
        for condition,rev in [("reverse",1),("forward",0),("random",2)]:
            go(seed,condition,common+f" --hops 7 --layers 3 --rev {rev} --steps 50 "+BLN)
        go(seed,"predecessor-forward8",common+" --hops 8 --layers 4 --rev 0 --steps 100 "+PRD)
(pathlib.Path.cwd()/f"audit_{args.paper}_seed_results.json").write_text(json.dumps(all_rows,indent=2))

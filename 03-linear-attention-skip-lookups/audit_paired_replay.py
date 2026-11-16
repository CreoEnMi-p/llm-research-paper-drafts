"""Within-one-run paired gate ablation, with full effective args and environment hashes.

This is the *replacement* for comparing margin trials across different Action jobs.
"""
import hashlib,json,os,platform,subprocess,sys
from pathlib import Path
import torch
ROOT=Path(__file__).parent
CODEFILES=("rc.py","mix.py","gate_exact.py")
sha={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in CODEFILES}
env={"torch":torch.__version__,"platform":platform.platform(),"python":sys.version,
     "git_sha":os.getenv("GITHUB_SHA"),"code_sha256":sha,
     "MKL_NUM_THREADS":os.environ.get("MKL_NUM_THREADS")}
print(json.dumps({"environment":env}),flush=True)
COMMON=["--width","32","--layers","2","--heads","2","--L","64",
        "--batch","32","--steps","300","--pairs","16","--lpsX","1",
        "--lpsC","64","--dev","cpu","--threads","1","--task","mqar",
        "--lr","0.05","--elr","0.04","--init","0.5","--alibi","0",
        "--sink","0","--mix","g","--lps","0.25","--lpsI","1"]
ARMS={
 "prev":[],
 "prev_margin":["--lpsM","0.2"],
 "key_margin":["--lpsM","0.2","--lpsR","2"],
 "prev_detach_margin":["--lpsM","0.2","--lpsD","1"],
}
out=[]
for seed in (0,1,2):
 for arm,flags in ARMS.items():
    tag=f"paired_replay:s{seed}:{arm}"
    cmd=[sys.executable,"rc.py",*COMMON,"--seed",str(seed),*flags,"--tag",tag]
    p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=900)
    if p.returncode:
       print("FAILED",tag,p.stdout[-1400:],p.stderr[-2500:],flush=True)
       raise RuntimeError(tag)
    d=json.loads((ROOT/"results.jsonl").read_text().splitlines()[-1])
    assert d["tag"]==tag
    rec=dict(seed=seed,arm=arm,loss=d["val"],acc=d["acc"],
             steps90=d["steps90"],curve=d["curve"],train_s=d["train_s"],
             effective_args=d["args"],S_l=d.get("S_l"),gst=d.get("gst"))
    out.append(rec)
    (ROOT/"audit_paired_replay.json").write_text(json.dumps({"environment":env,"runs":out},indent=2))
    print(json.dumps({k:v for k,v in rec.items() if k not in ("effective_args","curve")}),flush=True)
print("PAIRED REPLAY COMPLETE",flush=True)

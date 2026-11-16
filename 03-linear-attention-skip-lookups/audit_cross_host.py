"""Cross-GitHub-run audit of deterministic seeded C64 margin training.

Each GitHub Actions matrix worker is independent; this script executes the
same two seeds and prints hardware/PyTorch fingerprint to detect host-dependent
numerical drift. It does not change the underlying training algorithm.
"""
import hashlib,json,os,platform,subprocess,sys
from pathlib import Path
import torch,numpy
ROOT=Path(__file__).parent
cpuinfo=Path("/proc/cpuinfo")
cpulines=cpuinfo.read_text().splitlines() if cpuinfo.exists() else []
def val(name):
  found=[x.partition(":")[2].strip() for x in cpulines if x.lower().startswith(name+":")]
  return list(dict.fromkeys(found))[:4]
hw={"platform":platform.platform(),"machine":platform.machine(),
    "cpu_models":val("model name"),"cpu_vendors":val("vendor_id"),
    "torch":torch.__version__,"numpy":numpy.__version__,
    "torch_cpu_capability":torch._C._get_cpu_capability(),
    "omp_num_threads":os.getenv("OMP_NUM_THREADS"),
    "mkl_num_threads":os.getenv("MKL_NUM_THREADS"),
    "GITHUB_SHA":os.getenv("GITHUB_SHA"),
    "worker":os.getenv("AUDIT_WORKER")}
print("HOST",json.dumps(hw),flush=True)
BASE=["--width","32","--layers","2","--heads","2",
"--L","64","--batch","32","--steps","300","--pairs","16",
"--lpsX","1","--lpsC","64","--dev","cpu","--threads","2",
"--task","mqar","--lr","0.05","--elr","0.04","--init","0.5",
"--alibi","0","--sink","0","--mix","g","--lps","0.25","--lpsI","1","--lpsM","0.2"]
out=[]
for seed in (1,2):
  tag=f"cross_host:{os.getenv('AUDIT_WORKER','0')}:s{seed}"
  cmd=[sys.executable,"rc.py",*BASE,"--seed",str(seed),"--tag",tag]
  proc=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=600)
  if proc.returncode: print("FAILED",tag,proc.stderr[-3000:],flush=True);raise RuntimeError(tag)
  row=json.loads((ROOT/"results.jsonl").read_text().splitlines()[-1])
  assert row["tag"]==tag
  result={"seed":seed,"loss":row["val"],"acc":row["acc"],"curve":row["curve"],
          "S_l":row.get("S_l"),"train_s":row["train_s"]}
  out.append(result)
  print("REPLAY",json.dumps(result),flush=True)
(ROOT/"audit_cross_host.json").write_text(json.dumps({"hw":hw,"results":out},indent=2))
print("CROSS HOST COMPLETE",flush=True)

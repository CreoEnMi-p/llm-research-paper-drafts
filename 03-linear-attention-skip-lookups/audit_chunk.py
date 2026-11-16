"""Same trained model, multiple computational chunk sizes.

Predefined expected invariance: baseline recurrence max logit delta <1e-3
versus trained chunk size 16. A learned predicted-skip should be chunk
invariant if chunking were solely a backend execution detail.
These experiments compare model outputs at identical checkpoints.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
COMMON = ["--width", "32", "--layers", "2", "--heads", "2",
    "--L", "64", "--batch", "32", "--steps", "300", "--pairs", "16",
    "--lpsX", "1", "--lpsC", "16", "--dev", "cpu",
    "--threads", "2", "--task", "mqar", "--lr", "0.04",
    "--elr", "0.04", "--init", "0.5", "--alibi", "0", "--sink", "0",
    "--chunk_eval", "1"]
ARM = {
    "g-base": ["--mix", "g"],
    "g-ps": ["--mix", "g", "--lps", "0.25"],
    "D-base": ["--mix", "D"],
    "D-ps": ["--mix", "D", "--lps", "0.25"]
}
results=[]
for seed in range(3):
    for arm, flags in ARM.items():
        tag=f"chunk:seed{seed}:{arm}"
        cmd=[sys.executable,"rc.py",*COMMON,"--seed",str(seed),"--tag",tag,*flags]
        p=subprocess.run(cmd,cwd=HERE,capture_output=True,text=True,timeout=600)
        if p.returncode:
            print(tag,p.stdout[-6000:],p.stderr[-6000:],flush=True)
            raise RuntimeError(tag)
        row=json.loads((HERE/"results.jsonl").read_text().splitlines()[-1])
        assert row["tag"]==tag
        out={"seed":seed,"arm":arm,"training_loss_C16":row["val"],
             "training_acc_C16":row["acc"],"chunks":row["chunk_eval"]}
        results.append(out)
        print(json.dumps(out),flush=True)
(HERE/"audit_chunk_summary.json").write_text(json.dumps(results,indent=2))
print("DONE", len(results), "frozen checkpoints",flush=True)

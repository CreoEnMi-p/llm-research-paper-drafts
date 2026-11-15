"""Grid driver: reads 'tag | flags' lines from stdin, runs each one at a time in the foreground (nice -n 20), prints one line per run.
Defaults: MQAR P16, L64, 2 layers, width 32, 2 heads, 300 steps, batch 32, exact chunked recurrence (chunk 16), CPU.
'lm' in the flags runs lm.py (text bed) instead of rc.py."""
import json, subprocess, sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
RC = "--width 32 --layers 2 --heads 2 --L 64 --batch 32 --steps 300 --pairs 16 --lpsX 1 --lpsC 16 --dev cpu --threads 2 --task mqar --lr 0.06 --elr 0.04 --init 0.5 --alibi 0 --sink 0"
LM = "--width 32 --layers 2 --heads 2 --L 64 --batch 64 --steps 300 --lpsX 1 --lpsC 16 --dev cpu --threads 2 --lr 0.04 --elr 0.04 --init 0.25 --alibi 0 --sink 0"
from concurrent.futures import ThreadPoolExecutor
J = int(os.environ.get("J", "2"))
def one(line):
    tag, fl = [z.strip() for z in line.split("|", 1)]
    lm = fl.startswith("lm ")
    if lm: fl = fl[3:]
    cmd = f"nice -n 20 python3 {'lm.py' if lm else 'rc.py'} {LM if lm else RC} {fl} --tag G:{tag}"
    out = subprocess.run(cmd, shell=True, cwd=HERE, capture_output=True, text=True)
    if out.returncode: print(tag, "FAILED rc", out.returncode, out.stderr[-300:], out.stdout[-200:], flush=True); return
    rows = [json.loads(l) for l in open(os.path.join(HERE, "results.jsonl")) if f'"G:{tag}"' in l]
    rows = [x for x in rows if x["tag"] == "G:" + tag]
    report(tag, rows[-1])
def report(tag, r):
    g = r.get("gst", {}); gs = " ".join(f"L{k}:" + ",".join(f"{n}={v[n]:.2f}" for n in ("pv", "qv", "qk", "all", "h1", "h2") if n in v) for k, v in g.items())
    gr = r.get("gst_rep", {}); gs += (" rep " + " ".join(f"L{k}:h1={v['h1']:.2f},h2={v['h2']:.2f}" for k, v in gr.items())) if gr else ""
    gam = " gam " + "/".join(str(v.get("gam")) for v in g.values()) if g else ""
    ex = f" rep {r['val_rep']:.4f}" if "val_rep" in r else ""
    acc = f" acc {r['acc']:.3f}" if r.get("acc") is not None else ""
    print(f"{tag:28s} val {r['val']:.4f}{ex}{acc} {r['train_s']:.0f}s | {gs}{gam}", flush=True)
lines = [l.strip() for l in sys.stdin if l.strip() and not l.strip().startswith("#")]
with ThreadPoolExecutor(J) as ex: list(ex.map(one, lines))

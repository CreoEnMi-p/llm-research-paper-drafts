"""Paired-seed audit of answer-token interventions.
Runs against the checked-out rc.py / mix.py. No claim of statistical significance
from two seeds; this checks the code-level oracle definition and produces a
grounded small empirical result. Writes only to local audit_oracle_summary.json.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
COMMON = [
    "--width", "32", "--layers", "2", "--heads", "2",
    "--L", "64", "--batch", "32", "--steps", "300",
    "--pairs", "16", "--lpsX", "1", "--lpsC", "16",
    "--dev", "cpu", "--threads", "2", "--task", "mqar",
    "--lr", "0.04", "--elr", "0.04", "--init", "0.5",
    "--alibi", "0", "--sink", "0",
]
ARMS = [
    ("g-base", ["--mix", "g"]),
    ("g-predicted-skip", ["--mix", "g", "--lps", "0.25"]),
    ("D-base", ["--mix", "D"]),
    ("D-write-only-oracle", ["--mix", "D", "--orq", "0"]),
    ("D-full-update-oracle", ["--mix", "D", "--orq", "0", "--orqE", "1"]),
    ("D-predicted-skip", ["--mix", "D", "--lps", "0.25"]),
]
results = []
for seed in (0, 1):
    for arm, flags in ARMS:
        tag = f"audit:seed{seed}:{arm}"
        command = [sys.executable, "rc.py", *COMMON, "--seed", str(seed), "--tag", tag, *flags]
        completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, timeout=360)
        if completed.returncode:
            print("ERROR", tag, completed.stdout[-3000:], completed.stderr[-3000:], flush=True)
            raise RuntimeError(tag)
        with (ROOT / "results.jsonl").open() as f:
            last = f.readlines()[-1]
        result = json.loads(last)
        if result["tag"] != tag:
            raise AssertionError(f"Wrong output tag: {result['tag']} != {tag}")
        item = {"seed": seed, "arm": arm, "loss": result["val"],
                "accuracy": result["acc"], "train_seconds": result["train_s"]}
        results.append(item)
        print(json.dumps(item), flush=True)
output = ROOT / "audit_oracle_summary.json"
output.write_text(json.dumps(results, indent=2))
print("RESULTS:")
for item in results:
    print(f"{item['seed']:>4} {item['arm']:<23} {item['loss']:.5f} {item['accuracy']:.5f}")
print("Saved", output.name)

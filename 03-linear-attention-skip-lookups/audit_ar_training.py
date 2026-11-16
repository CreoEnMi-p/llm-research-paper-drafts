"""Precommitted 3-seed MQAR autoregressive exposure diagnostic.

Independent from Gemini's C64/P4/mqar2/hop2/margin matrix.
Research stop rule: if self-consistent previous-read gate fails to improve
autoregressive answer accuracy vs gated-linear baseline in >=2/3 seeds, STOP
the claim that the trained gate helps online retrieval. 3 seeds remain pilot
evidence regardless of outcome. Previous-read uniqueness is rejected as a
claim if current-key gate is competitive. Softmax provides a sanity baseline,
not compute-matched architecture comparison.

Run: python audit_ar_training.py --seed 0
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

P = argparse.ArgumentParser()
P.add_argument("--seed", type=int, required=True)
args = P.parse_args()
ROOT = Path(__file__).parent

COMMON = [
    "--width", "32", "--layers", "2", "--heads", "2",
    "--L", "64", "--batch", "32", "--steps", "300", "--pairs", "16",
    "--lpsX", "1", "--lpsC", "16", "--dev", "cpu", "--threads", "2",
    "--task", "mqar", "--lr", "0.05", "--elr", "0.04",
    "--init", "0.5", "--alibi", "0", "--sink", "0",
    "--ar_eval", "1",
]
ARMS = [
    ("g_base", ["--mix", "g"]),
    ("g_exact_prev", ["--mix", "g", "--lps", "0.25", "--lpsI", "1"]),
    ("g_exact_key", ["--mix", "g", "--lps", "0.25", "--lpsI", "1", "--lpsR", "2"]),
    ("softmax_reference", ["--mix", "s"]),
]

records = []
for name, flags in ARMS:
    tag = f"ar_exposure:seed{args.seed}:{name}"
    cmd = [sys.executable, "rc.py", *COMMON, *flags,
           "--seed", str(args.seed), "--tag", tag]
    run = subprocess.run(cmd, cwd=ROOT, capture_output=True,
                         text=True, timeout=900)
    if run.returncode:
        print("FAILED", tag, run.stdout[-3500:], run.stderr[-4500:], flush=True)
        raise RuntimeError(tag)
    rec = json.loads((ROOT / "results.jsonl").read_text().splitlines()[-1])
    assert rec["tag"] == tag
    ar = rec["ar_eval"]
    no = ar["no_intervention"]
    oracle = ar["suppress_answer_writes"]
    row = {
        "seed": args.seed, "arm": name,
        "teacher_forced_full_val_loss": rec["val"],
        "teacher_forced_full_val_accuracy": rec["acc"],
        "teacher_forced_96rows_accuracy": no["tf_acc"],
        "autoregressive_96rows_accuracy": no["ar_acc"],
        "autoregressive_full_row_accuracy": no["ar_full_sequence_acc"],
        "autoregressive_by_query": no["ar_by_query"],
        "oracle_posthoc_ar_accuracy": oracle["ar_acc"] if oracle else None,
        "oracle_posthoc_tf_accuracy": oracle["tf_acc"] if oracle else None,
        "train_seconds": rec["train_s"],
    }
    records.append(row)
    (ROOT / f"audit_ar_seed{args.seed}.json").write_text(
        json.dumps(records, indent=2))
    print(json.dumps(row), flush=True)
print("PILOT COMPLETE. Interpret with paired seeds, not individual losses.", flush=True)

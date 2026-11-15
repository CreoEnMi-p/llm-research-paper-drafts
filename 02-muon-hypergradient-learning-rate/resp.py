"""Controller response from the 10x-high start of each safe-recipe run: steps per tenfold fall of the step size
between its peak and the hold step, the adaptation window (hold - warmup end), and the held rate against the tuned constant.
usage: python3 resp.py"""
import json, math
SETS = [  # tag of high-start safe run, its start lr, tuned constant rate, tuned constant val
    ("r37-b32-gs-f68", 0.15, 0.01, 3.3590), ("r37-b16-gs-f135", 0.1, 0.005, 3.3455), ("r36-main-gs-freeze270", 0.04, 0.005, 3.3534),
    ("r38-L6-safe", 0.025, 0.0025, 3.3489), ("r38-L8-safe", 0.04, 0.005, 3.3672), ("r38-V6k-safe", 0.05, 0.005, 5.1214),
    ("r38-V24k-safe", 0.05, 0.005, 6.6376), ("r36-tpw3k-gs-freeze900", 0.03, 0.0025, 3.3289), ("r37-tpw80-gs-f450", 0.03, 0.0025, 3.4353),
    ("r38-tpw118-safe", 0.025, 0.0025, 3.4180), ("r38-w128-safe", 0.0125, 0.00125, 3.2956), ("r38-w128-safelin@750", 0.0125, 0.0025, 3.2729),
]
R = [json.loads(l) for l in open("results.jsonl")]
print(f"{'run':24s} steps hold window peak@step  held/tuned  steps/decade  window/decade  gap")
for tag, lr, tuned, cv in SETS:
    r = [x for x in R if x["tag"] == tag and x["args"]["lr"] == lr][-1]; a = r["args"]
    sched = a["sched"]; st = a["steps"]; w = max(1, int(a["warm"] * st)); hold = a["freeze_at"]
    def rate(step, s):
        sc = min((step + 1) / w, 1.0) if sched != "wsd" or step < st * (1 - a["cool"]) else 1.0
        if sched == "lin": sc = min((step + 1) / w, 1 - max(0, (step - w) / (st - w)))
        return a["lr"] * s * sc
    tr = [(x[0], rate(x[0], x[1]["b0.q"][0])) for x in r["slog"] if x[0] <= hold]
    pk = max(tr, key=lambda t: t[1]); hd = tr[-1]
    fall = pk[1] / hd[1]; dec = (hd[0] - pk[0]) * math.log(10) / math.log(fall) if fall > 1.05 else float("inf")
    print(f"{tag:24s} {st:5d} {hold:4d} {hold - w:6d} {pk[1]:.4f}@{pk[0]:<4d} {hd[1] / tuned:9.2f} {dec:12.0f} {(hold - w) / dec:13.2f}  {r['scores']['val'] - cv:+.4f}")

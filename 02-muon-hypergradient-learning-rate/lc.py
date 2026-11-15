"""Early loss excess of rule runs over the paired constant run (same data order and init).
Usage: python3 lc.py <tag-prefix> [const-tag]   prints max excess of the train-loss EMA in steps 0-300, and the step."""
import json, sys
R = [json.loads(l) for l in open("results.jsonl")]
ct = sys.argv[2] if len(sys.argv) > 2 else "r31-long-const-lc"
c = [r for r in R if r["tag"] == ct][-1]["scores"]["lcurve"]
for r in R:
    if r["tag"].startswith(sys.argv[1]) and "lcurve" in r["scores"]:
        lc = r["scores"]["lcurve"]; ex = [(x - y, 10 * i) for i, (x, y) in enumerate(zip(lc, c)) if i <= 30]
        mx = max(ex); early = sum(e for e, _ in ex[:16]) / 16
        print(f'{r["tag"]:32s} lr {r["args"]["lr"]:<7g} elr {r["args"]["elr"]:<6g} val {r["scores"]["val"]:.4f} peak {r["scores"]["peak_rate"]:.4f} max excess {mx[0]:+.3f} at step {mx[1]}  mean excess steps 0-150 {early:+.3f}')

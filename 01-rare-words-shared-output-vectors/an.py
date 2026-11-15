"""Per-bin comparison: python3 an.py tagA tagB [tagC ...]  (gain = A - X, in nats per target, and contribution = gain * share)"""
import json, sys
R = {}
for l in open(__import__("os").path.join(__import__("os").path.dirname(__file__), "grid.jsonl")):
    r = json.loads(l); R[r["tag"]] = r
A = R[sys.argv[1]]
print("bin".ljust(9), "share".rjust(6), A["tag"][:14].rjust(14), *[t[:16].rjust(16) for t in sys.argv[2:]], "  (gain, contribution)")
for b, (l, sh) in A["bins"].items():
    row = [b.ljust(9), f"{sh:6.3f}", f"{l:14.3f}"]
    for t in sys.argv[2:]:
        lb = R[t]["bins"][b][0]; row.append(f"{A['bins'][b][0]-lb:+7.3f} {sh*(A['bins'][b][0]-lb):+7.4f}".rjust(16))
    print(*row)
print("total".ljust(9), " " * 6, f"{A['scores']['val']:14.4f}", *[f"{A['scores']['val']-R[t]['scores']['val']:+16.4f}" for t in sys.argv[2:]])

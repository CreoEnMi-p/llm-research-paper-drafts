"""Geometric-mean step size (lr * s) the rule used between the end of warmup and the start of cooldown.
Usage: python3 wrate.py <tag-prefix>"""
import json, math, sys
for l in open("results.jsonl"):
    r = json.loads(l)
    if not r["tag"].startswith(sys.argv[1]) or r["args"]["rule"] == "none": continue
    a = r["args"]; T = a["steps"]; w0, c0 = 0.05 * T, 0.7 * T
    sc = [next(iter(d.values()))[0] for st, d in r["slog"] if w0 <= st < c0]
    g = math.exp(sum(math.log(a["lr"] * s) for s in sc) / len(sc))
    print(f'{r["tag"]:24s} start {a["lr"]:<7} val {r["scores"]["val"]:.4f} mean flat rate {g:.4f} (n={len(sc)})')

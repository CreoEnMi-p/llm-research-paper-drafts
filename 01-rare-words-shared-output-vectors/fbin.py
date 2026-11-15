"""Per-frequency-bin gain of Y over X: python3 fbin.py X Y cell ..."""
import json, os, sys
R = {json.loads(l)["tag"]: json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid.jsonl"))}
X, Y = sys.argv[1:3]; FN = ["<1e-6", "1-3e-6", "3e-6-1e-5", "1-3e-5", "3e-5-1e-4", "1-3e-4", "3e-4-1e-3", ">1e-3", "unk"]
print("cell".ljust(16), *[b.rjust(14) for b in FN])
for c in sys.argv[3:]:
    x, y = R[f"{c}_{X}"], R[f"{c}_{Y}"]
    if "fbins" not in x or "fbins" not in y: print(c, "no fbins"); continue
    print(c.ljust(16), *[(f"{x['fbins'][b][0]-y['fbins'][b][0]:+.3f}/{x['fbins'][b][1]:.3f}" if b in x["fbins"] else "-").rjust(14) for b in FN])

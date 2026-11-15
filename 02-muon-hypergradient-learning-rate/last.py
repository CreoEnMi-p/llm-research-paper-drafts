"""Print val, peak rate and final Adam scale for runs whose tag starts with a prefix.
Usage: python3 last.py <tag-prefix>"""
import json, sys
for l in open("results.jsonl"):
    r = json.loads(l)
    if not r["tag"].startswith(sys.argv[1]): continue
    a, s = r["args"], r["scores"]
    ad = s.get("adam_cos", {})
    print(f'{r["tag"]:28s} lr {a["lr"]:<7} elr {a.get("elr")!s:<6} val {s["val"]:.4f} peak {s.get("peak_rate", 0):.4f} final_s {s.get("final_scale", "")} adam {ad.get("scale", "")}')

"""Print grid rows: python3 tab.py PREFIX  -> tag, val, acc, train_s, layer-0 gate at query values / pair values."""
import json, sys
pre = sys.argv[1] if len(sys.argv) > 1 else "G:"
for l in open("results.jsonl"):
    r = json.loads(l)
    if not r["tag"].startswith(pre): continue
    g = r.get("gst", {}).get("0", {})
    print(f"{r['tag'][2:]:32s} {r['val']:.4f} {r.get('acc') or 0:.3f} {r['train_s']:5.0f}s  qv {g.get('qv', float('nan')):.2f} pv {g.get('pv', float('nan')):.2f}" + (f" rep {r['val_rep']:.4f}" if 'val_rep' in r else ""))

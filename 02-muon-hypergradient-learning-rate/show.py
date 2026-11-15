"""Print the step-size trajectory (geometric mean over matrices of lr*scale) and the mean cosine for the last N runs."""
import json, sys, math, numpy as np
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
rs = [json.loads(l) for l in open('results.jsonl')][-n:]
for r in rs:
    a = r['args']; print(f"{r['tag']} lr0={a['lr']} val={r['scores']['val']}")
    out = []
    for step, sc in r['slog'][::2]:
        gm = math.exp(np.mean([math.log(max(v[0], 1e-9)) for v in sc.values()])); c = np.mean([v[1] for v in sc.values()])
        out.append(f"{step}:{a['lr']*gm:.4f}/{c:.2f}")
    print("  ", " ".join(out))

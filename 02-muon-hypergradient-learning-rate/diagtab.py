"""For each tag prefix: val per learning rate and the mean cosine of each signal (over matrices, flat phase)."""
import json, sys, numpy as np
pre = sys.argv[1]
rs = [json.loads(l) for l in open('results.jsonl')]
rs = [r for r in rs if r['tag'].startswith(pre) and 'diag' in r['scores']]
tags = sorted(set(r['tag'] for r in rs))
for t in tags:
    print(t)
    for r in sorted([r for r in rs if r['tag'] == t], key=lambda r: r['args']['lr']):
        d = np.array(list(r['scores']['diag'].values()))
        if d.shape[1] < 6: continue
        print(f"  lr {r['args']['lr']:<8} val {r['scores']['val']:.4f} t {r['train_s']:5.1f}s  upd {d[:,0].mean():.3f} grad {d[:,1].mean():.3f} mom {d[:,2].mean():.3f} hyper {d[:,3].mean():.4f} path {d[:,4].mean():.4f} path100 {d[:,5].mean():.4f}" + (f" disp {d[:,6].mean():.4f}" if d.shape[1] > 6 else ""))

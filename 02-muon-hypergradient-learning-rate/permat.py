"""Which matrices want which rates.
(1) From constant runs with --diag: per matrix, the rate where the hyper cosine crosses zero (log-linear interpolation).
(2) From a per-matrix rule run: each matrix's final scale relative to the geometric mean."""
import json, sys, math, numpy as np, collections
rs = [json.loads(l) for l in open('results.jsonl')]
def zeros(tag):
    runs = sorted([r for r in rs if r['tag'] == tag and 'diag' in r['scores']], key=lambda r: r['args']['lr'])
    names = list(runs[0]['scores']['diag'].keys()); out = {}
    for n in names:
        xs = [math.log(r['args']['lr']) for r in runs]; ys = [r['scores']['diag'][n][3] for r in runs]
        z = None
        for i in range(len(xs) - 1):
            if ys[i] >= 0 > ys[i + 1]: z = math.exp(xs[i] + (xs[i + 1] - xs[i]) * ys[i] / (ys[i] - ys[i + 1])); break
        if z is None: z = runs[0]['args']['lr'] if ys[0] < 0 else runs[-1]['args']['lr'] * 2
        out[n] = z
    return out
def kind(n): return n.split('.')[-1]
def finals(tag):
    out = []
    for r in rs:
        if r['tag'] != tag: continue
        last = r['slog'][-1][1] if r.get('slog') else None
        # last logged step before cooldown with adaptation still on
        sl = [s for s in r['slog'] if s[0] <= int(0.7 * r['args']['steps'])]
        sc = {k: v[0] for k, v in sl[-1][1].items()}
        g = math.exp(np.mean([math.log(v) for v in sc.values()]))
        out.append((r['args']['lr'], {k: v / g for k, v in sc.items()}))
    return out
if __name__ == "__main__":
    for tag in sys.argv[1:]:
        if any(r['tag'] == tag and 'diag' in r['scores'] for r in rs):
            z = zeros(tag); by = collections.defaultdict(list)
            for n, v in z.items(): by[kind(n)].append(v)
            print(tag, 'hyper-zero rate by matrix kind:', {k: f"{math.exp(np.mean(np.log(v))):.4f}" for k, v in by.items()})
            print('  by name:', {n: f"{v:.4f}" for n, v in z.items()})
        else:
            for lr, f in finals(tag):
                by = collections.defaultdict(list)
                for n, v in f.items(): by[kind(n)].append(v)
                print(tag, lr, {k: f"{math.exp(np.mean(np.log(v))):.2f}" for k, v in by.items()})
                print('  by name:', {n: f"{v:.2f}" for n, v in f.items()})

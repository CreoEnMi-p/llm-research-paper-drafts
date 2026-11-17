"""Build a transparently instrumented copy of original pdg.py for fair baselines.

No changes to original code. Supports NoPE and tied-layer (looped) ablations,
multiple seeds and per-eval loss/accuracy curves. The tied-layer implementation
reuses all block weights and Q/K/RMS parameters. Looped conditions intentionally
use no ALiBi/sinks and have no extra per-layer bias. Report effective depth and
actual param count / training time, not just steps.
"""
from pathlib import Path

src = (Path(__file__).resolve().parents[1] / "pdg.py").read_text()

def sub(old, new, n=1):
    global src
    found = src.count(old)
    assert found == n, f"Expected {n}, found {found} for {old[:110]!r}"
    src = src.replace(old, new, n)

sub('p.add_argument("--nev", type=int, default=512, help="eval rows")',
    'p.add_argument("--seed", type=int, default=0)\n'
    'p.add_argument("--nope", type=int, default=0, help="NoPE attention, no RoPE on ordinary heads")\n'
    'p.add_argument("--loop", type=int, default=0, help="share one transformer block across all virtual layers")\n'
    'p.add_argument("--nev", type=int, default=512, help="eval rows")')
sub('torch.set_num_threads(a.threads); torch.manual_seed(0)',
    'torch.set_num_threads(a.threads); torch.manual_seed(a.seed)')
sub('rng = np.random.default_rng(0); curve = []; s90 = None',
    'rng = np.random.default_rng(a.seed); curve = []; full_curve = []; s90 = None')
sub('s.n1, s.n2 = nn.Parameter(torch.ones(nl, d)), nn.Parameter(torch.ones(nl, d))',
    's.n1, s.n2 = nn.Parameter(torch.ones(1 if a.loop else nl, d)), nn.Parameter(torch.ones(1 if a.loop else nl, d))')
sub('s.qn, s.kn = nn.Parameter(torch.full((nl, s.dk), a.qkg)), nn.Parameter(torch.full((nl, s.dk), a.qkg))',
    's.qn, s.kn = nn.Parameter(torch.full((1 if a.loop else nl, s.dk), a.qkg)), nn.Parameter(torch.full((1 if a.loop else nl, s.dk), a.qkg))')
sub('w = torch.randn(nl, o, i) * i ** -0.5 * a.init',
    'w = torch.randn(1 if a.loop else nl, o, i) * i ** -0.5 * a.init')
sub('        for i in range(s.nl):\n            h = F.rms_norm(x, (d,), s.n1[i])',
    '        for i in range(s.nl):\n            li = 0 if a.loop else i\n            h = F.rms_norm(x, (d,), s.n1[li])')
for name in ['q','k','v','o','gate','up','down']:
    sub('P["'+name+'"][i]', 'P["'+name+'"][li]', src.count('P["'+name+'"][i]'))
for name in ['n1','n2','qn','kn']:
    sub('s.'+name+'[i]', 's.'+name+'[li]', src.count('s.'+name+'[i]'))
sub('s.kraw = k; q, k = rope(q, s.freq), rope(k, s.freq)',
    's.kraw = k\n            if not a.nope: q, k = rope(q, s.freq), rope(k, s.freq)')
sub('    def body_params(s): return s.nl * sum(o * i for o, i in s.shape.values())',
    '    def body_params(s): return (1 if a.loop else s.nl) * sum(o * i for o, i in s.shape.values())')
sub('            l, acc = evaluate(m); curve.append(round(acc, 3))',
    '            l, acc = evaluate(m); curve.append(round(acc, 3))\n'
    '            full_curve.append({"step":step+1,"acc":acc,"val_loss":l,"train_loss":float(loss.detach().cpu())})')
sub('    res["steps_run"] = step + 1',
    '    res["steps_run"] = step + 1\n'
    '    res["full_curve"] = full_curve\n'
    '    res["total_params"] = sum(t.numel() for t in m.parameters())\n'
    '    res["steps_per_second"] = (step + 1) / max(1e-6, res["train_s"])\n'
    '    res["seed"] = a.seed')
out=Path(__file__).resolve().parents[1]/"pdg_fair_generated.py"
out.write_text(src)
print(f"wrote {out.name} ({len(src)} bytes)")

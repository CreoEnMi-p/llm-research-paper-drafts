"""Scaling-grid test bed for cluster-shared output rows (study session, 2026-11-15). Copied from tb.py.

Model: pre-norm RMSNorm, RoPE, QK-norm, SwiGLU, no biases, untied head by default.
Muon on block matrices, Adam on embedding, head, norms. Data: FineWeb GPT-2 tokens mapped to a V-word vocabulary
plus one unknown token, each token read once.

Arms:
  base                       plain head (tune --hlr)
  --clus K                   head[w] = own[w] + C[c(w)], k-means on head rows every --recl steps (function kept)
  --heps e / --hb2 b         head-only Adam epsilon / beta2 (frequency-aware row steps, competitor)
  --cls K                    class-based softmax p(c|h) p(w|c,h) (Goodman 2001; Mikolov 2011), --clsby freq|kmeans
  --asm div                  adaptive softmax (Grave 2017), cutoffs --cuts (fractions of V), tail width d/div^i
  --coupled 1                Coupled Adam on the head
Data knob:
  --tail r                   vocabulary = top V/2 words + V/2 words spread evenly over frequency ranks V/2 .. r*V
                             (r=1: natural top-V vocabulary; larger r: rarer tail at the same V)
Probes (printed and saved): val with C zeroed, val with C reset to its last re-cluster value,
  loss per bin of training occurrences of the target word.

usage: nice -n 20 python3 grid.py --V 6000 --steps 400 --clus 64 --tag name
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "llm-per-word-weights", "data"); EOT = 50256

p = argparse.ArgumentParser()
p.add_argument("--width", type=int, default=32); p.add_argument("--layers", type=int, default=2); p.add_argument("--heads", type=int, default=2)
p.add_argument("--V", type=int, default=2000); p.add_argument("--L", type=int, default=64); p.add_argument("--batch", type=int, default=32)
p.add_argument("--steps", type=int, default=400); p.add_argument("--lr", type=float, default=0.02); p.add_argument("--elr", type=float, default=0.05)
p.add_argument("--hlr", type=float, default=0.1)
p.add_argument("--mb", type=float, default=0.95); p.add_argument("--b1", type=float, default=0.9); p.add_argument("--b2", type=float, default=0.95)
p.add_argument("--eps", type=float, default=1e-8); p.add_argument("--escale", type=float, default=4.0)
p.add_argument("--heps", type=float, default=-1); p.add_argument("--hb2", type=float, default=-1)
p.add_argument("--cool", type=float, default=0.3)
p.add_argument("--tie", type=int, default=0)
p.add_argument("--clus", type=int, default=0); p.add_argument("--cwhere", default="head")
p.add_argument("--recl", type=int, default=50); p.add_argument("--cinit", default="kmeans", help="kmeans / rand / freq")
p.add_argument("--clr", type=float, default=0.02); p.add_argument("--cstop", type=float, default=1.0)
p.add_argument("--cls", type=int, default=0); p.add_argument("--clsby", default="freq")
p.add_argument("--asm", type=float, default=0); p.add_argument("--cuts", default="0.1,0.4")
p.add_argument("--coupled", type=int, default=0)
p.add_argument("--hinit", type=float, default=1.0)
p.add_argument("--vemb", default=""); p.add_argument("--smear", type=int, default=0)
p.add_argument("--tail", type=float, default=1.0); p.add_argument("--acc", type=int, default=1, help="split each batch into this many micro-batches (memory only)"); p.add_argument("--shrink", default="", help="post-hoc: shrink rows to k-means centroid by n0/(n0+count), list of n0")
p.add_argument("--nval", type=int, default=600); p.add_argument("--threads", type=int, default=1); p.add_argument("--dev", default="mps")
p.add_argument("--save", default="", help="save final head rows / assignments to this .pt path")
p.add_argument("--tag", default=""); p.add_argument("--out", default="grid.jsonl")
a = p.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(0); D = torch.device(a.dev)

counts = np.load(os.path.join(DATA, "counts.npy")).astype(np.float64)
rank = np.argsort(-counts, kind="stable")
if a.tail <= 1.0:
    top = rank[:a.V - 2]
else:
    h = (a.V - 2) // 2; tail_ranks = np.linspace(h, int(a.tail * a.V), a.V - 2 - h).astype(np.int64)
    top = np.concatenate([rank[:h], rank[np.unique(tail_ranks)]])
keep = np.unique(np.concatenate([top, [EOT]])); V = len(keep) + 1; UNK = V - 1
look = np.full(65536, UNK, np.int64); look[keep] = np.arange(len(keep))
freq = np.append(counts[keep], counts.sum() - counts[keep].sum())

need = a.steps * a.batch * a.L
tr = np.memmap(os.path.join(DATA, "fineweb_train_000001.bin"), dtype=np.uint16, mode="r", offset=1024)
rows = look[np.asarray(tr[:need]).astype(np.int64).reshape(-1, a.L)]
va = np.memmap(os.path.join(DATA, "fineweb_val_000000.bin"), dtype=np.uint16, mode="r", offset=1024)
val = look[np.stack([np.asarray(va[x:x + a.L]) for x in np.linspace(0, len(va) - a.L - 1, a.nval).astype(np.int64)]).astype(np.int64)]

# training occurrences of each word as a target in the tokens this run reads
occ = np.bincount(rows[:, 1:].reshape(-1), minlength=V).astype(np.int64)
EDGES = [0, 1, 10, 32, 100, 316, 1000, 3162, 10 ** 12]
BN = ["0", "1-9", "10-31", "32-99", "100-315", "316-999", "1k-3k", "3k+"]
obin = np.searchsorted(EDGES, occ, side="right") - 1; obin[UNK] = len(BN)  # unk separate
pw = freq / counts.sum()  # corpus frequency of each word
FEDGES = [0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1.1]
FN = ["<1e-6", "1-3e-6", "3e-6-1e-5", "1-3e-5", "3e-5-1e-4", "1-3e-4", "3e-4-1e-3", ">1e-3"]
fbin = np.searchsorted(FEDGES, pw, side="right") - 1; fbin[UNK] = len(FN)


_RC = {}


def rope(x):
    t, d = x.shape[-2], x.shape[-1]
    if (t, d) not in _RC:
        f = 1.0 / 10000 ** (torch.arange(0, d, 2).float() / d); ang = torch.outer(torch.arange(t).float(), f); _RC[(t, d)] = (ang.cos().to(x.device), ang.sin().to(x.device))
    c, s = _RC[(t, d)]
    x1, x2 = x[..., 0::2], x[..., 1::2]; return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


VEL = [int(z) for z in a.vemb.split(",") if z]


class Block(nn.Module):
    def __init__(s, d, H, i=0):
        super().__init__(); s.H, s.dk = H, d // H
        s.vt = nn.Parameter(torch.randn(V, d) * d ** -0.5) if i in VEL else None
        s.n1, s.n2, s.qn, s.kn = nn.RMSNorm(d), nn.RMSNorm(d), nn.RMSNorm(d // H), nn.RMSNorm(d // H)
        s.q, s.k, s.v, s.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        f = int(8 * d / 3) // 8 * 8; s.gate, s.up, s.down = nn.Linear(d, f, bias=False), nn.Linear(d, f, bias=False), nn.Linear(f, d, bias=False)

    def forward(s, x, ids=None):
        b, t, d = x.shape; h = s.n1(x); sp = lambda z: z.view(b, t, s.H, s.dk).transpose(1, 2)
        q, k = rope(s.qn(sp(s.q(h)))), rope(s.kn(sp(s.k(h))))
        y = F.scaled_dot_product_attention(q, k, sp(s.v(h) + s.vt[ids] if s.vt is not None else s.v(h)), is_causal=True)
        x = x + s.o(y.transpose(1, 2).reshape(b, t, d)); h = s.n2(x)
        return x + s.down(F.silu(s.gate(h)) * s.up(h))


def kmeans(X, K, seed=1, iters=10):
    mu = X[torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))[:K].to(X.device)].clone()
    for _ in range(iters):
        asg = torch.cdist(X, mu).argmin(1)
        n = torch.bincount(asg, minlength=K)
        s = torch.zeros_like(mu).index_add_(0, asg, X)
        mu = torch.where(n.unsqueeze(1) > 0, s / n.clamp(min=1).unsqueeze(1), mu)
    return torch.cdist(X, mu).argmin(1)


def freq_classes(K):  # Mikolov 2011: classes of equal unigram mass, by frequency rank
    o = np.argsort(-freq, kind="stable"); cum = np.cumsum(freq[o]) / freq.sum()
    c = np.minimum((cum * K).astype(np.int64), K - 1); out = np.empty(V, np.int64); out[o] = c; return out


class Model(nn.Module):
    def __init__(s):
        super().__init__(); d = a.width
        s.head = nn.Parameter(torch.randn(V, d) * d ** -0.5 * a.hinit)
        s.emb = nn.Parameter(torch.randn(V, d) * d ** -0.5)
        s.blocks = nn.ModuleList(Block(d, a.heads, i) for i in range(a.layers)); s.norm = nn.RMSNorm(d); s.sg = nn.Parameter(torch.full((d,), -2.0))
        K = max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d)); s.Clast = torch.zeros(K, d)
        if a.cinit == "freq": asg = freq_classes(K)
        else: asg = np.random.default_rng(5).integers(0, K, V)
        s.register_buffer("ah", torch.tensor(asg))
        s.uh = a.clus > 0
        if a.cls:
            s.Cc = nn.Parameter(torch.zeros(a.cls, d))
            s.register_buffer("ac", torch.tensor(freq_classes(a.cls) if a.clsby == "freq" else np.random.default_rng(5).integers(0, a.cls, V)))
        if a.asm:
            o = np.argsort(-freq, kind="stable"); s.register_buffer("perm", torch.tensor(np.argsort(o)))  # word id -> frequency rank
            cuts = [int(float(c) * V) for c in a.cuts.split(",")]
            s.asmod = nn.AdaptiveLogSoftmaxWithLoss(d, V, cuts, div_value=a.asm, head_bias=False)
        s.coff = False; s.crev = False

    def H(s):
        if not s.uh: return s.head
        C = s.Clast if s.crev else (torch.zeros_like(s.Ch) if s.coff else s.Ch)
        return s.head + C[s.ah]

    @torch.no_grad()
    def recluster(s):
        R = s.head + s.Ch[s.ah]; K = s.Ch.shape[0]
        asg = kmeans(R, K)
        cm = torch.zeros(K, R.shape[1], device=R.device).index_add_(0, asg, R) / torch.bincount(asg, minlength=K).clamp(min=1).unsqueeze(1)
        s.ah.copy_(asg); s.Ch.copy_(cm); s.head.copy_(R - cm[asg]); s.Clast = cm.clone()

    @torch.no_grad()
    def recluster_cls(s):
        s.ac.copy_(kmeans(s.head.detach(), a.cls))

    def hidden(s, ids):
        x = (s.H() if a.tie else s.emb)[ids]
        if a.smear: x = x + torch.sigmoid(s.sg) * F.pad(x, (0, 0, 1, 0))[:, :-1]
        x = x * a.escale
        for blk in s.blocks: x = blk(x, ids)
        return s.norm(x)

    def nll(s, ids, tg):  # per-token negative log-likelihood
        h = s.hidden(ids).reshape(-1, a.width); tg = tg.reshape(-1)
        if a.asm:
            return -s.asmod.log_prob(h).gather(1, s.perm[tg].unsqueeze(1)).squeeze(1)
        lg = F.linear(h, s.H())
        if a.cls:
            m = lg.max(1, keepdim=True).values.detach(); e = (lg - m).exp()
            S = torch.zeros(len(lg), a.cls, device=lg.device).index_add_(1, s.ac, e); lse_c = S.clamp(min=1e-30).log() + m
            clp = F.log_softmax(F.linear(h, s.Cc), -1)
            c = s.ac[tg]
            return -(clp.gather(1, c[:, None]) + lg.gather(1, tg[:, None]) - lse_c.gather(1, c[:, None])).squeeze(1)
        return F.cross_entropy(lg, tg, reduction="none")


def newton_schulz(G, steps=5):
    A, B, C = 3.4445, -4.7750, 2.0315; X = G / (G.norm() + 1e-7); tr_ = X.shape[0] > X.shape[1]
    if tr_: X = X.T
    for _ in range(steps): M = X @ X.T; X = A * X + (B * M + C * M @ M) @ X
    return X.T if tr_ else X


class Muon(torch.optim.Optimizer):
    def __init__(s, ps, lr): super().__init__(ps, dict(lr=lr))

    @torch.no_grad()
    def step(s):
        for g in s.param_groups:
            for w in g["params"]:
                if w.grad is None: continue
                m = s.state[w].setdefault("m", torch.zeros_like(w)); m.mul_(a.mb).add_(w.grad)
                w.add_(newton_schulz(w.grad + a.mb * m), alpha=-g["lr"] * max(1, w.shape[0] / w.shape[1]) ** 0.5)


class CoupledAdam(torch.optim.Optimizer):
    def __init__(s, ps, lr): super().__init__(ps, dict(lr=lr)); s.t = 0

    @torch.no_grad()
    def step(s):
        s.t += 1
        for g in s.param_groups:
            for w in g["params"]:
                if w.grad is None: continue
                st = s.state[w]; mm = st.setdefault("m", torch.zeros_like(w)); vv = st.setdefault("v", torch.zeros(1, w.shape[1], device=w.device))
                mm.mul_(a.b1).add_(w.grad, alpha=1 - a.b1); vv.mul_(a.b2).add_(w.grad.pow(2).mean(0, keepdim=True), alpha=1 - a.b2)
                w.add_((mm / (1 - a.b1 ** s.t)) / ((vv / (1 - a.b2 ** s.t)).sqrt() + a.eps), alpha=-g["lr"])


VX = torch.tensor(val).to(D); VB = obin[val[:, 1:].reshape(-1)]; VF = fbin[val[:, 1:].reshape(-1)]


@torch.no_grad()
def evaluate(m):
    m.eval(); per = torch.cat([m.nll(VX[i:i + 50, :-1], VX[i:i + 50, 1:]) for i in range(0, len(VX), 50)]).cpu().numpy(); m.train()
    return per


def main():
    m = Model().to(D); m.Clast = m.Clast.to(D)
    vts = [b.vt for b in m.blocks if b.vt is not None]
    mats = [w for n, w in m.named_parameters() if w.ndim == 2 and n.startswith("blocks") and all(w is not z for z in vts)]
    special = [m.head, m.emb, m.Ch] + ([m.Cc] if a.cls else [])
    rest = [w for n, w in m.named_parameters() if (not n.startswith("blocks") or w.ndim < 2) and not n.startswith("asmod")]
    rest = [w for w in rest if all(w is not z for z in special)] + vts
    hg = dict(params=[m.head] + ([m.Cc] if a.cls else []), lr=a.hlr)
    if a.heps > 0: hg["eps"] = a.heps
    if a.hb2 > 0: hg["betas"] = (a.b1, a.hb2)
    groups = [dict(params=rest + [m.emb], lr=a.elr), hg, dict(params=[m.Ch], lr=a.clr)]
    if a.asm: groups = [groups[0], dict(params=list(m.asmod.parameters()), lr=a.hlr)]
    opts = [Muon(mats, a.lr)]
    if a.coupled:
        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))
    opts.append(torch.optim.Adam(groups, betas=(a.b1, a.b2), eps=a.eps))
    base = [[g["lr"] for g in o.param_groups] for o in opts]
    t0 = time.time(); X = torch.tensor(rows).to(D)
    for step in range(a.steps):
        if a.clus and a.cinit == "kmeans" and a.recl and step % a.recl == 0 and step > 0 and step < a.cstop * a.steps: m.recluster()
        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()
        sc = min((step + 1) / min(50, max(1, a.steps // 10)), 1.0, (a.steps - step) / (a.cool * a.steps))
        for o, bl in zip(opts, base):
            for g, lr in zip(o.param_groups, bl): g["lr"] = lr * sc
        x = X[step * a.batch:(step + 1) * a.batch]
        for xc in x.chunk(a.acc):
            (m.nll(xc[:, :-1], xc[:, 1:]).sum() / (x.shape[0] * (x.shape[1] - 1))).backward()
        for o in opts: o.step()
        m.zero_grad(set_to_none=True)
    if a.dev == 'mps': torch.mps.synchronize()
    ts = time.time() - t0
    per = evaluate(m)
    r = {"val": float(per.mean())}
    bins = {}
    for i, n in enumerate(BN + ["unk"]):
        sel = VB == i
        if sel.any(): bins[n] = [round(float(per[sel].mean()), 4), round(float(sel.mean()), 4)]
    fb = {}
    for i, n in enumerate(FN + ["unk"]):
        sel = VF == i
        if sel.any(): fb[n] = [round(float(per[sel].mean()), 4), round(float(sel.mean()), 4)]
    if m.uh:
        m.crev = True; r["val_crev"] = float(evaluate(m).mean()); m.crev = False
        m.coff = True; r["val_coff"] = float(evaluate(m).mean()); m.coff = False
    if a.shrink:  # post-hoc shrinkage of rare rows toward their k-means centroid (Khassanov-style), on the final head
        with torch.no_grad():
            R0 = m.H().detach().clone(); asg = kmeans(R0, 64)
            cm = torch.zeros(64, R0.shape[1], device=R0.device).index_add_(0, asg, R0) / torch.bincount(asg, minlength=64).clamp(min=1).unsqueeze(1)
            oc = torch.tensor(occ, dtype=torch.float32, device=R0.device).unsqueeze(1)
            hold = m.head.data.clone(); uh = m.uh; m.uh = False
            for n0 in [float(z) for z in a.shrink.split(",")]:
                m.head.data.copy_(R0 + n0 / (n0 + oc) * (cm[asg] - R0)); r[f"val_shr{n0:g}"] = float(evaluate(m).mean())
            m.head.data.copy_(hold); m.uh = uh
    nw = a.width
    tpw = need / (sum(w.numel() for w in mats) + 2 * V * nw)
    rare_mass = float(sum(v[1] for k, v in bins.items() if k in ("0", "1-9", "10-31", "32-99")))
    res = dict(tag=a.tag, args=vars(a), tokens=need, tok_per_weight=round(tpw, 2), rare_mass_lt100=round(rare_mass, 4),
               train_s=round(ts, 1), scores={k: round(v, 4) for k, v in r.items()}, bins=bins, fbins=fb)
    if a.save:
        torch.save(dict(head=m.H().detach().cpu(), ah=m.ah.cpu(), occ=torch.tensor(occ)), a.save)
    with open(os.path.join(HERE, a.out), "a") as f: f.write(json.dumps(res) + "\n")
    extra = " ".join(f"{k} {v:.4f}" for k, v in r.items() if k != "val")
    print(f"{a.tag}: val {r['val']:.4f} {extra} | rare<100 {rare_mass:.3f} | {res['train_s']}s", flush=True)


if __name__ == "__main__":
    main()

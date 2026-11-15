"""Test bed for embedding / output-layer ideas (embeddings-area session, 2026-11-15).

Model: pre-norm RMSNorm, RoPE, QK-norm, SwiGLU, no biases, untied head by default.
Muon on block matrices (beta --mb), Adam on embedding, head, norms (betas --b1/--b2, eps --eps).
Data: FineWeb GPT-2 tokens mapped to the top-V words plus one unknown token, each token read once.

Idea flags:
  --tie 1            tied model (head rows used as input rows)
  --untie_at f       start tied, at fraction f of training copy the head into a new input table and train both
                     (f=0: untied model whose input table starts as a copy of the head)
  --soft lam         untied, plus penalty lam * mean ||E - H||^2 pulling input rows toward head rows

usage: nice -n 20 python3 tb.py --V 2000 --width 32 --steps 400 --threads 1 --tag name
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "llm-per-word-weights", "data"); EOT = 50256

p = argparse.ArgumentParser()
p.add_argument("--width", type=int, default=32); p.add_argument("--layers", type=int, default=2); p.add_argument("--heads", type=int, default=2)
p.add_argument("--V", type=int, default=2000); p.add_argument("--L", type=int, default=64); p.add_argument("--batch", type=int, default=32)
p.add_argument("--steps", type=int, default=400); p.add_argument("--lr", type=float, default=0.02); p.add_argument("--elr", type=float, default=0.05)
p.add_argument("--hlr", type=float, default=-1, help="Adam rate of the head (default --elr)")
p.add_argument("--mb", type=float, default=0.95); p.add_argument("--b1", type=float, default=0.9); p.add_argument("--b2", type=float, default=0.95)
p.add_argument("--eps", type=float, default=1e-8); p.add_argument("--escale", type=float, default=4.0)
p.add_argument("--cool", type=float, default=0.3)
p.add_argument("--tie", type=int, default=0); p.add_argument("--untie_at", type=float, default=-1); p.add_argument("--soft", type=float, default=0)
p.add_argument("--clus", type=int, default=0, help="K>0: each row = own vector + shared vector of its cluster (K clusters)")
p.add_argument("--cwhere", default="both", help="both / head / emb")
p.add_argument("--recl", type=int, default=50, help="re-cluster by k-means on the effective rows every N steps (0: never)")
p.add_argument("--cinit", default="kmeans", help="kmeans (re-clustered) / rand (fixed random assignment) / freq (fixed frequency-rank buckets)")
p.add_argument("--clr", type=float, default=-1, help="Adam rate of cluster vectors (default --elr)")
p.add_argument("--cstop", type=float, default=1.0, help="re-cluster only while step < cstop * steps")
p.add_argument("--coupled", type=int, default=0, help="1: Coupled Adam on the head (second moment averaged over vocabulary rows)")
p.add_argument("--lrank", type=int, default=0, help="r>0: head row = own row + A[w] @ P (rank r shared part, A starts at zero); control for cluster sharing")
p.add_argument("--hinit", type=float, default=1.0, help="head init scale times d^-0.5 (0 = zero init)")
p.add_argument("--cfeat", default="row", help="row: k-means on head rows; ctx: on mean final hidden state at positions where the word is the target")
p.add_argument("--einit", type=float, default=1.0, help="input table init scale times d^-0.5")
p.add_argument("--vemb", default="", help="layers with value embeddings v += T_l[current word]")
p.add_argument("--smear", type=int, default=0, help="1: input x_t += sigmoid(gate) * x_{t-1}")
p.add_argument("--rown", default="", help="head / both: train these tables with RowNorm (2605.18106): heavy-ball momentum 0.95, each row stepped to norm lr")
p.add_argument("--rlr", type=float, default=0.01)
p.add_argument("--rlre", type=float, default=0.3)
p.add_argument("--clus2", type=int, default=0, help="second head codebook fitted to the residual after the first (residual quantization)")
p.add_argument("--nval", type=int, default=600); p.add_argument("--threads", type=int, default=1)
p.add_argument("--tag", default=""); p.add_argument("--out", default="results.jsonl")
a = p.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(0)
HLR = a.hlr if a.hlr > 0 else a.elr

counts = np.load(os.path.join(DATA, "counts.npy")).astype(np.float64)
keep = np.unique(np.concatenate([np.argsort(counts)[-(a.V - 2):], [EOT]])); V = len(keep) + 1; UNK = V - 1
look = np.full(65536, UNK, np.int64); look[keep] = np.arange(len(keep))
freq = np.append(counts[keep], counts.sum() - counts[keep].sum())
order = np.argsort(-freq[:-1], kind="stable"); cover = np.cumsum(freq[order]) / freq[:-1].sum()
cov_of = np.empty(V); cov_of[order] = cover - freq[order] / freq[:-1].sum(); cov_of[UNK] = 2.0
bucket = np.where(cov_of < 0.5, 0, np.where(cov_of < 0.8, 1, 2)); bucket[UNK] = 3
NB = ["common", "middle", "rare", "unk"]

need = a.steps * a.batch * a.L
tr = np.memmap(os.path.join(DATA, "fineweb_train_000001.bin"), dtype=np.uint16, mode="r", offset=1024)
rows = look[np.asarray(tr[:need]).astype(np.int64).reshape(-1, a.L)]
va = np.memmap(os.path.join(DATA, "fineweb_val_000000.bin"), dtype=np.uint16, mode="r", offset=1024)
val = look[np.stack([np.asarray(va[x:x + a.L]) for x in np.linspace(0, len(va) - a.L - 1, a.nval).astype(np.int64)]).astype(np.int64)]


def rope(x):
    t, d = x.shape[-2], x.shape[-1]
    f = 1.0 / 10000 ** (torch.arange(0, d, 2).float() / d); ang = torch.outer(torch.arange(t).float(), f); c, s = ang.cos(), ang.sin()
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


class Model(nn.Module):
    def __init__(s):
        super().__init__(); d = a.width
        s.head = nn.Parameter(torch.randn(V, d) * d ** -0.5 * a.hinit)
        s.emb = nn.Parameter(torch.randn(V, d) * d ** -0.5 * a.einit)
        s.tied = bool(a.tie) or a.untie_at > 0
        if a.untie_at == 0: s.emb.data.copy_(s.head.data)
        s.blocks = nn.ModuleList(Block(d, a.heads, i) for i in range(a.layers)); s.norm = nn.RMSNorm(d); s.sg = nn.Parameter(torch.full((d,), -2.0))
        K = max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d)); s.Ce = nn.Parameter(torch.zeros(K, d))
        if a.cinit == "freq": asg = np.minimum((np.argsort(np.argsort(-freq)) * K) // V, K - 1)
        else: asg = np.random.default_rng(5).integers(0, K, V)
        s.Ch2 = nn.Parameter(torch.zeros(max(a.clus2, 1), d)); s.register_buffer("ah2", torch.zeros(V, dtype=torch.long))
        s.LA = nn.Parameter(torch.zeros(V, max(a.lrank, 1))); s.LP = nn.Parameter(torch.randn(max(a.lrank, 1), d) * d ** -0.5)
        s.register_buffer("ah", torch.tensor(asg)); s.register_buffer("ae", torch.tensor(asg))
        s.ctx = torch.zeros(V, a.width); s.cnt = torch.zeros(V)
        s.uh = a.clus > 0 and a.cwhere in ("both", "head"); s.ue = a.clus > 0 and a.cwhere in ("both", "emb")

    ctx = None

    def H(s):
        W = s.head + s.Ch[s.ah] if s.uh else s.head
        if a.clus2: W = W + s.Ch2[s.ah2]
        return W + s.LA @ s.LP if a.lrank else W

    def E(s): return s.emb + s.Ce[s.ae] if s.ue else s.emb

    @torch.no_grad()
    def recluster(s):
        for use, W, C, A in ((s.uh, s.head, s.Ch, s.ah), (s.ue, s.emb, s.Ce, s.ae)):
            if not use: continue
            R = W + C[A] + (s.Ch2[s.ah2] if a.clus2 and W is s.head else 0); K = C.shape[0]
            Fe = R
            if a.cfeat == "ctx" and W is s.head:
                Fe = F.normalize(R, dim=-1).clone(); seen = s.cnt > 0; Fe[seen] = F.normalize(s.ctx[seen], dim=-1)
            mu = Fe[torch.randperm(len(Fe), generator=torch.Generator().manual_seed(1))[:K]].clone()
            for _ in range(10):
                asg = torch.cdist(Fe, mu).argmin(1)
                for k in range(K):
                    sel = asg == k
                    if sel.any(): mu[k] = Fe[sel].mean(0)
            asg = torch.cdist(Fe, mu).argmin(1)
            cm = torch.zeros(K, R.shape[1]).index_add_(0, asg, R) / torch.bincount(asg, minlength=K).clamp(min=1).unsqueeze(1)
            A.copy_(asg); C.copy_(cm); R1 = R - cm[asg]
            if a.clus2 and W is s.head:
                K2 = a.clus2; mu = R1[torch.randperm(V, generator=torch.Generator().manual_seed(2))[:K2]].clone()
                for _ in range(10):
                    a2 = torch.cdist(R1, mu).argmin(1)
                    for k in range(K2):
                        sel = a2 == k
                        if sel.any(): mu[k] = R1[sel].mean(0)
                a2 = torch.cdist(R1, mu).argmin(1)
                c2 = torch.zeros(K2, R.shape[1]).index_add_(0, a2, R1) / torch.bincount(a2, minlength=K2).clamp(min=1).unsqueeze(1)
                s.ah2.copy_(a2); s.Ch2.copy_(c2); R1 = R1 - c2[a2]
            W.copy_(R1)

    def forward(s, ids):
        x = (s.H() if s.tied else s.E())[ids]
        if a.smear: x = x + torch.sigmoid(s.sg) * F.pad(x, (0, 0, 1, 0))[:, :-1]
        x = x * a.escale
        for blk in s.blocks: x = blk(x, ids)
        s.last = s.norm(x); return F.linear(s.last, s.H())


def newton_schulz(G, steps=5):
    A, B, C = 3.4445, -4.7750, 2.0315; X = G / (G.norm() + 1e-7); tr_ = X.shape[0] > X.shape[1]
    if tr_: X = X.T
    for _ in range(steps): M = X @ X.T; X = A * X + (B * M + C * M @ M) @ X
    return X.T if tr_ else X


class RowNorm(torch.optim.Optimizer):
    def __init__(s, ps, lr): super().__init__(ps, dict(lr=lr))

    @torch.no_grad()
    def step(s):
        for g in s.param_groups:
            for w in g["params"]:
                if w.grad is None: continue
                m = s.state[w].setdefault("m", torch.zeros_like(w)); m.mul_(0.95).add_(w.grad)
                w.add_(m / (m.norm(dim=1, keepdim=True) + 1e-8), alpha=-g["lr"])


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
    """Adam whose second moment is averaged over the rows (vocabulary) of the matrix (Stollenwerk & Stollenwerk 2025)."""
    def __init__(s, ps, lr): super().__init__(ps, dict(lr=lr)); s.t = 0

    @torch.no_grad()
    def step(s):
        s.t += 1
        for g in s.param_groups:
            for w in g["params"]:
                if w.grad is None: continue
                st = s.state[w]; mm = st.setdefault("m", torch.zeros_like(w)); vv = st.setdefault("v", torch.zeros(1, w.shape[1]))
                mm.mul_(a.b1).add_(w.grad, alpha=1 - a.b1); vv.mul_(a.b2).add_(w.grad.pow(2).mean(0, keepdim=True), alpha=1 - a.b2)
                w.add_((mm / (1 - a.b1 ** s.t)) / ((vv / (1 - a.b2 ** s.t)).sqrt() + a.eps), alpha=-g["lr"])


VX = torch.tensor(val); VTG = bucket[val[:, 1:].reshape(-1)]


@torch.no_grad()
def evaluate(m):
    m.eval(); per = []
    for i in range(0, len(VX), 200):
        x = VX[i:i + 200]; per.append(F.cross_entropy(m(x[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1), reduction="none"))
    per = torch.cat(per).numpy(); m.train()
    r = {"val": float(per.mean())}
    for i, n in enumerate(NB): r[f"tg_{n}"] = float(per[VTG == i].mean())
    return {k: round(v, 4) for k, v in r.items()}


def main():
    m = Model()
    vts = [b.vt for b in m.blocks if b.vt is not None]
    mats = [w for n, w in m.named_parameters() if w.ndim == 2 and n.startswith("blocks") and all(w is not z for z in vts)]
    rest = [w for n, w in m.named_parameters() if not n.startswith("blocks") or w.ndim < 2]
    rest = [w for w in rest if w is not m.head and w is not m.emb] + vts
    rest = [w for w in rest if all(w is not z for z in (m.Ch, m.Ce, m.LA, m.LP, m.Ch2))]
    groups = [dict(params=rest + [m.emb], lr=a.elr), dict(params=[m.head], lr=HLR), dict(params=[m.Ch, m.Ce, m.LA, m.LP, m.Ch2], lr=a.clr if a.clr > 0 else a.elr)]
    if a.coupled: groups = [g for g in groups if g["params"][0] is not m.head]
    if a.rown:
        rn = [m.head] + ([m.emb] if a.rown == "both" else [])
        for g in groups: g["params"] = [w for w in g["params"] if all(w is not z for z in rn)]
        groups = [g for g in groups if g["params"]]
    opts = [Muon(mats, a.lr), torch.optim.Adam(groups, betas=(a.b1, a.b2), eps=a.eps)]
    if a.rown: opts.append(RowNorm([dict(params=[m.head], lr=a.rlr)] + ([dict(params=[m.emb], lr=a.rlre)] if a.rown == "both" else []), a.rlr))
    if a.coupled: opts.append(CoupledAdam([m.head], HLR))
    base = [[g["lr"] for g in o.param_groups] for o in opts]
    t0 = time.time(); X = torch.tensor(rows); ua = int(a.untie_at * a.steps) if a.untie_at > 0 else -1
    for step in range(a.steps):
        if step == ua:
            with torch.no_grad(): m.emb.copy_(m.head)
            m.tied = False
        if a.clus and a.cinit == "kmeans" and a.recl and step % a.recl == 0 and step > 0 and step < a.cstop * a.steps: m.recluster()
        sc = min((step + 1) / min(50, max(1, a.steps // 10)), 1.0, (a.steps - step) / (a.cool * a.steps))
        for o, bl in zip(opts, base):
            for g, lr in zip(o.param_groups, bl): g["lr"] = lr * sc
        x = X[step * a.batch:(step + 1) * a.batch]
        loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1))
        if a.soft and not m.tied: loss = loss + a.soft * (m.emb - m.head).pow(2).sum(-1).mean()
        loss.backward()
        if a.cfeat == "ctx":
            with torch.no_grad(): tg = x[:, 1:].reshape(-1); m.ctx.index_add_(0, tg, m.last.reshape(-1, a.width)); m.cnt.index_add_(0, tg, torch.ones(len(tg)))
        for o in opts: o.step()
        m.zero_grad(set_to_none=True)
    r = evaluate(m)
    res = dict(tag=a.tag, args=vars(a), tokens=need, train_s=round(time.time() - t0, 1), scores=r)
    with open(os.path.join(HERE, a.out), "a") as f: f.write(json.dumps(res) + "\n")
    print(f"{a.tag}: val {r['val']:.4f} c/m/r/u {r['tg_common']:.3f}/{r['tg_middle']:.3f}/{r['tg_rare']:.3f}/{r['tg_unk']:.3f} | {res['train_s']}s", flush=True)


if __name__ == "__main__":
    main()

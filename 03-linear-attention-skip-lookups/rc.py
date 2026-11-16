"""Copy-circuit bed (MQAR + induction) built on the attention test bed. Same model and flags as idea-attn-baseline-as/at.py.
--task mqar: N key-value pairs, then query keys each followed by its value (loss + accuracy on the values after query keys).
--task ind: random tokens with one segment of length --seg copied from the first half into the second half (loss + accuracy on the copied tokens).
Reports final accuracy and steps90 (first eval step with accuracy >= 0.9, evaluated every --ee steps on 512 fixed rows).
Original header:
Attention test bed (from llm-partial-rope/pr.py). Modern toy: RMSNorm, RoPE, QK-norm, SwiGLU, no biases, untied head,
Muon on block matrices, Adam on embeddings / norms / scalars. One run = one JSON line in results.jsonl.
Attention variants (all off by default = plain causal softmax attention):
  --vp P      multiply each head's output by (sum_j a_ij^2)^(-P/2)  (P=1: variance-preserving, output scale independent of spread)
  --vpl 1     learned per-head exponent P (init --vp)
  --hnorm 1   per-head RMSNorm of the attention output with learned gain (DIFF-Transformer style)
  --sink 1    learned per-head sink logit in the softmax denominator (gpt-oss)
  --gate 1    head-wise sigmoid output gate from the layer input (Qiu et al. 2025)
usage: nice -n 20 python3 at.py --threads 1 --tag x
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import mix as MIX

p = argparse.ArgumentParser()
p.add_argument("--width", type=int, default=64); p.add_argument("--layers", type=int, default=4); p.add_argument("--heads", type=int, default=4)
p.add_argument("--V", type=int, default=2000); p.add_argument("--L", type=int, default=64); p.add_argument("--batch", type=int, default=64)
p.add_argument("--steps", type=int, default=150); p.add_argument("--lr", type=float, default=0.04); p.add_argument("--elr", type=float, default=0.04)
p.add_argument("--beta", type=float, default=0.7); p.add_argument("--ab1", type=float, default=0.9); p.add_argument("--ab2", type=float, default=0.95)
p.add_argument("--eps", type=float, default=1e-8); p.add_argument("--slr", type=float, default=0.0, help="Adam lr of attention scalars (0 = --elr)")
p.add_argument("--warm", type=int, default=20); p.add_argument("--cool", type=float, default=0.3)
p.add_argument("--data", default="tok"); p.add_argument("--base", type=float, default=10000.0); p.add_argument("--qkn", type=int, default=1)
p.add_argument("--vp", type=float, default=0.0); p.add_argument("--vpl", type=int, default=0)
p.add_argument("--js", type=int, default=0, help="joint head softmax: 1 fixed (share = softmax_h(lse_h)), 2 learned per-layer beta init 1 + head bias, 3 learned beta init 0 + head bias"); p.add_argument("--hnorm", type=int, default=0); p.add_argument("--sink", type=int, default=1); p.add_argument("--gate", type=int, default=0)
p.add_argument("--bos", type=int, default=0, help="1: put an end-of-text token at position 0 of every row")
p.add_argument("--testoff", type=int, default=0, help="1: also report val with the variant switched off at test time")
p.add_argument("--compile", type=int, default=0); p.add_argument("--threads", type=int, default=1); p.add_argument("--dev", default="mps")
p.add_argument("--kt", default="", help="layers with a per-word key table, e.g. 0,1,2,3")
p.add_argument("--qt", default="", help="layers with a per-word query table")
p.add_argument("--vt", default="", help="layers with a per-word value table (value embeddings, published competitor)")
p.add_argument("--tlr", type=float, default=0.0, help="Adam lr of the tables (0 = --elr)")
p.add_argument("--dd", type=int, default=0, help="dedup: logit_ij -= g_h * log C[j,i]; 1: C = sum over l<=i of exp(tau(cos(k_j,k_l)-1)); 2: only later copies j<=l<=i (newest copy wins)")
p.add_argument("--dg0", type=float, default=0.0, help="initial g per head"); p.add_argument("--dtau", type=float, default=8.0)
p.add_argument("--dgfix", type=int, default=0, help="1: g fixed at --dg0 (not learned)")
p.add_argument("--ddctl", type=int, default=0, help="1: content-free control, cos(k_j,k_l)=0 off the diagonal (penalty = log(1 + e^-tau (i-j)))")
p.add_argument("--alibi", type=int, default=1, help="1: learned per-head linear distance penalty (ALiBi slopes, init 2^-(8h/H) * --ab0)")
p.add_argument("--ab0", type=float, default=1.0)
p.add_argument("--sel", type=int, default=0, help="1: Selective Attention (Leviathan 2024): head 0 logits, ReLU, later tokens mask earlier ones for all heads")
p.add_argument("--qslope", type=int, default=0, help="1: per-query slope softplus(w_h . x_i + b_h) * (i - j), starts as learned ALiBi (w=0); 2: per-key slope (control)")
p.add_argument("--fox", type=int, default=0, help="1: Forgetting Transformer gate: bias_ij = sum_{j<l<=i} log sigmoid(w_h . x_l + b_h)")
p.add_argument("--fb0", type=float, default=3.0)
p.add_argument("--lme", type=int, default=0, help="soft-max pooling of values: y = tau log(A exp(v/tau)); 1 fixed tau=--lt, 2 learned per-head tau (init --lt)")
p.add_argument("--lt", type=float, default=1.0)
p.add_argument("--kc", type=int, default=0, help="key centering before QK-norm: k_j -= alpha * mean(k_l, l<j); 1 alpha=1, 2 learned per-head alpha (init --kca), 3 static learned mean vector per head (control)")
p.add_argument("--kca", type=float, default=0.0); p.add_argument("--qc", type=int, default=0, help="1: center queries the same way")
p.add_argument("--lsef", type=int, default=0, help="write each head's match strength into the residual: 1 lse - log(n), 2 lse, 3 lse - log(n) and max logit")
p.add_argument("--mix", default="", help="layer codes cycled over layers: s softmax, d gated delta, D delta, g gated linear (see mix.py)"); p.add_argument("--sc", type=int, default=4); p.add_argument("--ba", type=float, default=-4.0); p.add_argument("--loop", type=int, default=0)
p.add_argument("--tag", default=""); p.add_argument("--init", type=float, default=0.25); p.add_argument("--diag", type=int, default=0); p.add_argument("--dconv", type=int, default=0, help="Primer MDHA (2109.08668): depthwise causal conv of kernel K on q, k, v channels, identity init"); p.add_argument("--kvs", type=int, default=0, help="KV shifting attention (2411.19574): k = a1 k_j + a2 k_{j-1}, v = b1 v_j + b2 v_{j-1}, per-head learned"); p.add_argument("--task", default="mqar"); p.add_argument("--pairs", type=int, default=16); p.add_argument("--tv", type=int, default=256, help="synthetic vocabulary"); p.add_argument("--seg", type=int, default=16); p.add_argument("--ee", type=int, default=25); p.add_argument("--rep", type=float, default=0.0, help="share of train rows whose second half repeats the first half; also reports val_rep (loss on the repeated half)"); p.add_argument("--qkg", type=float, default=1.0); p.add_argument("--smear", type=int, default=0, help="smeared keys (Olsson 2022): k_j = s*k_j + (1-s)*k_{j-1}, per-head learned s")
MIX.args(p)
p.add_argument("--qdist", type=int, default=0, help="1: each key queried at most once")
p.add_argument("--seed", type=int, default=0, help="paired initialization and train-data seed; eval seed remains fixed")
p.add_argument("--posthoc", type=int, default=0, help="1: evaluate frozen trained model with write and full-update oracles (no retraining)")
a = p.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
DEV = torch.device(a.dev)
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data"); EOT = 50256
MATS = ["q", "k", "v", "o", "gate", "up", "down"]

V = a.tv; BOS = 0; H2 = a.L // 2; need = a.steps * a.batch * a.L


def make(B, rng):
    """returns (seq (B, L+1) int64, mask (B, L) bool over input positions whose next token is scored)"""
    L = a.L + 1; seq = rng.integers(1, V, (B, L)); mask = np.zeros((B, L - 1), bool)
    if a.task == "mqar":
        N = a.pairs; Kn = V // 2 - 1; assert Kn >= N and 2 * N < L
        keys = np.argsort(rng.random((B, Kn)), 1)[:, :N] + 1; vals = rng.integers(Kn + 1, V, (B, N))
        seq[:, 0:2 * N:2], seq[:, 1:2 * N:2] = keys, vals
        Q = (L - 2 * N) // 2; idx = rng.integers(0, N, (B, Q)); r = np.arange(B)[:, None]
        if a.qdist: Q = min(Q, N); idx = np.argsort(rng.random((B, N)), 1)[:, :Q]
        seq[:, 2 * N:2 * N + 2 * Q:2], seq[:, 2 * N + 1:2 * N + 2 * Q:2] = keys[r, idx], vals[r, idx]
        mask[:, 2 * N:2 * N + 2 * Q:2] = True
    elif a.task == "mqar2":   # pairs A, queries on A, new pairs B, queries on A+B (new writes arrive after a query phase)
        N = a.pairs; h_ = N // 2; Kn = V // 2 - 1; Q = (L - 1 - 2 * N) // 4; r = np.arange(B)[:, None]
        keys = np.argsort(rng.random((B, Kn)), 1)[:, :N] + 1; vals = rng.integers(Kn + 1, V, (B, N)); p_ = 0
        for blk, hi in ((0, h_), (1, N)):
            lo = blk * h_; seq[:, p_:p_ + 2 * h_:2], seq[:, p_ + 1:p_ + 2 * h_:2] = keys[:, lo:hi], vals[:, lo:hi]; p_ += 2 * h_
            idx = rng.integers(0, hi, (B, Q)); seq[:, p_:p_ + 2 * Q:2], seq[:, p_ + 1:p_ + 2 * Q:2] = keys[r, idx], vals[r, idx]
            mask[:, p_:p_ + 2 * Q:2] = True; p_ += 2 * Q
    elif a.task == "hop2":   # pairs k -> f(k), f a random permutation of N keys; query k, answer f(f(k))
        N = a.pairs; Kn = V - 1; assert Kn >= N and 2 * N < L
        keys = np.argsort(rng.random((B, Kn)), 1)[:, :N] + 1; perm = np.argsort(rng.random((B, N)), 1); r = np.arange(B)[:, None]
        seq[:, 0:2 * N:2], seq[:, 1:2 * N:2] = keys, keys[r, perm]
        Q = min(N, (L - 2 * N) // 2); idx = np.argsort(rng.random((B, N)), 1)[:, :Q]   # each key queried at most once (no copying earlier answers)
        seq[:, 2 * N:2 * N + 2 * Q:2], seq[:, 2 * N + 1:2 * N + 2 * Q:2] = keys[r, idx], keys[r, perm[r, perm[r, idx]]]
        mask[:, 2 * N:2 * N + 2 * Q:2] = True
    else:
        g = a.seg
        for b in range(B):
            st = rng.integers(0, L // 2 - g); pp = rng.integers(L // 2, L - g)
            seq[b, pp:pp + g] = seq[b, st:st + g]; mask[b, pp:pp + g - 1] = True
    return seq, mask


EV = make(512, np.random.default_rng(12345))


def rope(x, f):
    t = x.shape[-2]; ang = torch.arange(t, device=x.device).float()[:, None] * f[None, :]
    c, s = ang.cos(), ang.sin(); x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class Model(nn.Module):
    def __init__(s, V, d, nl, H):
        super().__init__(); s.d, s.nl, s.H, s.dk = d, nl, H, d // H
        f = int(8 * d / 3) // 8 * 8; s.shape = dict(q=(d, d), k=(d, d), v=(d, d), o=(d, d), gate=(f, d), up=(f, d), down=(d, f))
        s.emb = nn.Embedding(V, d); nn.init.normal_(s.emb.weight, std=d ** -0.5)
        s.n1, s.n2 = nn.Parameter(torch.ones(nl, d)), nn.Parameter(torch.ones(nl, d))
        s.qn, s.kn = nn.Parameter(torch.full((nl, s.dk), a.qkg)), nn.Parameter(torch.full((nl, s.dk), a.qkg))
        s.norm = nn.RMSNorm(d); s.head = nn.Linear(d, V, bias=False)
        s.register_buffer("freq", 1.0 / a.base ** (torch.arange(0, s.dk, 2).float() / s.dk))
        S = {}   # attention scalars (Adam, lr --slr)
        if a.vpl: S["vpp"] = nn.Parameter(torch.full((nl, H), a.vp))
        if a.hnorm: S["hg"] = nn.Parameter(torch.ones(nl, H, s.dk))
        if a.sink: S["sk"] = nn.Parameter(torch.zeros(nl, H))
        if a.kvs: S["kva"] = nn.Parameter(torch.tensor([1.0, 0.0, 1.0, 0.0]).repeat(nl, H, 1))
        if a.dconv: w0 = torch.zeros(nl, 3, d, a.dconv); w0[..., 0] = 1.0; S["dcw"] = nn.Parameter(w0)
        if a.smear: S["sm"] = nn.Parameter(torch.full((nl, H), 2.0))
        if a.js >= 2: S["jb"] = nn.Parameter(torch.full((nl, 1), 1.0 if a.js == 2 else 0.0)); S["jh"] = nn.Parameter(torch.zeros(nl, H))
        if a.gate: S["gw"] = nn.Parameter(torch.zeros(nl, H, d)); S["gb"] = nn.Parameter(torch.full((nl, H), 2.0))
        s.S = nn.ParameterDict(S)
        if a.qslope:
            sl0 = a.ab0 * 2.0 ** (-8.0 * (torch.arange(H).float() + 1) / H); S["qsb"] = nn.Parameter(torch.log(torch.expm1(sl0)).repeat(nl, 1)); S["qsw"] = nn.Parameter(torch.zeros(nl, H, d))
        if a.fox: S["fw"] = nn.Parameter(torch.zeros(nl, H, d)); S["fb"] = nn.Parameter(torch.full((nl, H), a.fb0))
        if a.lme == 2: S["lt"] = nn.Parameter(torch.full((nl, H), float(np.log(a.lt))))
        if a.kc == 2: S["kca"] = nn.Parameter(torch.full((nl, H), a.kca))
        if a.kc == 3: S["kmu"] = nn.Parameter(torch.zeros(nl, H, s.dk))
        if a.lsef: S["lu"] = nn.Parameter(torch.zeros(nl, H * (2 if a.lsef == 3 else 1), d))
        if a.alibi: S["al"] = nn.Parameter(torch.log(a.ab0 * 2.0 ** (-8.0 * (torch.arange(H).float() + 1) / H)).repeat(nl, 1))
        if a.dd and not a.dgfix: S["dg"] = nn.Parameter(torch.full((nl, H), a.dg0))
        if a.mix: MIX.init(S, a, nl, H, d)
        s.S = nn.ParameterDict(S)
        s.T = nn.ParameterDict()
        for nm, spec in (("k", a.kt), ("q", a.qt), ("v", a.vt)):
            for l in filter(None, spec.split(",")): s.T[f"{nm}{l}"] = nn.Parameter(torch.zeros(V, d))
        P = {}
        for m in MATS:
            o, i = s.shape[m]; w = torch.randn(nl, o, i) * i ** -0.5 * a.init
            if m in ("o", "down"): w.mul_(0.5)
            P[m] = nn.Parameter(w)
        s.P = nn.ParameterDict(P)
        s.off = False

    def attn(s, i, q, k, v, h):
        t = q.shape[-2]; plain = s.off or not (a.vp or a.vpl or a.sink or a.js or a.dd or a.alibi or a.sel or a.qslope or a.fox or a.lme or a.lsef)
        if plain: y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            lg = (q @ k.transpose(-1, -2)) * s.dk ** -0.5
            if a.dd:
                kn = F.normalize(s.kraw, dim=-1); cs = kn @ kn.transpose(-1, -2)
                if a.ddctl: cs = torch.eye(t, device=q.device).expand_as(cs)
                E = torch.exp(a.dtau * (cs - 1))   # (b,H,j,l)
                if a.dd == 2: E = E * torch.ones(t, t, device=q.device).triu()
                pen = torch.log(E.cumsum(-1).clamp_min(1e-6)).transpose(-1, -2)                               # (b,H,i,j)
                g = s.S["dg"][i][None, :, None, None] if not a.dgfix else a.dg0
                lg = lg - g * pen
            if a.alibi:
                dist = (torch.arange(t, device=q.device)[:, None] - torch.arange(t, device=q.device)[None, :]).clamp_min(0).float()
                lg = lg - torch.exp(s.S["al"][i])[None, :, None, None] * dist
            if a.qslope:
                dist = (torch.arange(t, device=q.device)[:, None] - torch.arange(t, device=q.device)[None, :]).clamp_min(0).float()
                sl = F.softplus(torch.einsum("btd,hd->bht", h, s.S["qsw"][i]) + s.S["qsb"][i][None, :, None])   # (b,H,t)
                lg = lg - (sl[..., :, None] if a.qslope == 1 else sl[..., None, :]) * dist
            if a.fox:
                lf = F.logsigmoid(torch.einsum("btd,hd->bht", h, s.S["fw"][i]) + s.S["fb"][i][None, :, None]).cumsum(-1)   # (b,H,t)
                lg = lg + lf[..., :, None] - lf[..., None, :]
            if a.sel:
                Sm = F.relu(lg[:, :1]) * torch.ones(t, t, device=q.device).tril(-1)          # S[l, j], j < l, from head 0
                if a.bos: Sm[..., 0] = 0
                Fm = torch.cumsum(Sm, -2); Fm = torch.cat([torch.zeros_like(Fm[..., :1, :]), Fm[..., :-1, :]], -2)   # F[i, j] = sum over j < l < i
                lg = lg - Fm
            lg = lg.masked_fill(~torch.ones(t, t, dtype=torch.bool, device=q.device).tril(), float("-inf"))
            if a.lsef:
                lse = torch.logsumexp(lg, -1)                                          # (b,H,t)
                feats = [lse - torch.log(torch.arange(1, t + 1, device=q.device).float())] if a.lsef != 2 else [lse]
                if a.lsef == 3: feats.append(lg.amax(-1))
                s.mfeat = torch.cat(feats, 1)                                          # (b,H*,t)
            if a.sink:
                sk = s.S["sk"][i][None, :, None, None].expand(lg.shape[0], -1, t, 1)
                A = torch.softmax(torch.cat([lg, sk], -1), -1)[..., :t]
            else: A = torch.softmax(lg, -1)
            if a.lme and not s.off:
                tau = torch.exp(s.S["lt"][i])[None, :, None, None] if a.lme == 2 else a.lt
                mx = v.amax(-2, keepdim=True).detach()
                y = mx + tau * torch.log((A @ torch.exp((v - mx) / tau)).clamp_min(1e-30))
            else: y = A @ v
            if a.js and not s.off:
                lse = torch.logsumexp(lg, -1)                                   # (b, H, t)
                z = lse if a.js == 1 else s.S["jb"][i][None, :, None] * lse + s.S["jh"][i][None, :, None]
                y = y * (torch.softmax(z, 1) * s.H)[..., None]
            if (a.vp or a.vpl) and not s.off:
                P_ = s.S["vpp"][i][None, :, None, None] if a.vpl else a.vp
                y = y * (A.pow(2).sum(-1, keepdim=True).clamp_min(1e-6)) ** (-P_ / 2)
        if getattr(s, "diag", False):
            lg_ = (q @ k.transpose(-1, -2)) * s.dk ** -0.5
            lg_ = lg_.masked_fill(~torch.ones(t, t, dtype=torch.bool, device=q.device).tril(), float("-inf")); A_ = torch.softmax(lg_, -1)
            dist = (torch.arange(t, device=q.device)[:, None] - torch.arange(t, device=q.device)[None, :]).clamp_min(0).float()
            ent = -(A_ * A_.clamp_min(1e-12).log()).sum(-1)
            s.stats.append(dict(layer=i, ent=ent.mean((0, 2)).tolist(), dist=(A_ * dist).sum(-1).mean((0, 2)).tolist(),
                                self=A_.diagonal(dim1=-2, dim2=-1).mean((0, 2)).tolist(), first=A_[..., 1:, 0].mean((0, 2)).tolist(), prev=A_.diagonal(-1, dim1=-2, dim2=-1).mean((0, 2)).tolist()))
        if a.hnorm and not s.off: y = F.rms_norm(y, (s.dk,)) * s.S["hg"][i][None, :, None, :]
        if a.gate and not s.off:
            g = torch.sigmoid(torch.einsum("btd,hd->bht", h, s.S["gw"][i]) + s.S["gb"][i][None, :, None]); y = y * g[..., None]
        return y

    def center(s, z, i):
        if a.kc == 3: return z - s.S["kmu"][i][None, :, None, :]
        t = z.shape[-2]; cnt = torch.arange(t, device=z.device).float().clamp_min(1)[:, None]
        prev = (z.cumsum(-2) - z) / cnt                          # mean over l < j (0 at j = 0)
        al = s.S["kca"][i][None, :, None, None] if a.kc == 2 else 1.0
        return z - al * prev

    def tab(s, nm, i, ids, z):
        key = f"{nm}{i}"
        return z if (key not in s.T or s.off) else z + s.T[key][ids]

    def forward(s, ids):
        x = s.emb(ids); b, t, d = x.shape; P = s.P
        sp = lambda z: z.view(b, t, s.H, s.dk).transpose(1, 2)
        for i in range(s.nl):
            h = F.rms_norm(x, (d,), s.n1[i])
            if a.mix and a.mix[i % len(a.mix)] != 's':
                x = x + F.linear(MIX.layer(s, a, i, h, a.mix[i % len(a.mix)], P), P["o"][i]); h = F.rms_norm(x, (d,), s.n2[i])
                x = x + F.linear(F.silu(F.linear(h, P["gate"][i])) * F.linear(h, P["up"][i]), P["down"][i]); continue
            if a.dconv and not s.off:
                def dc(z, c):   # z (b,t,d); out_t = sum_m w[m] * z_{t-m}
                    w = s.S["dcw"][i, c]; return sum(w[:, m] * F.pad(z, (0, 0, m, 0))[:, :t] for m in range(a.dconv))
                q, k = sp(dc(F.linear(h, P["q"][i]), 0)), sp(dc(F.linear(h, P["k"][i]), 1)); s.dcv = lambda z: dc(z, 2)
            else:
                q, k = sp(s.tab("q", i, ids, F.linear(h, P["q"][i]))), sp(s.tab("k", i, ids, F.linear(h, P["k"][i])))
            if a.kc and not s.off: k = s.center(k, i)
            if a.qc and not s.off: q = s.center(q, i)
            if a.qkn: q, k = F.rms_norm(q, (s.dk,), s.qn[i]), F.rms_norm(k, (s.dk,), s.kn[i])
            if a.smear: sg = torch.sigmoid(s.S["sm"][i])[None, :, None, None]; k = sg * k + (1 - sg) * F.pad(k, (0, 0, 1, 0))[:, :, :-1]
            s.kraw = k; q, k = rope(q, s.freq), rope(k, s.freq)
            vv = sp(s.dcv(F.linear(h, P["v"][i]))) if a.dconv and not s.off else sp(s.tab("v", i, ids, F.linear(h, P["v"][i])))
            if a.kvs and not s.off:
                ka = s.S["kva"][i][None, :, None, :]; sh = lambda z: F.pad(z, (0, 0, 1, 0))[:, :, :-1]
                k = ka[..., 0:1] * k + ka[..., 1:2] * sh(k); vv = ka[..., 2:3] * vv + ka[..., 3:4] * sh(vv)
            y = s.attn(i, q, k, vv, h)
            x = x + F.linear(y.transpose(1, 2).reshape(b, t, d), P["o"][i])
            if a.lsef and not s.off: x = x + torch.einsum("bht,hd->btd", s.mfeat, s.S["lu"][i])
            h = F.rms_norm(x, (d,), s.n2[i])
            x = x + F.linear(F.silu(F.linear(h, P["gate"][i])) * F.linear(h, P["up"][i]), P["down"][i])
        return s.head(s.norm(x))

    def body_params(s): return s.nl * sum(o * i for o, i in s.shape.values())


def newton_schulz(G, steps=5):
    A, B, C = 3.4445, -4.7750, 2.0315; X = G / (G.norm(dim=(-2, -1), keepdim=True) + 1e-7); tr = G.shape[-2] > G.shape[-1]
    if tr: X = X.transpose(-2, -1)
    for _ in range(steps): M = X @ X.transpose(-2, -1); X = A * X + (B * M + C * M @ M) @ X
    return X.transpose(-2, -1) if tr else X


class Muon:
    def __init__(s, model, lr): s.m, s.lr, s.sc = model, lr, 1.0; s.mom = {k: torch.zeros_like(w) for k, w in model.P.items()}

    @torch.no_grad()
    def step(s):
        for k, w in s.m.P.items():
            G = w.grad; mo = s.mom[k]; mo.mul_(a.beta).add_(G); u = G + a.beta * mo
            w.add_(newton_schulz(u), alpha=-s.lr * s.sc * max(1, w.shape[1] / w.shape[2]) ** 0.5)


def evaluate(m):
    m.eval(); x = torch.tensor(EV[0], device=DEV); mk = torch.tensor(EV[1], device=DEV)
    with torch.no_grad(): lg = m(x[:, :-1])
    y = x[:, 1:]; l = F.cross_entropy(lg[mk].float(), y[mk]).item(); acc = (lg[mk].argmax(-1) == y[mk]).float().mean().item(); m.train(); return l, acc


def main():
    m = Model(V, a.width, a.layers, a.heads).to(DEV)
    sc_ = [w for n, w in m.named_parameters() if n.startswith("S.")]
    tb = [w for n, w in m.named_parameters() if n.startswith("T.")]
    other = [w for n, w in m.named_parameters() if not n.startswith("P.") and not n.startswith("S.") and not n.startswith("T.")]
    groups = [dict(params=other, lr=a.elr)] + ([dict(params=sc_, lr=a.slr or a.elr)] if sc_ else []) + ([dict(params=tb, lr=a.tlr or a.elr)] if tb else [])
    muon = Muon(m, a.lr); adam = torch.optim.Adam(groups, betas=(a.ab1, a.ab2), eps=a.eps)
    f = torch.compile(m) if a.compile else m; t0 = time.time()
    rng = np.random.default_rng(a.seed); curve = []; s90 = None
    for step in range(a.steps):
        sc = min((step + 1) / max(1, a.warm), 1.0, (a.steps - step) / max(1e-9, a.cool * a.steps)); muon.sc = sc
        for g in adam.param_groups: g.setdefault("base", g["lr"]); g["lr"] = g["base"] * sc
        sq, mk = make(a.batch, rng); x = torch.tensor(sq, device=DEV); mk = torch.tensor(mk, device=DEV)
        mf = mk.float(); loss = (F.cross_entropy(f(x[:, :-1]).flatten(0, 1).float(), x[:, 1:].reshape(-1), reduction='none') * mf.reshape(-1)).sum() / mf.sum(); loss.backward()
        muon.step(); adam.step(); m.zero_grad(set_to_none=True)
        if (step + 1) % a.ee == 0:
            l, acc = evaluate(m); curve.append(round(acc, 3))
            if s90 is None and acc >= 0.9: s90 = step + 1
    v, acc = evaluate(m)
    res = dict(tag=a.tag, args=vars(a), body_params=m.body_params(), train_s=round(time.time() - t0, 2), val=v, acc=acc, steps90=s90, curve=curve)
    if a.posthoc:
        # Intervention at evaluation only; all forward passes share the same
        # trained checkpoint and the same fixed heldout batch.
        saved = (a.orq, a.orqE, a.orqP)
        posthoc = {}
        for label, do_erase, parity in (
            ("answer_write_only", 0, 1),
            ("answer_full_update", 1, 1),
            ("query_key_full_update", 1, 0),
            ("entire_query_phase", 1, 2),
        ):
            a.orq, a.orqE, a.orqP = 0.0, do_erase, parity
            iv, ia = evaluate(m)
            posthoc[label] = dict(loss=iv, accuracy=ia)
        a.orq, a.orqE, a.orqP = saved
        res["posthoc"] = posthoc

    if a.testoff: m.off = True; res["val_off"], res["acc_off"] = evaluate(m); m.off = False
    if a.vpl: res["vpp"] = [[round(float(z), 3) for z in row] for row in m.S["vpp"].detach().cpu()]
    if a.js >= 2: res["jb"] = [round(float(z), 3) for z in m.S["jb"].detach().cpu().flatten()]; res["jh"] = [[round(float(z), 2) for z in row] for row in m.S["jh"].detach().cpu()]
    if a.dd and not a.dgfix: res["dg"] = [[round(float(z), 3) for z in row] for row in m.S["dg"].detach().cpu()]
    if a.sink: res["sink"] = [[round(float(z), 2) for z in row] for row in m.S["sk"].detach().cpu()]
    if a.diag:
        m.diag, m.stats = True, []; m.eval()
        with torch.no_grad(): m(torch.tensor(EV[0][:64], device=DEV)[:, :-1])
        m.diag = False; res["stats"] = m.stats
        for st in m.stats: print("L%d ent %s dist %s self %s first %s prev %s" % (st["layer"], *[[round(z, 2) for z in st[k]] for k in ("ent", "dist", "self", "first", "prev")]))
    res["S_l"] = {k_: [round(z, 3) for z in w_.detach().cpu().flatten().tolist()] for k_, w_ in m.S.items() if k_.startswith("l") and w_.numel() <= 64}
    if res["S_l"]: print("S_l", res["S_l"])
    if getattr(m, "gst", None): res["gst"] = {str(k_): v_ for k_, v_ in m.gst.items()}; print("gate", res["gst"])
    if getattr(m, "psg", None): print("gate pair-vals / query-vals", [round(z, 3) for z in m.psg])
    if getattr(m, "gq", None): print("gate pairs/qkeys/qvals", [round(z, 3) for z in m.gq])
    with open(os.path.join(HERE, "results.jsonl"), "a") as fo: fo.write(json.dumps(res) + "\n")
    print(f"{a.tag or 'run'}: acc {acc:.3f}, steps90 {s90}, loss {v:.3f}, body {m.body_params() / 1e3:.0f}K, train {res['train_s']}s, curve {curve[::max(1, len(curve) // 8)]}", flush=True)


if __name__ == "__main__":
    main()

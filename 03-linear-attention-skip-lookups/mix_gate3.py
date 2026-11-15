"""Recurrent / linear-attention token mixers for lm.py and rc.py.
Layer codes in --mix (cycled over layers): s softmax attention (unchanged), d gated delta rule (Gated DeltaNet style),
D delta rule without decay, g gated linear attention (scalar decay per head, Mamba-2 style write).
All recurrent layers: q/k/v projections, causal depthwise short conv (kernel --sc, identity init) + SiLU, no RoPE,
L2-normalized q and k, write strength beta = sigmoid(w_b . h), decay log a = -softplus(w_a . h + b_a) (b_a init --ba),
per-head RMSNorm on the output. Recurrence is an explicit loop over time (runs are short)."""
import torch, torch.nn as nn, torch.nn.functional as F


def args(p):
    p.add_argument("--lnv", type=float, default=0.0, help="novelty-gated write: beta *= exp(-alpha * occupancy), alpha = lnv * exp(learned)")
    p.add_argument("--lnvc", type=int, default=0, help="control: occupancy without key similarity (count of decayed writes)")
    p.add_argument("--bb0", type=float, default=0.0)
    p.add_argument("--lorc", type=float, default=-1, help="oracle: multiply beta by this in the MQAR query phase (t >= 2N)")
    p.add_argument("--rbw", type=int, default=0, help="1: read before write (drop the i = t self term from the read)")
    p.add_argument("--lorcp", type=int, default=-1, help="oracle only at query-phase positions of this parity (0 key, 1 value)")
    p.add_argument("--lps", type=float, default=0.0, help="predicted-skip write: beta_t *= exp(-gamma relu(cos(read_{t-1}, v_t))), gamma = lps * exp(learned)")
    p.add_argument("--lpsc", type=int, default=0, help="control: compare read_{t-1} with v of another row (content-free)")
    p.add_argument("--lpsC", type=int, default=0, help="gate read uses only earlier chunks of this size (the inter-chunk read a chunked kernel already has)")
    p.add_argument("--lpsX", type=int, default=0, help="1: exact chunked gated-linear recurrence; gate uses S_chunkstart q_{t-1} (chunk size --lpsC)")
    p.add_argument("--lpsf", type=int, default=0, help="1: gamma fixed")
    p.add_argument("--lnvf", type=int, default=0, help="1: alpha fixed")


def init(S, a, nl, H, d):
    pass
    if a.sc: w0 = torch.zeros(nl, 3, d, a.sc); w0[..., 0] = 1.0; S["cw"] = nn.Parameter(w0)
    S["wb"] = nn.Parameter(torch.zeros(nl, H, d)); S["bb"] = nn.Parameter(torch.full((nl, H), a.bb0))
    S["wa"] = nn.Parameter(torch.zeros(nl, H, d)); S["ba"] = nn.Parameter(torch.full((nl, H), a.ba))
    S["on"] = nn.Parameter(torch.ones(nl, H, d // H))
    if a.lps: S["lpg"] = nn.Parameter(torch.zeros(nl, H))
    if a.lnv: S["lna"] = nn.Parameter(torch.zeros(nl, H))


def conv(s, a, z, i, c):
    t = z.shape[1]; w = s.S["cw"][i, c]
    return F.silu(sum(w[:, m] * F.pad(z, (0, 0, m, 0))[:, :t] for m in range(a.sc)))


def layer(s, a, i, h, code, P):
    b, t, d = h.shape; H, dk = s.H, s.dk
    q, k, v = F.linear(h, P["q"][i]), F.linear(h, P["k"][i]), F.linear(h, P["v"][i])
    if a.sc: q, k, v = conv(s, a, q, i, 0), conv(s, a, k, i, 1), conv(s, a, v, i, 2)
    sp = lambda z: z.view(b, t, H, dk).transpose(1, 2)            # (b,H,t,dk)
    q, k, v = F.normalize(sp(q), dim=-1), F.normalize(sp(k), dim=-1), sp(v)
    beta = torch.sigmoid(torch.einsum("btd,hd->bht", h, s.S["wb"][i]) + s.S["bb"][i][None, :, None])
    la = -F.softplus(torch.einsum("btd,hd->bht", h, s.S["wa"][i]) + s.S["ba"][i][None, :, None])
    if code == "D": la = torch.zeros_like(la)
    al = la.exp()
    hook = getattr(s, "mixhook", None)
    if getattr(a, "lpsX", 0) and code == "g":
        C = a.lpsC or 16; St = torch.zeros(b, H, dk, dk, device=h.device, dtype=h.dtype); outs = []
        qprev = F.pad(q, (0, 0, 1, 0))[..., :-1, :]
        gam = (a.lps * (torch.ones_like(s.S["lpg"][i]) if a.lpsf else s.S["lpg"][i].exp())) if a.lps else None
        for s0 in range(0, t, C):
            e0 = min(t, s0 + C); qc, kc, vc, bc, lc = q[..., s0:e0, :], k[..., s0:e0, :], v[..., s0:e0, :], beta[..., s0:e0], la[..., s0:e0].cumsum(-1)
            if gam is not None:
                r = torch.einsum("bhvk,bhtk->bhtv", St, qprev[..., s0:e0, :])
                vv = vc if not a.lpsc else vc.roll(1, 0)
                bc = bc * torch.exp(-gam[None, :, None] * F.relu(F.cosine_similarity(r, vv, dim=-1, eps=1e-6)))
            n = e0 - s0; low = torch.ones(n, n, dtype=torch.bool, device=h.device).tril()
            Gc = (lc[..., :, None] - lc[..., None, :]).masked_fill(~low, float("-inf")).exp()
            U = bc[..., None] * vc
            y = lc.exp()[..., None] * torch.einsum("bhvk,bhtk->bhtv", St, qc) + ((qc @ kc.transpose(-1, -2)) * Gc) @ U
            w = (lc[..., -1:] - lc).exp()
            St = lc[..., -1].exp()[..., None, None] * St + torch.einsum("bhtv,bhtk->bhvk", w[..., None] * U, kc)
            outs.append(y)
        y = F.rms_norm(torch.cat(outs, -2), (dk,)) * s.S["on"][i][None, :, None, :]
        return y.transpose(1, 2).reshape(b, t, d)
    if hook is None and not getattr(a, "loop", 0):
        c = la.cumsum(-1)                                          # (b,H,t)
        low = torch.ones(t, t, dtype=torch.bool, device=h.device).tril()
        G = (c[..., :, None] - c[..., None, :]).masked_fill(~low, float("-inf")).exp()   # gamma_t / gamma_i, i <= t
        if a.lnv:
            occ = ((beta[..., None, :] * ((k @ k.transpose(-1, -2)).pow(2) if not a.lnvc else 1.0 / k.shape[-1]) * G).tril(-1)).sum(-1)   # stored key energy along k_t
            alp = a.lnv * (torch.ones_like(s.S["lna"][i]) if a.lnvf else s.S["lna"][i].exp())
            beta = beta * torch.exp(-alp[None, :, None] * occ)
            s.occ = occ.mean().item()
        if a.lorc >= 0:
            mlt = torch.ones(t, device=h.device); par = getattr(a, "lorcp", -1)
            sel = torch.arange(t, device=h.device) >= 2 * a.pairs
            if par >= 0: sel = sel & (torch.arange(t, device=h.device) % 2 == par)
            beta = beta * torch.where(sel, a.lorc * mlt, mlt)
        if a.lnv and t > 2 * getattr(a, "pairs", t):
            gf = torch.exp(-alp[None, :, None] * occ); s.gq = (gf[..., :2 * a.pairs].mean().item(), gf[..., 2 * a.pairs::2].mean().item(), gf[..., 2 * a.pairs + 1::2].mean().item())
        if a.lps:   # pass 1: reads with ungated beta, then gate each write by how well the previous read predicted this value
            M0 = ((q @ k.transpose(-1, -2)) * G)
            if a.lpsC:
                ch = torch.arange(t, device=h.device) // a.lpsC
                M0 = M0 * (ch[None, :] < ch[:, None]).float()
            U0 = beta[..., None] * v
            if code in "dD":
                A0 = (beta[..., None] * (k @ k.transpose(-1, -2)) * G).tril(-1) + torch.eye(t, device=h.device)
                U0 = torch.linalg.solve_triangular(A0, U0, upper=False, unitriangular=True)
            y0 = M0 @ U0
            vv = v if not a.lpsc else v.roll(1, 0)
            cs = F.cosine_similarity(y0[..., :-1, :], vv[..., 1:, :], dim=-1)          # read at t-1 vs value at t
            cs = F.pad(cs, (1, 0))
            gam = a.lps * (torch.ones_like(s.S["lpg"][i]) if a.lpsf else s.S["lpg"][i].exp())
            beta = beta * torch.exp(-gam[None, :, None] * F.relu(cs))
            if t > 2 * getattr(a, "pairs", t): s.psg = (torch.exp(-gam[None, :, None] * F.relu(cs))[..., 1:2 * a.pairs:2].mean().item(), torch.exp(-gam[None, :, None] * F.relu(cs))[..., 2 * a.pairs + 1::2].mean().item())
        U = beta[..., None] * v
        if code in "dD":   # u_t = b_t (v_t - sum_{i<t} G_ti (k_i.k_t) u_i)
            A = (beta[..., None] * (k @ k.transpose(-1, -2)) * G).tril(-1) + torch.eye(t, device=h.device)
            U = torch.linalg.solve_triangular(A, U, upper=False, unitriangular=True)
        y = (((q @ k.transpose(-1, -2)) * G).tril(-1 if a.rbw else 0)) @ U
        y = F.rms_norm(y, (dk,)) * s.S["on"][i][None, :, None, :]
        return y.transpose(1, 2).reshape(b, t, d)
    St = torch.zeros(b, H, dk, dk, device=h.device, dtype=h.dtype); outs = []
    for j in range(t):
        kj, vj, qj, bj, aj = k[:, :, j], v[:, :, j], q[:, :, j], beta[:, :, j, None], al[:, :, j, None, None]
        if hook is not None: St, oj = hook(s, a, i, j, St, qj, kj, vj, bj, aj, code)
        else:
            St = aj * St
            if code in "dD": St = St + (bj * (vj - (St @ kj[..., None])[..., 0]))[..., None] * kj[..., None, :]
            else: St = St + (bj * vj)[..., None] * kj[..., None, :]
            oj = (St @ qj[..., None])[..., 0]
        outs.append(oj)
    y = torch.stack(outs, 2)                                      # (b,H,t,dk)
    y = F.rms_norm(y, (dk,)) * s.S["on"][i][None, :, None, :]
    return y.transpose(1, 2).reshape(b, t, d)

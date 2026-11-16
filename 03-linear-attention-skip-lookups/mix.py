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
    p.add_argument("--leb", type=float, default=-2.0, help="code e (GDN-2-style): learned erase gate sigmoid(w.h + b), b init")
    p.add_argument("--lpsE", type=int, default=0, help="delta family: 1 also scale the erase strength by the gate (default: write strength only)")
    p.add_argument("--lpsQ", type=int, default=0, help="1: gate only at MQAR query-phase positions (t >= 2N)")
    p.add_argument("--lpsR", type=int, default=0, help="read used by the gate: 0 S_c q_{t-1}; 1 S_c q_t (same token, control); 2 S_c k_t (the key being written)")
    p.add_argument("--lpsI", type=int, default=0, help="1: use mathematically self-consistent sequential predicted-skip reference, not a fast chunked kernel")
    p.add_argument("--lpsD", type=int, default=0, help="1: stop gradient through read/value cosine only")
    p.add_argument("--lpsM", type=float, default=0.0, help="cosine gate margin: relu(cos - margin)")
    p.add_argument("--localS", type=int, default=0, help="learned local token feature-only write suppression; does not read recurrent memory")
    p.add_argument("--phaseOracle", type=int, default=0, help="synthetic update-task-only oracle: suppress known query-answer token writes; not deployable")
    p.add_argument("--orq", type=float, default=-1, help="oracle: multiply write strength by this at MQAR query-phase value positions")


def init(S, a, nl, H, d):
    pass
    if a.sc: w0 = torch.zeros(nl, 3, d, a.sc); w0[..., 0] = 1.0; S["cw"] = nn.Parameter(w0)
    S["wb"] = nn.Parameter(torch.zeros(nl, H, d)); S["bb"] = nn.Parameter(torch.full((nl, H), a.bb0))
    S["wa"] = nn.Parameter(torch.zeros(nl, H, d)); S["ba"] = nn.Parameter(torch.full((nl, H), a.ba))
    S["on"] = nn.Parameter(torch.ones(nl, H, d // H))
    if a.lps: S["lpg"] = nn.Parameter(torch.zeros(nl, H))
    if getattr(a, "localS", 0):
        # Generous local-only control, O(H*d) additional parameters per layer.
        # Extra sigmoid compounds existing beta_t=sigmoid(w_b h_t+b_b),
        # allowing a role-specific suppression policy without any state read.
        S["localw"] = nn.Parameter(torch.zeros(nl, H, d))
        S["localb"] = nn.Parameter(torch.full((nl, H), 2.2))
    if a.lnv: S["lna"] = nn.Parameter(torch.zeros(nl, H))
    if "e" in a.mix: S["lwe"] = nn.Parameter(torch.zeros(nl, H, d)); S["lbe"] = nn.Parameter(torch.full((nl, H), a.leb))
    if "Z" in a.mix:
        # GDN-2 EQUATION-LEVEL CPU CONTROL only (not NVIDIA official layer).
        # b and w are independent per-channel key-side erase/value-side write
        # gates, and decay is dynamic per key channel. Same common toy block,
        # convolutions and q/k normalization across compared mixers.
        S["g2be"] = nn.Parameter(torch.zeros(nl, d, d))
        S["g2beb"] = nn.Parameter(torch.zeros(nl, d))
        S["g2wv"] = nn.Parameter(torch.zeros(nl, d, d))
        S["g2wvb"] = nn.Parameter(torch.zeros(nl, d))
        S["g2dt"] = nn.Parameter(torch.zeros(nl, d, d))
        S["g2dtb"] = nn.Parameter(torch.full((nl, d), -4.5))
        S["g2rate"] = nn.Parameter(torch.zeros(nl, H))


def conv(s, a, z, i, c):
    t = z.shape[1]; w = s.S["cw"][i, c]
    return F.silu(sum(w[:, m] * F.pad(z, (0, 0, m, 0))[:, :t] for m in range(a.sc)))


def layer(s, a, i, h, code, P):
    b, t, d = h.shape; H, dk = s.H, s.dk
    q, k, v = F.linear(h, P["q"][i]), F.linear(h, P["k"][i]), F.linear(h, P["v"][i])
    if a.sc: q, k, v = conv(s, a, q, i, 0), conv(s, a, k, i, 1), conv(s, a, v, i, 2)
    sp = lambda z: z.view(b, t, H, dk).transpose(1, 2)            # (b,H,t,dk)
    q, k, v = F.normalize(sp(q), dim=-1), F.normalize(sp(k), dim=-1), sp(v)
    if code == "Z":
        return gdn2_equation_reference(s, a, i, h, q, k, v)
    beta = torch.sigmoid(torch.einsum("btd,hd->bht", h, s.S["wb"][i]) + s.S["bb"][i][None, :, None])
    if getattr(a, "localS", 0):
        local_mult = torch.sigmoid(torch.einsum("btd,hd->bht", h, s.S["localw"][i]) + s.S["localb"][i][None, :, None])
        beta = beta * local_mult
    if getattr(a, "phaseOracle", 0):
        # This knows synthetic event ROLES from the task protocol. It is an
        # upper-bound causal control, not a general deployable baseline. Use
        # full configured length to keep the same mask on prefix-only AR calls.
        if a.task not in ("mqar_rewrite", "mqar_alias"):
            raise ValueError("--phaseOracle only for controlled two-stage tasks")
        beta = beta * phase_write_multiplier(a, t, h.device, beta.dtype)[None, None, :]
    la = -F.softplus(torch.einsum("btd,hd->bht", h, s.S["wa"][i]) + s.S["ba"][i][None, :, None])
    if code == "D": la = torch.zeros_like(la)
    s.erase = None
    if code == "e": s.erase = torch.sigmoid(torch.einsum("btd,hd->bht", h, s.S["lwe"][i]) + s.S["lbe"][i][None, :, None])
    al = la.exp()
    hook = getattr(s, "mixhook", None)
    if getattr(a, "lpsX", 0):
        return chunked(s, a, i, h, code, q, k, v, beta, la)
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
        if code in "dDe":   # u_t = b_t (v_t - sum_{i<t} G_ti (k_i.k_t) u_i)
            er_ = beta * (s.erase if code == "e" else 1.0)
            A = (er_[..., None] * (k @ k.transpose(-1, -2)) * G).tril(-1) + torch.eye(t, device=h.device)
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


def chunked(s, a, i, h, code, q, k, v, beta, la):
    """Exact chunked recurrence for g (gated linear), D (delta), d (gated delta), e (gated delta + learned erase gate).
    S_t = a_t S_{t-1} - e_t (a_t S_{t-1} k_t) k_t^T + b_t v_t k_t^T  (e_t = 0 for g, b_t for D/d, b_t * erase gate for e).
    Predicted-skip gate (--lps): b_t *= exp(-gamma relu(cos(S_c r_t, v_t))), S_c = state at the start of t's chunk, r_t = q_{t-1}."""
    if getattr(a, "lpsI", 0):
        from gate_exact import exact_skip
        return exact_skip(s, a, i, h, code, q, k, v, beta, la)
    b, H, t, dk = q.shape; C = a.lpsC or 16; delta = code in "dDe"
    er = None
    if delta: er = beta * (s.erase if code == "e" else 1.0)
    pos = torch.arange(t, device=h.device); N = getattr(a, "pairs", 0); qph = (pos >= 2 * N) if N and t > 2 * N else None
    if a.orq >= 0 and qph is not None:
        beta = beta * torch.where(qph & (pos % 2 == 1), a.orq, 1.0)
    rsrc = {0: F.pad(q, (0, 0, 1, 0))[..., :-1, :], 1: q, 2: k}[a.lpsR]
    gam = (a.lps * (torch.ones_like(s.S["lpg"][i]) if a.lpsf else s.S["lpg"][i].exp())) if a.lps else None
    St = torch.zeros(b, H, dk, dk, device=h.device, dtype=h.dtype); outs = []; gfs = []
    for s0 in range(0, t, C):
        e0 = min(t, s0 + C); n = e0 - s0
        qc, kc, vc, bc, lc = q[..., s0:e0, :], k[..., s0:e0, :], v[..., s0:e0, :], beta[..., s0:e0], la[..., s0:e0].cumsum(-1)
        ec = er[..., s0:e0] if delta else None
        if gam is not None:
            r = torch.einsum("bhvk,bhtk->bhtv", St, rsrc[..., s0:e0, :])
            vv = vc if not a.lpsc else vc.roll(1, 0)
            gf = torch.exp(-gam[None, :, None] * F.relu(F.cosine_similarity(r, vv, dim=-1, eps=1e-6)))
            if a.lpsQ and qph is not None: gf = torch.where(qph[s0:e0], gf, torch.ones_like(gf))
            bc = bc * gf
            if delta and a.lpsE: ec = ec * gf
            gfs.append(gf)
        low = torch.ones(n, n, dtype=torch.bool, device=h.device).tril()
        Gc = (lc[..., :, None] - lc[..., None, :]).masked_fill(~low, float("-inf")).exp()
        U = bc[..., None] * vc
        if delta:
            S0k = lc.exp()[..., None] * torch.einsum("bhvk,bhtk->bhtv", St, kc)
            A = (ec[..., None] * (kc @ kc.transpose(-1, -2)) * Gc).tril(-1) + torch.eye(n, device=h.device)
            U = torch.linalg.solve_triangular(A, U - ec[..., None] * S0k, upper=False, unitriangular=True)
        y = lc.exp()[..., None] * torch.einsum("bhvk,bhtk->bhtv", St, qc) + ((qc @ kc.transpose(-1, -2)) * Gc) @ U
        w = (lc[..., -1:] - lc).exp()
        St = lc[..., -1].exp()[..., None, None] * St + torch.einsum("bhtv,bhtk->bhvk", w[..., None] * U, kc)
        outs.append(y)
    if gfs and not s.training:
        gf = torch.cat(gfs, -1).detach()
        if not hasattr(s, "gst"): s.gst = {}
        st = dict(all=gf.mean().item(), sup=(gf < 0.5).float().mean().item(), gam=[round(z, 3) for z in gam.tolist()])
        if qph is not None:
            pv, qv, qk = gf[..., 1:2 * N:2], gf[..., 2 * N + 1::2], gf[..., 2 * N::2]
            st.update(pv=pv.mean().item(), qv=qv.mean().item(), qk=qk.mean().item(), pv_sup=(pv < 0.5).float().mean().item(), qv_sup=(qv < 0.5).float().mean().item())
        if qph is None: st.update(h1=gf[..., :t // 2].mean().item(), h2=gf[..., t // 2:].mean().item())
        s.gst[i] = {kk: (round(vv_, 4) if isinstance(vv_, float) else vv_) for kk, vv_ in st.items()}
    y = F.rms_norm(torch.cat(outs, -2), (dk,)) * s.S["on"][i][None, :, None, :]
    return y.transpose(1, 2).reshape(b, t, d_(h))



def gdn2_equation_reference(s, a, i, h, q, k, v):
    """Independent CPU reference of the *GDN-2 recurrence equation*.

    S_t = diag(alpha_t) S_(t-1) + k_t [w_t * v_t -
                                     (b_t * k_t)^T diag(alpha_t) S_(t-1)]^T.
    Transposes are used because this bed stores state in [value, key] form.

    This is NOT NVIDIA's full official GDN-2 implementation: the upstream
    optimized Triton kernels, beta-contrast safeguard, grouped heads,
    output SiLU gate, projection/SwiGLU recipe, and training configuration
    are not reproduced. It is a targeted mathematical expressivity control.
    """
    B, H, T, D = q.shape
    f = lambda weight, bias: F.linear(h, s.S[weight][i], s.S[bias][i]).view(B, T, H, D).transpose(1, 2)
    be = f("g2be", "g2beb").sigmoid()       # [B,H,T,dk] erase on KEY axis
    wv = f("g2wv", "g2wvb").sigmoid()       # [B,H,T,dv] write on VALUE axis
    raw_dt = f("g2dt", "g2dtb")
    decay = torch.exp(-s.S["g2rate"][i].exp()[None, :, None, None] * F.softplus(raw_dt))
    state = torch.zeros(B, H, D, D, device=h.device, dtype=h.dtype)
    outs = []
    for j in range(T):
        kj, qj, vj = k[:, :, j, :], q[:, :, j, :], v[:, :, j, :]
        state = state * decay[:, :, j, :].unsqueeze(-2)
        erased = (state @ (be[:, :, j, :] * kj).unsqueeze(-1)).squeeze(-1)
        new_value = wv[:, :, j, :] * vj - erased
        state = state + new_value.unsqueeze(-1) * kj.unsqueeze(-2)
        outs.append((state @ qj.unsqueeze(-1)).squeeze(-1))
    y = F.rms_norm(torch.stack(outs, -2), (D,)) * s.S["on"][i][None, :, None, :]
    return y.transpose(1, 2).reshape(B, T, h.shape[-1])


def phase_write_multiplier(a, prefix_length, device, dtype):
    """Task-role oracle mask, stable under shorter autoregressive prefixes.

    Returns zeros only at answer-value input positions of both query phases;
    second-stage externally supplied fact values are deliberately UNMASKED.
    """
    if a.task not in ("mqar_rewrite", "mqar_alias"):
        raise ValueError("Task-structure oracle cannot apply outside update tasks")
    from audit_update_task import layout
    _, _, _, bounds = layout(a.L, a.pairs)
    roles = torch.ones(prefix_length,device=device,dtype=dtype)
    for start, stop in ((bounds[1]+1,bounds[2]),(bounds[3]+1,bounds[4])):
        roles[start:min(stop,prefix_length):2]=0.0
    return roles

def d_(h): return h.shape[-1]

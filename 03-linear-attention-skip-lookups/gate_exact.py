"""Reference causal predicted-skip recurrence.

This deliberately sequential O(T d^2) implementation is a correctness oracle,
NOT an optimized chunkwise kernel. q/k/v are B,H,T,D; beta/la are B,H,T.
"""
import torch
import torch.nn.functional as F

def exact_skip(s, a, i, h, code, q, k, v, beta, la):
    B, H, T, D = q.shape
    delta = code in "dDe"
    erase = beta * (s.erase if code == "e" else 1.0) if delta else None
    pos = torch.arange(T, device=q.device)
    N = getattr(a, "pairs", 0)
    # This phase convention is valid only for the original two-phase MQAR.
    # mqar2 interleaves training facts and queries; do not mislabel its gates.
    qph = (pos >= 2 * N) if getattr(a, "task", "mqar") == "mqar" and N and T > 2 * N else None
    if getattr(a, "orq", -1.0) >= 0 and qph is not None:
        # Legacy --orq suppresses writes but (for delta) does NOT erase.
        beta = beta * torch.where(qph & (pos % 2 == 1), a.orq, 1.0)
    gam = (a.lps * (torch.ones_like(s.S["lpg"][i]) if a.lpsf
                    else s.S["lpg"][i].exp())) if a.lps else None
    state = torch.zeros(B, H, D, D, dtype=q.dtype, device=q.device)
    prev_q = torch.zeros_like(q[:, :, 0])
    outputs, factors = [], []
    for t in range(T):
        kt, qt, vt = k[:, :, t], q[:, :, t], v[:, :, t]
        bt, at = beta[:, :, t], la[:, :, t]
        gf = torch.ones_like(bt)
        if gam is not None:
            read_q = prev_q if a.lpsR == 0 else qt if a.lpsR == 1 else kt
            # state is the fully updated S_(t-1), prior to the current write.
            read = (state @ read_q.unsqueeze(-1)).squeeze(-1)
            compare_v = vt if not a.lpsc else v.roll(1, 0)[:, :, t]
            if getattr(a, "lpsD", 0):
                read, compare_v = read.detach(), compare_v.detach()
            cs = F.cosine_similarity(read, compare_v, dim=-1, eps=1e-6)
            gf = torch.exp(-gam[None, :] * F.relu(cs - getattr(a, "lpsM", 0.0)))
            if a.lpsQ and qph is not None:
                gf = torch.where(qph[t], gf, torch.ones_like(gf))
            factors.append(gf)
        bt = bt * gf
        et = erase[:, :, t] if delta else None
        if delta and a.lpsE:
            et = et * gf
        decayed_state = at.exp()[..., None, None] * state
        update = bt[..., None] * vt
        if delta:
            old_read = (decayed_state @ kt.unsqueeze(-1)).squeeze(-1)
            update = update - et[..., None] * old_read
        state = decayed_state + update.unsqueeze(-1) * kt.unsqueeze(-2)
        outputs.append((state @ qt.unsqueeze(-1)).squeeze(-1))
        prev_q = qt
    if factors and not s.training:
        gf = torch.stack(factors, -1).detach()
        if not hasattr(s, "gst"):
            s.gst = {}
        st = dict(all=gf.mean().item(), sup=(gf < .5).float().mean().item(),
                  gam=[round(z, 3) for z in gam.tolist()])
        if qph is not None:
            pv, qv, qk = gf[..., 1:2*N:2], gf[..., 2*N+1::2], gf[..., 2*N::2]
            st.update(pv=pv.mean().item(), qv=qv.mean().item(), qk=qk.mean().item(),
                      pv_sup=(pv < .5).float().mean().item(),
                      qv_sup=(qv < .5).float().mean().item())
        elif getattr(a, "task", "") == "mqar2" and N > 0 and N % 2 == 0:
            # Match the four blocks constructed by rc.make (for even N):
            # A facts [0,N), query A [N,N+2Q),
            # B facts [N+2Q,2N+2Q), query A+B [2N+2Q,T).
            Q = (T - 2 * N) // 4
            boundaries = (0, N, N + 2 * Q, 2 * N + 2 * Q, T)
            for name, start, end in zip(("factA", "queryA", "factB", "queryAB"),
                                        boundaries[:-1], boundaries[1:]):
                st[name+"_v"] = gf[..., start+1:end:2].mean().item()
                st[name+"_k"] = gf[..., start:end:2].mean().item()
                st[name+"_v_sup"] = (gf[..., start+1:end:2] < 0.5).float().mean().item()
        else:
            st.update(h1=gf[..., :T//2].mean().item(),
                      h2=gf[..., T//2:].mean().item())
        s.gst[i] = {name: round(value, 4) if isinstance(value, float) else value
                    for name, value in st.items()}
    y = F.rms_norm(torch.stack(outputs, -2), (D,)) * s.S["on"][i][None, :, None, :]
    return y.transpose(1, 2).reshape(B, T, h.shape[-1])

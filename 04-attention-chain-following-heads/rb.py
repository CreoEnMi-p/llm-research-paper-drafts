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
p.add_argument("--tag", default=""); p.add_argument("--init", type=float, default=0.25); p.add_argument("--diag", type=int, default=0); p.add_argument("--dconv", type=int, default=0, help="Primer MDHA (2109.08668): depthwise causal conv of kernel K on q, k, v channels, identity init"); p.add_argument("--kvs", type=int, default=0, help="KV shifting attention (2411.19574): k = a1 k_j + a2 k_{j-1}, v = b1 v_j + b2 v_{j-1}, per-head learned"); p.add_argument("--task", default="mqar"); p.add_argument("--hops", type=int, default=2, help="hopn task: number of chained lookups"); p.add_argument("--eqh", type=int, default=0, help="match-by-identity heads: first n heads per layer use k_j = q_{j-1} (same projection, previous token), no RoPE"); p.add_argument("--eqt", type=float, default=0.0, help=">0: learned per-layer temperature on identity heads (query scaled), init this"); p.add_argument("--eqctl", type=int, default=0, help="1: eq heads use k_j = k_{j-1} from W_k (untied control); 2: tied but keep RoPE"); p.add_argument("--bmix", type=int, default=0, help="hop2: block B lists pairs as (z, y), so hop 2 is backward: find y, answer the token before it"); p.add_argument("--eqb", type=int, default=0, help="last n identity heads are backward: key = own query q_j (self-match), value = v_{j-1}, diagonal masked"); p.add_argument("--eqbctl", type=int, default=0, help="backward-head controls: 1 no diagonal mask, 2 untied key (W_k, no RoPE)"); p.add_argument("--rev", type=int, default=0, help="hopn: blocks in reverse chain order (last hop first)"); p.add_argument("--gap", type=int, default=0, help="extra eval: insert this many filler tokens (id 0) between the data blocks and the queries (train has none)"); p.add_argument("--mta", type=int, default=0, help="Multi-Token Attention (2504.00927) key-query conv: c x c causal depthwise conv over pre-softmax logits of every head, identity init"); p.add_argument("--eqn2", type=float, default=0.0, help=">0: identity heads also score the previous pair, logit += g * q_{i-1}.q_{j-2} (soft bigram match), g learned per layer, init this"); p.add_argument("--ngh", type=int, default=0, help="Akyurek 2401.12973 n-gram heads: first n heads per layer attend uniformly to j with exact token match x_{j-1}=x_i (and x_{j-2}=x_{i-1} if --ngo 2), value v_j"); p.add_argument("--ngo", type=int, default=1); p.add_argument("--ptr", type=int, default=0, help="pointer stream: identity-head queries come from a pointer P (P_0 = embeddings, P_{l+1} = A_eq0 @ embeddings) and keys from previous-position embeddings, at every layer; 2: control, P stays = embeddings"); p.add_argument("--h1", type=float, default=0.0, help="hop2: share of training rows whose queries are one-hop (x->y or y->z); eval stays two-hop"); p.add_argument("--cope", type=int, default=0, help="CoPE contextual positions (2405.18719): number of position vectors"); p.add_argument("--occ", type=int, default=0, help="occurrence-index embedding added to token embeddings (2 = hashed-class control)"); p.add_argument("--nrep", type=int, default=3, help="nth: occurrences per key"); p.add_argument("--km", type=int, default=3, help="nkey: tokens per key"); p.add_argument("--nc", type=int, default=1, help="nkey: corrupted key tokens per query"); p.add_argument("--nperm", type=int, default=0, help="nkey: 1 = query lists the key tokens in random order"); p.add_argument("--pairs", type=int, default=16); p.add_argument("--tv", type=int, default=256, help="synthetic vocabulary"); p.add_argument("--seg", type=int, default=16); p.add_argument("--ee", type=int, default=25); p.add_argument("--rep", type=float, default=0.0, help="share of train rows whose second half repeats the first half; also reports val_rep (loss on the repeated half)"); p.add_argument("--qkg", type=float, default=1.0); p.add_argument("--smear", type=int, default=0, help="smeared keys (Olsson 2022): k_j = s*k_j + (1-s)*k_{j-1}, per-head learned s"); p.add_argument("--lpos", type=int, default=0); p.add_argument("--vrep", type=int, default=0); p.add_argument("--shared", type=int, default=0); p.add_argument("--vrev", type=int, default=0); p.add_argument("--delim", type=int, default=0); p.add_argument("--roleor", type=int, default=0); p.add_argument("--conjor", type=int, default=0, help="conj oracle: fused tuple token"); p.add_argument("--ca", type=int, default=4, help="conj: alphabet size per key position"); p.add_argument("--mtar", type=int, default=0, help="MTA variants: 1 = conv over ReLU(logit - b) (rectified per-offset matches), 2 = conv over 4*sigmoid(4(logit - b)) (match detectors); b learned per head"); p.add_argument("--dualk", type=int, default=0, help="dual keys: score = logaddexp(RoPE score, unrotated score + per-head bias); 2 = control, plain average of the two scores")
a = p.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(0)
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
        if a.shared: vals = np.take_along_axis(keys, rng.integers(0, N, (B, N)), 1)   # role confusion: every value is also some record's key
        seq[:, 0:2 * N:2], seq[:, 1:2 * N:2] = keys, vals
        if a.shared and a.delim:   # same records as 'k v ;' triples with a fixed delimiter token V-1
            seq[:] = rng.integers(1, V, seq.shape); seq[:, 0:3 * N:3], seq[:, 1:3 * N:3], seq[:, 2:3 * N:3] = keys, vals, V - 1
            Q = (L - 3 * N) // 3; idx = rng.integers(0, N, (B, Q)); r = np.arange(B)[:, None]; o = 3 * N
            seq[:, o:o + 3 * Q:3], seq[:, o + 1:o + 3 * Q:3], seq[:, o + 2:o + 3 * Q:3] = keys[r, idx], vals[r, idx], V - 1
            mask[:, o:o + 3 * Q:3] = True
        if a.vrev:   # reverse recall: values distinct; query = a value, answer = its key (--rev 2: half the queries forward, half reverse, each query prefixed by a direction token)
            vals = np.argsort(rng.random((B, V - Kn - 3)), 1)[:, :N] + Kn + 1; seq[:, 1:2 * N:2] = vals; r = np.arange(B)[:, None]
            if a.vrev == 1:
                Q = (L - 2 * N) // 2; idx = rng.integers(0, N, (B, Q)); seq[:, 2 * N:2 * N + 2 * Q:2], seq[:, 2 * N + 1:2 * N + 2 * Q:2] = vals[r, idx], keys[r, idx]; mask[:, 2 * N:2 * N + 2 * Q:2] = True
            else:
                Q = (L - 2 * N) // 3; idx = rng.integers(0, N, (B, Q)); d_ = rng.integers(0, 2, (B, Q)) * (a.vrev == 2); o = 2 * N
                seq[:, o:o + 3 * Q:3] = V - 2 + d_; seq[:, o + 1:o + 3 * Q:3] = np.where(d_ == 1, vals[r, idx], keys[r, idx]); seq[:, o + 2:o + 3 * Q:3] = np.where(d_ == 1, keys[r, idx], vals[r, idx]); mask[:, o + 1:o + 3 * Q:3] = True
        elif not (a.shared and a.delim):
            Q = (L - 2 * N) // 2; idx = rng.integers(0, N, (B, Q)); r = np.arange(B)[:, None]
            seq[:, 2 * N:2 * N + 2 * Q:2], seq[:, 2 * N + 1:2 * N + 2 * Q:2] = keys[r, idx], vals[r, idx]
            mask[:, 2 * N:2 * N + 2 * Q:2] = True
    elif a.task == "nth":   # each of N keys occurs R times (random order) with fresh values; query = (ordinal token o, key k), answer = value after the o-th occurrence of k
        N, R_ = a.pairs, a.nrep; Kn = V // 2 - 1; r = np.arange(B)[:, None]; assert 2 * N * R_ + 3 < L
        keys = np.argsort(rng.random((B, Kn)), 1)[:, :N] + 1; order = np.argsort(rng.random((B, N * R_)), 1) % N
        vals = rng.integers(Kn + 1, V - R_, (B, N * R_)); seq[:, 0:2 * N * R_:2] = keys[r, order]; seq[:, 1:2 * N * R_:2] = vals
        occ = np.zeros((B, N * R_), np.int64)
        for j in range(N * R_): occ[:, j] = (order[:, :j] == order[:, j:j + 1]).sum(1)
        o = 2 * N * R_; Q = (L - o) // 3
        for qn in range(Q):
            kk = rng.integers(0, N, B); oo = rng.integers(0, R_, B); hit = (order == kk[:, None]) & (occ == oo[:, None]); jj = hit.argmax(1)
            p0 = o + 3 * qn; seq[:, p0] = V - R_ + oo; seq[:, p0 + 1] = keys[np.arange(B), kk]; seq[:, p0 + 2] = vals[np.arange(B), jj]; mask[:, p0 + 1] = True
    elif a.task == "last":   # overwrite recall: each of N keys is reassigned R times (random order); query = key, answer = its latest value (--lpos 1: value before the latest)
        N, R_ = a.pairs, a.nrep; Kn = V // 2 - 1; r = np.arange(B)[:, None]; assert 2 * N * R_ + 2 < L
        keys = np.argsort(rng.random((B, Kn)), 1)[:, :N] + 1; order = np.argsort(rng.random((B, N * R_)), 1) % N
        if a.vrep:   # varying repeat counts: each key at least twice, the rest random
            base = np.concatenate([np.tile(np.arange(N), (B, 2)), rng.integers(0, N, (B, N * R_ - 2 * N))], 1)
            order = np.take_along_axis(base, np.argsort(rng.random(base.shape), 1), 1)
        vals = rng.integers(Kn + 1, V, (B, N * R_)); seq[:, 0:2 * N * R_:2] = keys[r, order]; seq[:, 1:2 * N * R_:2] = vals
        o = 2 * N * R_; Q = min(N, (L - o) // 2); idx = np.argsort(rng.random((B, N)), 1)[:, :Q]
        for qn in range(Q):
            kk = idx[:, qn]; hit = order == kk[:, None]; pos_ = np.where(hit, np.arange(N * R_)[None, :], -1)
            jj = np.sort(pos_, 1)[:, -1 - a.lpos]
            seq[:, o + 2 * qn] = keys[np.arange(B), kk]; seq[:, o + 2 * qn + 1] = vals[np.arange(B), jj]; mask[:, o + 2 * qn] = True
    elif a.task == "conj":   # conjunctive keys: N records (a_1..a_m, v) with each a_c drawn from a small alphabet of --ca symbols (so every single token is shared by many records, only the full tuple is unique); query = a full key, answer v
        N, m_, A_ = a.pairs, a.km, a.ca; W_ = m_ + 1; assert A_ ** m_ >= N and W_ * N + W_ < L and m_ * A_ < V // 2; r = np.arange(B)[:, None]
        codes = np.argsort(rng.random((B, A_ ** m_)), 1)[:, :N]   # distinct tuples per row
        keys = np.stack([(codes // A_ ** c) % A_ + 1 + c * A_ for c in range(m_)], -1); vals = rng.integers(V // 2, V, (B, N))
        if a.conjor: keys[:, :, -1] = codes + 1 + m_ * A_; assert m_ * A_ + A_ ** m_ < V // 2   # oracle: last key token names the whole tuple
        for c in range(m_): seq[:, c:W_ * N:W_] = keys[:, :, c]
        seq[:, m_:W_ * N:W_] = vals
        o = W_ * N; Q = (L - o) // W_; idx = rng.integers(0, N, (B, Q)); qk = keys[r, idx]
        for c in range(m_): seq[:, o + c:o + W_ * Q:W_] = qk[:, :, c]
        seq[:, o + m_:o + W_ * Q:W_] = vals[r, idx]; mask[:, o + m_ - 1:o + W_ * Q:W_] = True
    elif a.task == "nkey":   # noisy-key recall: N records (k_1..k_m, v); query = key with --nc of its m tokens replaced by random key tokens, answer v
        N, m_ = a.pairs, a.km; Kn = V // 2 - 1; W_ = m_ + 1; assert W_ * N + W_ < L; r = np.arange(B)[:, None]
        keys = rng.integers(1, Kn + 1, (B, N, m_)); vals = rng.integers(Kn + 1, V, (B, N))
        for c in range(m_): seq[:, c:W_ * N:W_] = keys[:, :, c]
        seq[:, m_:W_ * N:W_] = vals
        o = W_ * N; Q = (L - o) // W_; idx = rng.integers(0, N, (B, Q)); qk = keys[r, idx].copy()
        for _ in range(a.nc):
            pos = rng.integers(0, m_, (B, Q)); qk[r, np.arange(Q)[None, :], pos] = rng.integers(1, Kn + 1, (B, Q))
        if a.nperm: qk = np.take_along_axis(qk, np.argsort(rng.random(qk.shape), -1), -1)
        for c in range(m_): seq[:, o + c:o + W_ * Q:W_] = qk[:, :, c]
        seq[:, o + m_:o + W_ * Q:W_] = vals[r, idx]; mask[:, o + m_ - 1:o + W_ * Q:W_] = True
    elif a.task == "hop2":   # block A: N pairs x -> y (x in K1, y in K2); block B: same y's in new order -> z (z in K3); query x, answer z
        N = a.pairs; T3 = (V - 1) // 3; assert T3 >= N and 4 * N + 2 < L; r = np.arange(B)[:, None]
        pick = lambda: np.argsort(rng.random((B, T3)), 1)[:, :N] + 1
        x, y, z = pick(), pick() + T3, pick() + 2 * T3; ob = np.argsort(rng.random((B, N)), 1)
        seq[:, 0:2 * N:2], seq[:, 1:2 * N:2] = x, y
        if a.bmix: seq[:, 2 * N:4 * N:2], seq[:, 2 * N + 1:4 * N:2] = z[r, ob], y[r, ob]
        else: seq[:, 2 * N:4 * N:2], seq[:, 2 * N + 1:4 * N:2] = y[r, ob], z[r, ob]
        Q = min(N, (L - 4 * N) // 2); idx = np.argsort(rng.random((B, N)), 1)[:, :Q]   # each x queried once
        qs, ans = x[r, idx], z[r, idx]
        if H1 > 0:
            md = rng.random(B); m1 = md < H1 / 2; m2 = (md >= H1 / 2) & (md < H1)
            qs[m1], ans[m1] = x[r, idx][m1], y[r, idx][m1]; qs[m2], ans[m2] = y[r, idx][m2], z[r, idx][m2]
        seq[:, 4 * N:4 * N + 2 * Q:2], seq[:, 4 * N + 1:4 * N + 2 * Q:2] = qs, ans
        mask[:, 4 * N:4 * N + 2 * Q:2] = True
    elif a.task == "hopn":   # hops blocks; block b maps range b -> range b+1 (each block in its own random order); query x in range 0, answer after all hops
        N, Hn = a.pairs, a.hops; Tr = (V - 1) // (Hn + 1); assert Tr >= N and 2 * N * Hn + 2 < L; r = np.arange(B)[:, None]
        ch = [np.argsort(rng.random((B, Tr)), 1)[:, :N] + 1 + c * Tr for c in range(Hn + 1)]   # ch[c][:, n] = chain n at stage c
        for c in range(Hn):
            ob = np.argsort(rng.random((B, N)), 1); o = 2 * N * ((Hn - 1 - c) if a.rev else c)
            seq[:, o:o + 2 * N:2], seq[:, o + 1:o + 2 * N:2] = ch[c][r, ob], ch[c + 1][r, ob]
        o = 2 * N * Hn; Q = min(N, (L - o) // 2); idx = np.argsort(rng.random((B, N)), 1)[:, :Q]
        seq[:, o:o + 2 * Q:2], seq[:, o + 1:o + 2 * Q:2] = ch[0][r, idx], ch[Hn][r, idx]
        mask[:, o:o + 2 * Q:2] = True
    else:
        g = a.seg
        for b in range(B):
            st = rng.integers(0, L // 2 - g); pp = rng.integers(L // 2, L - g)
            seq[b, pp:pp + g] = seq[b, st:st + g]; mask[b, pp:pp + g - 1] = True
    if GAP:
        e = 2 * a.pairs if a.task == "mqar" else 4 * a.pairs
        seq = np.concatenate([seq[:, :e], np.zeros((B, GAP), seq.dtype), seq[:, e:]], 1); mask = np.concatenate([mask[:, :e], np.zeros((B, GAP), bool), mask[:, e:]], 1)
    return seq, mask


H1 = 0.0; GAP = 0; EV = make(512, np.random.default_rng(12345)); H1 = a.h1
if a.gap: GAP = a.gap; EVG = make(512, np.random.default_rng(12345)); GAP = 0


POS = None   # oracle: optional (b, t) float position ids replacing 0..t-1 in RoPE and ALiBi


def rope(x, f):
    t = x.shape[-2]; ang = (torch.arange(t, device=x.device).float() if POS is None else POS[:, None, :])[..., None] * f
    c, s = ang.cos(), ang.sin(); x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class Model(nn.Module):
    def __init__(s, V, d, nl, H):
        super().__init__(); s.d, s.nl, s.H, s.dk = d, nl, H, d // H
        f = int(8 * d / 3) // 8 * 8; s.shape = dict(q=(d, d), k=(d, d), v=(d, d), o=(d, d), gate=(f, d), up=(f, d), down=(d, f))
        s.emb = nn.Embedding(V, d); nn.init.normal_(s.emb.weight, std=d ** -0.5)
        if a.roleor: s.role = nn.Embedding(2, d); nn.init.normal_(s.role.weight, std=d ** -0.5)
        if a.occ: s.oe = nn.Embedding(16, d); nn.init.normal_(s.oe.weight, std=d ** -0.5)
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
        if a.mta: w_ = torch.zeros(nl, H, 1, a.mta, a.mta); w_[..., -1, -1] = 1.0; S["mta"] = nn.Parameter(w_)
        if a.eqn2: S["eqg"] = nn.Parameter(torch.full((nl,), a.eqn2))
        if a.ptr == 3: S["pg"] = nn.Parameter(torch.full((nl,), 2.0))
        if a.ptr == 8: S["pg"] = nn.Parameter(torch.zeros(nl))
        if a.cope: S["cope"] = nn.Parameter(torch.randn(nl, a.cope, s.dk) * 0.02)
        if a.dualk: S["dkb"] = nn.Parameter(torch.zeros(nl, H))
        if a.mtar: S["mtb"] = nn.Parameter(torch.zeros(nl, H))
        if a.eqh and a.eqt > 0: S["eqt"] = nn.Parameter(torch.full((nl,), a.eqt))
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
        t = q.shape[-2]; plain = s.off or not (a.vp or a.vpl or a.sink or a.js or a.dd or a.alibi or a.sel or a.qslope or a.fox or a.lme or a.lsef or a.eqn2 or a.eqb or a.mta or a.cope or a.dualk)
        if plain and getattr(s, "km", None) is not None:   # oracle: hide filler keys (token 0) from attention, self kept
            am = torch.ones(t, t, dtype=torch.bool, device=q.device).tril() & (s.km[:, None, None, :] | torch.eye(t, dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        elif plain: y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            lg = (q @ k.transpose(-1, -2)) * s.dk ** -0.5
            if a.dualk and not s.off and q.shape[1] == s.qraw.shape[1]:
                lu = (s.qraw @ s.kraw.transpose(-1, -2)) * s.dk ** -0.5 + s.S["dkb"][i][None, :, None, None]
                lg = torch.logaddexp(lg, lu) if a.dualk == 1 else 0.5 * (lg + lu)
            if a.eqn2 and not s.off: lg = torch.cat([lg[:, :a.eqh] + s.xl, lg[:, a.eqh:]], 1)
            if a.cope and not s.off:   # CoPE (2405.18719): gates sigmoid(q.k), position p_ij = sum_{m=j..i} g_im, logit += q_i . E[p_ij] (interpolated)
                cm = torch.ones(t, t, dtype=torch.bool, device=lg.device).tril(); gt = torch.sigmoid(lg).masked_fill(~cm, 0.0)
                pp = gt.flip(-1).cumsum(-1).flip(-1).clamp_max(a.cope - 1); z = q @ s.S["cope"][i].transpose(-1, -2) * s.dk ** -0.5   # (b,H,t,P)
                lo = pp.floor().long().clamp_max(a.cope - 2); w_ = pp - lo; lg = lg + torch.gather(z, -1, lo) * (1 - w_) + torch.gather(z, -1, lo + 1) * w_
            if a.mta and not s.off: c_ = a.mta; lm = lg.masked_fill(~torch.ones(t, t, dtype=torch.bool, device=lg.device).tril(), 0.0); lm = F.relu(lm - s.S["mtb"][i][None, :, None, None]) if a.mtar == 1 else (torch.sigmoid(4 * (lm - s.S["mtb"][i][None, :, None, None])) * 4 if a.mtar == 2 else lm); lg = F.conv2d(F.pad(lm, (c_ - 1, 0, c_ - 1, 0)), s.S["mta"][i], groups=lg.shape[1])
            if a.eqb and a.eqbctl != 1 and not s.off: dm = torch.eye(t, device=lg.device, dtype=torch.bool) & (torch.arange(lg.shape[1], device=lg.device) >= a.eqh - a.eqb)[:, None, None] & (torch.arange(lg.shape[1], device=lg.device) < a.eqh)[:, None, None]; lg = lg.masked_fill(dm, -1e4)
            if a.dd:
                kn = F.normalize(s.kraw, dim=-1); cs = kn @ kn.transpose(-1, -2)
                if a.ddctl: cs = torch.eye(t, device=q.device).expand_as(cs)
                E = torch.exp(a.dtau * (cs - 1))   # (b,H,j,l)
                if a.dd == 2: E = E * torch.ones(t, t, device=q.device).triu()
                pen = torch.log(E.cumsum(-1).clamp_min(1e-6)).transpose(-1, -2)                               # (b,H,i,j)
                g = s.S["dg"][i][None, :, None, None] if not a.dgfix else a.dg0
                lg = lg - g * pen
            if a.alibi:
                pz = torch.arange(t, device=q.device).float() if POS is None else POS[:, None, :]
                dist = (pz[..., :, None] - pz[..., None, :]).clamp_min(0)
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
            if getattr(s, "km", None) is not None: lg = lg.masked_fill(~(s.km[:, None, None, :] | torch.eye(t, dtype=torch.bool, device=q.device)), float("-inf"))
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
        if a.roleor: x = x + s.role(torch.arange(t, device=ids.device) % 2)[None]   # oracle: key/value role embedding (parity)
        if a.occ:   # occurrence-index embedding: how many earlier positions hold the same token (1 = exact counts; 2 = control, counts over a random token hash with 8 classes)
            cid = ids if a.occ == 1 else (ids * 2654435761) % 8
            oc = (cid[:, :, None] == cid[:, None, :]).tril(-1).sum(-1).clamp_max(15) if a.occ < 3 else (torch.arange(t, device=ids.device) * 16 // t).expand_as(ids)
            x = x + s.oe(oc)
        e0 = x; ptr = x
        sp = lambda z: z.view(b, t, s.H, s.dk).transpose(1, 2)
        for i in range(s.nl):
            h = F.rms_norm(x, (d,), s.n1[i])
            if a.dconv and not s.off:
                def dc(z, c):   # z (b,t,d); out_t = sum_m w[m] * z_{t-m}
                    zp = F.pad(z, (0, 0, a.dconv - 1, 0)); return (torch.stack([zp[:, a.dconv - 1 - m:a.dconv - 1 - m + t] for m in range(a.dconv)], -1) * s.S["dcw"][i, c]).sum(-1)
                q, k = sp(dc(F.linear(h, P["q"][i]), 0)), sp(dc(F.linear(h, P["k"][i]), 1)); s.dcv = lambda z: dc(z, 2)
            else:
                q, k = sp(s.tab("q", i, ids, F.linear(h, P["q"][i]))), sp(s.tab("k", i, ids, F.linear(h, P["k"][i])))
            if a.kc and not s.off: k = s.center(k, i)
            if a.qc and not s.off: q = s.center(q, i)
            if a.qkn: q, k = F.rms_norm(q, (s.dk,), s.qn[i]), F.rms_norm(k, (s.dk,), s.kn[i])
            if a.smear: sg = torch.sigmoid(s.S["sm"][i])[None, :, None, None]; k = sg * k + (1 - sg) * F.pad(k, (0, 0, 1, 0))[:, :, :-1]
            if a.eqh and not s.off:
                n_ = a.eqh; shq = F.pad(q[:, :n_], (0, 0, 1, 0))[:, :, :-1] if a.eqctl != 1 else F.pad(k[:, :n_], (0, 0, 1, 0))[:, :, :-1]
                qe, ke = q[:, :n_], shq
                if a.eqb: ke = torch.cat([ke[:, :n_ - a.eqb], (k if a.eqbctl == 2 else q)[:, n_ - a.eqb:n_]], 1)
                if a.ptr:
                    pq = lambda z: F.rms_norm(sp(F.linear(F.rms_norm(z, (d,), s.n1[i]), P["q"][i]))[:, :n_], (s.dk,), s.qn[i])
                    qp, kp = pq(ptr)[:, :1], F.pad(pq(e0), (0, 0, 1, 0))[:, :1, :-1]
                    if a.ptr == 3: g_ = torch.sigmoid(s.S["pg"][i]); qp, kp = g_ * qp + (1 - g_) * qe[:, :1], g_ * kp + (1 - g_) * ke[:, :1]
                    if a.ptr in (6, 7, 8):   # predecessor doubling: head 0 query = pointer, key = own token (self-match, diagonal masked); head 1 readout: query = token, key = pointer of previous position
                        kp = pq(e0)[:, :1]; qr, kr = pq(e0)[:, 1:2], F.pad(pq(ptr), (0, 0, 1, 0))[:, 1:2, :-1]
                        qe = torch.cat([qp, qr, qe[:, 2:]], 1); ke = torch.cat([kp, kr, ke[:, 2:]], 1)
                    else: qe = torch.cat([qp, qe[:, 1:]], 1); ke = torch.cat([kp, ke[:, 1:]], 1)
                if a.eqt > 0: qe = qe * s.S["eqt"][i]
                if a.eqn2: sh_ = lambda z: F.pad(z, (0, 0, 1, 0))[:, :, :-1]; s.xl = s.S["eqg"][i] * (sh_(qe) @ sh_(ke).transpose(-1, -2)) * s.dk ** -0.5
                s.ptr_old = ptr
                if a.ptr in (1, 3, 4, 5):
                    lg0 = (qe[:, 0] @ ke[:, 0].transpose(-1, -2)) * s.dk ** -0.5
                    lg0 = lg0.masked_fill(~torch.ones(t, t, dtype=torch.bool, device=lg0.device).tril(), float("-inf"))
                    ptr = torch.softmax(lg0, -1) @ (ptr if a.ptr in (4, 5) else e0)
                if a.ptr in (6, 7, 8):
                    lg0 = (qe[:, 0] @ ke[:, 0].transpose(-1, -2)) * s.dk ** -0.5
                    lg0 = lg0.masked_fill(torch.eye(t, dtype=torch.bool, device=lg0.device), -1e4).masked_fill(~torch.ones(t, t, dtype=torch.bool, device=lg0.device).tril(), float("-inf"))
                    ptr = torch.softmax(lg0, -1) @ F.pad(ptr if a.ptr == 6 else (e0 if a.ptr == 7 else torch.sigmoid(s.S["pg"][i]) * ptr + (1 - torch.sigmoid(s.S["pg"][i])) * e0), (0, 0, 1, 0))[:, :-1]
            s.kraw = k; s.qraw = q; q, k = rope(q, s.freq), rope(k, s.freq)
            if a.eqh and not s.off:
                if a.eqctl == 2: qe, ke = rope(qe, s.freq), rope(ke, s.freq)
                q = torch.cat([qe, q[:, n_:]], 1); k = torch.cat([ke, k[:, n_:]], 1)
            vv = sp(s.dcv(F.linear(h, P["v"][i]))) if a.dconv and not s.off else sp(s.tab("v", i, ids, F.linear(h, P["v"][i])))
            if a.ptr == 4 and a.eqh and not s.off: vv = torch.cat([sp(F.linear(F.rms_norm(s.ptr_old, (d,), s.n1[i]), P["v"][i]))[:, :1], vv[:, 1:]], 1)
            if a.eqb and not s.off: lo_ = a.eqh - a.eqb; vv = torch.cat([vv[:, :lo_], F.pad(vv[:, lo_:a.eqh], (0, 0, 1, 0))[:, :, :-1], vv[:, a.eqh:]], 1)
            if a.kvs and not s.off:
                ka = s.S["kva"][i][None, :, None, :]; sh = lambda z: F.pad(z, (0, 0, 1, 0))[:, :, :-1]
                k = ka[..., 0:1] * k + ka[..., 1:2] * sh(k); vv = ka[..., 2:3] * vv + ka[..., 3:4] * sh(vv)
            y = s.attn(i, q, k, vv, h)
            if a.ngh and not s.off:
                if i == 0 or not hasattr(s, "ngA") or s.ngA.shape[-1] != t:
                    pv = F.pad(ids, (1, 0), value=-1)[:, :-1]; M = ids[:, :, None] == pv[:, None, :]
                    if a.ngo >= 2: pv2 = F.pad(ids, (2, 0), value=-2)[:, :-2]; ip = F.pad(ids, (1, 0), value=-3)[:, :-1]; M = M & (ip[:, :, None] == pv2[:, None, :])
                    M = (M & torch.ones(t, t, dtype=torch.bool, device=ids.device).tril()).float(); s.ngA = M / M.sum(-1, keepdim=True).clamp_min(1)
                y = torch.cat([(s.ngA[:, None] @ vv[:, :a.ngh]), y[:, a.ngh:]], 1)
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


def evaluate(m, E=None):
    E = E or EV; m.eval(); X = torch.tensor(E[0], device=DEV); MK = torch.tensor(E[1], device=DEV); KM = getattr(m, "km", None); global POS; P0 = POS
    ls, cs, n = 0.0, 0.0, 0
    for c in range(0, X.shape[0], 64):   # 64-row chunks keep eval memory small (long gap evals hit 9 GB in one batch)
        x, mk = X[c:c + 64], MK[c:c + 64]
        if KM is not None: m.km = KM[c:c + 64]
        if P0 is not None: POS = P0[c:c + 64]
        with torch.no_grad(): lg = m(x[:, :-1])
        y = x[:, 1:]; k_ = int(mk.sum()); ls += F.cross_entropy(lg[mk].float(), y[mk], reduction="sum").item(); cs += (lg[mk].argmax(-1) == y[mk]).float().sum().item(); n += k_
    m.km = KM; POS = P0; m.train(); return ls / max(n, 1), cs / max(n, 1)


def main():
    m = Model(V, a.width, a.layers, a.heads).to(DEV)
    sc_ = [w for n, w in m.named_parameters() if n.startswith("S.")]
    tb = [w for n, w in m.named_parameters() if n.startswith("T.")]
    other = [w for n, w in m.named_parameters() if not n.startswith("P.") and not n.startswith("S.") and not n.startswith("T.")]
    groups = [dict(params=other, lr=a.elr)] + ([dict(params=sc_, lr=a.slr or a.elr)] if sc_ else []) + ([dict(params=tb, lr=a.tlr or a.elr)] if tb else [])
    muon = Muon(m, a.lr); adam = torch.optim.Adam(groups, betas=(a.ab1, a.ab2), eps=a.eps)
    f = torch.compile(m) if a.compile else m; t0 = time.time()
    rng = np.random.default_rng(0); curve = []; s90 = None
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
    with open(os.path.join(HERE, "results.jsonl"), "a") as fo: fo.write(json.dumps(res) + "\n")
    if a.gap: gl, ga = evaluate(m, EVG); print(f"  gap {a.gap}: acc {ga:.3f}, loss {gl:.3f}", flush=True)
    if a.gap:   # oracle: same gap, filler hidden from attention (positions still shifted)
        m.km = torch.tensor(EVG[0][:, :-1] != 0, device=DEV); gl2, ga2 = evaluate(m, EVG); m.km = None
        print(f"  gap {a.gap} filler hidden: acc {ga2:.3f}, loss {gl2:.3f}", flush=True)
        global POS
        km = torch.tensor(EVG[0][:, :-1] != 0, device=DEV); POS = (km.float().cumsum(-1) - 1).clamp_min(0)
        gl3, ga3 = evaluate(m, EVG); m.km = km; gl4, ga4 = evaluate(m, EVG); m.km = None; POS = None
        print(f"  gap {a.gap} positions frozen over filler: acc {ga3:.3f}; frozen + hidden: acc {ga4:.3f}", flush=True)
    print(f"{a.tag or 'run'}: acc {acc:.3f}, steps90 {s90}, loss {v:.3f}, body {m.body_params() / 1e3:.0f}K, train {res['train_s']}s, curve {curve[::max(1, len(curve) // 8)]}", flush=True)


if __name__ == "__main__":
    main()

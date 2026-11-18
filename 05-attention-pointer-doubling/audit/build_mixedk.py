"""Generate a controlled mixed-k benchmark from the original pdg.py.

No hidden intermediate-answer supervision. Every row contains a full reverse-ordered
7-link chain and *multiple queries* with random target hop counts 1..7.
Query format: [COUNT_k, original_key, answer_after_k_links].
Mixed-k gates:
  0 existing learned global layer gates,
  1 learn per-k gate lookup table (task count available in input),
  2 oracle-provided hard binary bits (positive control, not learned),
  3 learn count-token-conditioned linear gate (model input, not oracle).
All use original pointer-doubling P/Q attention and are trained end-to-end.
"""
from pathlib import Path
import sys

src = Path(__file__).resolve().parents[1].joinpath("pdg.py").read_text()


def replace_one(old, new):
    global src
    count = src.count(old)
    if count != 1:
        raise RuntimeError(f"expected 1 exact patch site, found {count}: {old[:110]!r}")
    src = src.replace(old, new, 1)


replace_one(
    'p.add_argument("--nev", type=int, default=512, help="eval rows")',
    'p.add_argument("--mixed", type=int, default=0)\n'
    'p.add_argument("--ctrl", type=int, default=0, help="mixed-k control: 0 global,1 table,2 oracle,3 embedding-conditioned")\n'
    'p.add_argument("--single", type=int, default=0, help="one query per row to exclude earlier answer tokens")\n'
    'p.add_argument("--nev", type=int, default=512, help="eval rows")'
)
replace_one(
    'a.L = (a.hops * a.pairs * (2 + a.fill) + 2 * a.pairs) if a.task == "hopx" else (6 * a.pairs if a.task == "mq2" else 64)',
    'a.L = (a.hops * a.pairs * (2 + a.fill) + (3 if a.mixed else 2) * (1 if a.single else a.pairs)) if a.task == "hopx" else (6 * a.pairs if a.task == "mq2" else 64)'
)
replace_one(
    '        dat = np.concatenate(blocks, 1); o = dat.shape[1]; Q = min(N, (L - o) // 2); assert Q >= 1, "L too short"\n'
    '        idx = np.argsort(rng.random((B, N)), 1)[:, :Q]; qk = ch[0][r, idx].copy()\n'
    '        if a.syn > 0: sw = rng.random((B, Q)) < a.syn; qk[sw] += V0\n'
    '        seq[:, :o] = dat; seq[:, o:o + 2 * Q:2], seq[:, o + 1:o + 2 * Q:2] = qk, ch[Hn][r, idx]\n'
    '        if L > o + 2 * Q: seq[:, o + 2 * Q:] = rng.integers(V0 - max(Vf, 1), V0, (B, L - o - 2 * Q))\n'
    '        mask[:, o:o + 2 * Q:2] = True\n',
    '        dat = np.concatenate(blocks, 1); o = dat.shape[1]\n'
    '        Q = min(N, (L - o) // (3 if a.mixed else 2)); assert Q >= 1, "L too short"\n'
    '        if a.mixed and a.single: Q = 1\n'
    '        idx = np.argsort(rng.random((B, N)), 1)[:, :Q]; qk = ch[0][r, idx].copy()\n'
    '        if a.syn > 0: sw = rng.random((B, Q)) < a.syn; qk[sw] += V0\n'
    '        seq[:, :o] = dat\n'
    '        if a.mixed:\n'
    '            assert not a.syn and not a.fill and not a.dup and V0 - (Hn + 1) * Tr >= Hn, "reserve Hn count token IDs"\n'
    '            counts = rng.integers(1, Hn + 1, (B, Q))\n'
    '            ends = np.stack(ch, axis=0)[counts, r, idx]\n'
    '            seq[:, o:o + 3 * Q:3] = V - Hn - 1 + counts\n'
    '            seq[:, o + 1:o + 3 * Q:3], seq[:, o + 2:o + 3 * Q:3] = qk, ends\n'
    '            mask[:, o + 1:o + 3 * Q:3] = True\n'
    '        else:\n'
    '            seq[:, o:o + 2 * Q:2], seq[:, o + 1:o + 2 * Q:2] = qk, ch[Hn][r, idx]\n'
    '            if L > o + 2 * Q: seq[:, o + 2 * Q:] = rng.integers(V0 - max(Vf, 1), V0, (B, L - o - 2 * Q))\n'
    '            mask[:, o:o + 2 * Q:2] = True\n'
)
replace_one(
    '        if a.ptr == 11: S["pg"] = nn.Parameter(torch.full((nl,), a.pgi))',
    '        if a.ptr == 11: S["pg"] = nn.Parameter(torch.full((nl,), a.pgi))\n'
    '        if a.mixed and a.ctrl == 1: S["gt"] = nn.Parameter(torch.zeros(nl, a.hops + 1))\n'
    '        if a.mixed and a.ctrl == 3: S["gw_mixed"] = nn.Parameter(torch.zeros(nl, d))'
)
replace_one(
    '                        A1_ = torch.softmax(lg1, -1); g_ = torch.sigmoid(s.S["pg"][i])\n'
    '                        rq = g_ * ((A1_ @ rr) if rr is not None else qq) + (1 - g_) * rq; qq = g_ * (A1_ @ ptr) + (1 - g_) * qq',
    '                        A1_ = torch.softmax(lg1, -1)\n'
    '                        if a.mixed and a.ctrl in (1, 2):\n'
    '                            cnt = (F.pad(ids, (1, 0))[:, :-1] - (V - a.hops - 1)).clamp(0, a.hops)\n'
    '                            if a.ctrl == 1: glogit = s.S["pg"][i] + s.S["gt"][i][cnt]\n'
    '                            else: glogit = torch.where((cnt & (1 << i)) != 0, torch.ones_like(cnt, dtype=e0.dtype) * 8, torch.ones_like(cnt, dtype=e0.dtype) * -8)\n'
    '                            g_ = torch.sigmoid(glogit)[..., None]\n'
    '                        elif a.mixed and a.ctrl == 3:\n'
    '                            prev_emb = F.pad(e0, (0, 0, 1, 0))[:, :-1]\n'
    '                            g_ = torch.sigmoid(prev_emb @ s.S["gw_mixed"][i] + s.S["pg"][i])[..., None]\n'
    '                        else: g_ = torch.sigmoid(s.S["pg"][i])\n'
    '                        rq = g_ * ((A1_ @ rr) if rr is not None else qq) + (1 - g_) * rq; qq = g_ * (A1_ @ ptr) + (1 - g_) * qq'
)
replace_one(
    '    res["steps_run"] = step + 1',
    '''    res["steps_run"] = step + 1
    if a.mixed:
        n_true = {k: 0 for k in range(1, a.hops + 1)}
        n_total = {k: 0 for k in range(1, a.hops + 1)}
        o = a.hops * a.pairs * (2 + a.fill)
        Q = 1 if a.single else min(a.pairs, (a.L + 1 - o) // 3)
        pidx = np.arange(o + 1, o + 3 * Q, 3)
        m.eval()
        with torch.no_grad():
            for batch0 in range(0, a.nev, 16):
                sample = torch.tensor(EV[0][batch0:batch0 + 16], device=DEV)
                pred = m(sample[:, :-1]).argmax(-1)
                for ki in range(1, a.hops + 1):
                    cnt = sample[:, pidx - 1] - (V - a.hops - 1)
                    active = (cnt == ki)
                    n_true[ki] += int(((pred[:, pidx] == sample[:, pidx + 1]) & active).sum().item())
                    n_total[ki] += int(active.sum().item())
        m.train()
        res["acc_by_hop"] = {str(k): round(n_true[k] / max(1, n_total[k]), 4) for k in n_true}
        res["query_counts"] = n_total
        print("MIXED_K_RESULTS", json.dumps(res["acc_by_hop"]), flush=True)'''
)
Path(__file__).resolve().parents[1].joinpath("pdg_mixed_generated.py").write_text(src)
print("Generated pdg_mixed_generated.py; run with --mixed 1 --ctrl 0|1|2|3")

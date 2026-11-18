"""Exact-token, training-free pointer-doubling mechanism test; NOT model validation.
Causal links are stored as pairs (key, value). P_l at a value token can read
only earlier pairs. Query Q is after all pairs. Hard binary-lifting gates
are optimized exhaustively per query. Outputs are reachability, not accuracy.
"""
import itertools
import numpy as np


def oracle(depth, hops, block_order):
    pos = {stage: index for index, stage in enumerate(block_order)}
    P = [np.arange(hops + 1, dtype=int)]
    for layer in range(depth):
        prev, nxt = P[-1], P[-1].copy()
        for stage in range(1, hops + 1):
            j = prev[stage]
            if j < hops and pos[j] < pos[stage - 1]:
                nxt[stage] = prev[j + 1]
        j = prev[0]
        if j < hops:
            nxt[0] = prev[j + 1]
        P.append(nxt)
    reachable = {0}
    for layer in range(depth):
        nxt = set(reachable)
        for q in reachable:
            if q < hops:
                nxt.add(int(P[layer][q + 1]))
        reachable = nxt
    return hops in reachable, max(reachable), len(reachable)


if __name__ == "__main__":
    for L, k in [(3, 7), (4, 15), (5, 31)]:
        reverse = oracle(L, k, list(reversed(range(k))))
        forward = oracle(L, k, list(range(k)))
        if k == 7:
            orders = itertools.permutations(range(k))
            data = [oracle(L, k, order) for order in orders]
            print(f"L={L} k={k}: reverse={reverse}, forward={forward}, "
                  f"exact random {sum(x[0] for x in data)}/{len(data)} = "
                  f"{sum(x[0] for x in data)/len(data):.6f}")
        else:
            rng = np.random.default_rng(20261116)
            data = [oracle(L, k, rng.permutation(k)) for _ in range(3000)]
            print(f"L={L} k={k}: reverse={reverse}, forward={forward}, "
                  f"random successes={sum(x[0] for x in data)}/3000, "
                  f"mean maximum reach={np.mean([x[1] for x in data]):.3f}")

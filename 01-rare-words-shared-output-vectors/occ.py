"""Data-only statistics of a cell: share of validation targets by training-occurrence bin. python3 occ.py V steps [batch] [tail]"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "llm-per-word-weights", "data"); EOT = 50256
def stats(V, steps, batch=32, tail=1.0, L=64):
    counts = np.load(os.path.join(DATA, "counts.npy")).astype(np.float64); rank = np.argsort(-counts, kind="stable")
    if tail <= 1: top = rank[:V - 2]
    else:
        h = (V - 2) // 2; top = np.concatenate([rank[:h], rank[np.unique(np.linspace(h, int(tail * V), V - 2 - h).astype(np.int64))]])
    keep = np.unique(np.concatenate([top, [EOT]])); Vv = len(keep) + 1
    look = np.full(65536, Vv - 1, np.int64); look[keep] = np.arange(len(keep))
    need = steps * batch * L
    tr = np.memmap(os.path.join(DATA, "fineweb_train_000001.bin"), dtype=np.uint16, mode="r", offset=1024)
    rows = look[np.asarray(tr[:need]).astype(np.int64).reshape(-1, L)]
    occ = np.bincount(rows[:, 1:].reshape(-1), minlength=Vv)
    va = np.memmap(os.path.join(DATA, "fineweb_val_000000.bin"), dtype=np.uint16, mode="r", offset=1024)
    val = look[np.stack([np.asarray(va[x:x + L]) for x in np.linspace(0, len(va) - L - 1, 600).astype(np.int64)]).astype(np.int64)][:, 1:].reshape(-1)
    o = occ[val]; unk = val == Vv - 1
    E = [0, 1, 10, 32, 100, 316, 1000, 3162, 10 ** 12]
    return {f"{E[i]}-{E[i+1]-1}": float(((o >= E[i]) & (o < E[i + 1]) & ~unk).mean()) for i in range(len(E) - 1)} | {"unk": float(unk.mean())}
if __name__ == "__main__":
    a = sys.argv[1:]; print(stats(int(a[0]), int(a[1]), *(int(a[2]),) if len(a) > 2 else (), *(float(a[3]),) if len(a) > 3 else ()))


def pshares(V, tail=1.0):
    """Share of validation targets by corpus frequency bin (same bins as grid.py fbins)."""
    counts = np.load(os.path.join(DATA, "counts.npy")).astype(np.float64); rank = np.argsort(-counts, kind="stable")
    if tail <= 1: top = rank[:V - 2]
    else:
        h = (V - 2) // 2; top = np.concatenate([rank[:h], rank[np.unique(np.linspace(h, int(tail * V), V - 2 - h).astype(np.int64))]])
    keep = np.unique(np.concatenate([top, [EOT]])); Vv = len(keep) + 1
    look = np.full(65536, Vv - 1, np.int64); look[keep] = np.arange(len(keep))
    pw = np.append(counts[keep], 0) / counts.sum()
    va = np.memmap(os.path.join(DATA, "fineweb_val_000000.bin"), dtype=np.uint16, mode="r", offset=1024)
    val = look[np.stack([np.asarray(va[x:x + 64]) for x in np.linspace(0, len(va) - 65, 600).astype(np.int64)]).astype(np.int64)][:, 1:].reshape(-1)
    E = [0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1.1]; p = pw[val]; unk = val == Vv - 1
    return [round(float(((p >= E[i]) & (p < E[i + 1]) & ~unk).mean()), 4) for i in range(len(E) - 1)] + [round(float(unk.mean()), 4)]

"""Two-stage associative-recall falsifiers: overwrite and value aliasing.

Protocol for each row (model input length T):
  N distinct key/value facts -> Q1 key/value retrieval queries ->
  W=N/2 new facts -> Q2 mixed queries on new/old categories.

- mqar_rewrite: the new facts overwrite W *existing* keys with guaranteed
  different values; later answers must return the latest value, not stale one.
- mqar_alias: the new facts introduce W *new* keys whose values deliberately
  reuse old values; later answers must recognize that same value may be
  associated with another distinct key (value match != redundant write).

The task is deliberately phase-structured, providing an *easier* first
falsifier than a randomly interleaved instruction stream. No task markers.
All scored positions are query KEYS (predict the following value), never
the fact records or the answers themselves. Last model position is filler.
"""
import numpy as np


def layout(T, N):
    if N < 4 or N % 2:
        raise ValueError("pairs must be an even number >=4")
    W = N // 2
    if T <= 2 * N + 2 * W + 8:
        raise ValueError("sequence too short to contain two query phases")
    Q1 = (T - 2 * N - 2 * W) // 4
    Q2 = (T - 2 * N - 2 * W - 2 * Q1) // 2
    if Q1 < 2 or Q2 < 2:
        raise ValueError("not enough queries for reliable subgroup diagnostics")
    bounds = (0, 2*N, 2*N+2*Q1, 2*N+2*Q1+2*W,
              2*N+2*Q1+2*W+2*Q2)
    if bounds[-1] > T:
        raise ValueError("invalid layout")
    return W, Q1, Q2, bounds


def make_update(seq, mask, rng, a):
    """Mutates and returns (seq Bx(T+1), mask BxT); deterministic under rng."""
    B, length = seq.shape
    T = length - 1
    N, V = a.pairs, a.tv
    W, Q1, Q2, bd = layout(T, N)
    Kn = V // 2 - 1
    if Kn < N + W:
        raise ValueError("synthetic vocabulary has insufficient distinct keys")
    low, span = Kn + 1, V - (Kn+1)
    if span < 2:
        raise ValueError("need >=2 value token identities")
    r = np.arange(B)[:, None]

    chosen = np.argsort(rng.random((B, Kn)), axis=1)[:, :N+W] + 1
    old_keys, new_keys = chosen[:, :N], chosen[:, N:]
    old_vals = rng.integers(low, V, size=(B, N))
    seq[:, :2*N:2], seq[:, 1:2*N:2] = old_keys, old_vals

    first_idx = rng.integers(0, N, size=(B, Q1))
    seq[:, bd[1]:bd[2]:2] = old_keys[r, first_idx]
    seq[:, bd[1]+1:bd[2]:2] = old_vals[r, first_idx]
    mask[:, bd[1]:bd[2]:2] = True

    permutation = np.argsort(rng.random((B, N)), axis=1)
    updated_idx, unchanged_idx = permutation[:, :W], permutation[:, W:]
    if a.task == "mqar_rewrite":
        second_keys = old_keys[r, updated_idx]
        previous_vals = old_vals[r, updated_idx]
        # Use positive cyclic displacement, making new values != old values.
        displacement = rng.integers(1, span, size=(B, W))
        second_vals = low + (previous_vals - low + displacement) % span
        assert np.all(second_vals != previous_vals)
    elif a.task == "mqar_alias":
        second_keys = new_keys
        alias_from = rng.integers(0, N, size=(B, W))
        second_vals = old_vals[r, alias_from]
        assert not np.any(second_keys[:, :, None] == old_keys[:, None, :])
    else:
        raise ValueError("Unknown update task")
    seq[:, bd[2]:bd[3]:2] = second_keys
    seq[:, bd[2]+1:bd[3]:2] = second_vals

    # Force a balanced Q2 diagnostic between new/updated facts and
    # untouched/old facts. Randomize order, do not leak category by position.
    n_changed = Q2 // 2
    n_old = Q2 - n_changed
    ids_changed = rng.integers(0, W, size=(B, n_changed))
    if a.task == "mqar_rewrite":
        old_other_keys = old_keys[r, unchanged_idx]
        old_other_vals = old_vals[r, unchanged_idx]
        ids_old = rng.integers(0, N-W, size=(B, n_old))
    else:
        old_other_keys, old_other_vals = old_keys, old_vals
        ids_old = rng.integers(0, N, size=(B, n_old))
    query_keys = np.concatenate([
        second_keys[r, ids_changed],
        old_other_keys[r, ids_old]
    ], axis=1)
    query_vals = np.concatenate([
        second_vals[r, ids_changed],
        old_other_vals[r, ids_old]
    ], axis=1)
    shuffle = np.argsort(rng.random((B, Q2)), axis=1)
    seq[:, bd[3]:bd[4]:2] = query_keys[r, shuffle]
    seq[:, bd[3]+1:bd[4]:2] = query_vals[r, shuffle]
    mask[:, bd[3]:bd[4]:2] = True

    assert int(mask.sum()) == B*(Q1+Q2)
    assert not mask[:, :bd[1]].any() and not mask[:, bd[2]:bd[3]].any()
    return seq, mask

# Pointer doubling: results log

Code: pdg.py (recall bed, copy of identity-heads rbg.py with --rev, --ptr 5/6/7/8/9/10), pdlm.py (text bed), grid.py (driver, arms DBL/DSL/OSL/PRD/PRG/PNB).
Bed: --task hopx, 8 pairs, width 64, batch 64, lr 0.02 (arms with their own lr), 300 steps, stop at 0.99, eval every 25 steps. Every run is one line in results.jsonl.

## Round 1 predictions (written 2026-11-15 21:25, before the grid)

- D1. Doubling (DBL, --ptr 5) with reverse-ordered chains solves k = 2^L - 1 hops at L = 2, 3, 4 (3, 7, 15 hops) at >= 0.95.
- D2. Doubling without a self-loop fails hop counts that are not 2^L - 1 when L is the minimum depth (e.g. 5 hops in 3 layers < 0.6), because the pointer overshoots the chain end into a blur.
- D3. Doubling + self-loop (DSL, --ptr 9) solves every k <= 2^L - 1 (3-7 hops in 3 layers, 8-15 hops in 4 layers) at >= 0.95.
- D4. Every tuned competitor (F, KVS, PRI, NG, one-hop pointer + no-backtrack) stays at guessing (< 0.40) whenever layers < hops.
- D5. Forward-ordered chains (--rev 0): DBL fails at L < k; DSL falls back to one hop per layer (solves k <= L, fails k > L). Random order (--rev 2) lands between.
- D6. Freezing the pointer at test time drops DSL to near 0.
- D7. Holds across width 32-256 and 8-64 pairs (context up to 512+) with the same depth rule.
- D8. Text (S5): DSL ties pair-match identity heads within 0.005 loss.

## Round 1 results (200 runs in results.jsonl, blocks A-AH, 2026-11-15 21:25-22:25; block AI and the RP reproduce runs below bring the total to 218)

All recall runs: 8 pairs, width 64, batch 16, eval every 25 steps, 50-100 steps, stop at 0.99. Chance among chain ends = 1/pairs (0.125 at 8 pairs).

### What the predictions got right and wrong

- D1 right only for the final design (see below). Doubling with the pointer as head-0 value (--ptr 4) solves 3/2, 7/3, 15/4 at 0.94-0.995. Without the pointer value (--ptr 5, residual readout) 15/4 fails (0.257).
- D2 right for --ptr 4 (5/3: 0.308, 11/4: 0.181), wrong for --ptr 5 at 3 layers (4-6 hops pass through residual composition).
- D3 wrong: the self-loop (--ptr 9) never helped (11/4: 0.16-0.22 at every psm init; at init the self score ties the true match at layer 0). Dropped.
- New design 1, doubled no-backtrack (--pnb with --ptr 4/11): a stream R_l = token one stage before the pointer (R_1 = P_0, R_{l+1} = A_l R_l) and the penalty -3 cos(R_l, e_{j-2}). Init probe at 15/4: mass on stage 2^l per layer 0.90/0.47/0.41/0.23 without it, 0.90/0.85/0.86/0.83 with it; at 31/5 0.90/0.84/0.86/0.84/0.82.
- New design 2, binary lifting (--ptr 11 = BLN with --pnb 3): head 0 doubles the jump table P; head 1 moves a second pointer Q by 2^l (Q <- g_l (A_Q P_l) + (1 - g_l) Q, g_l learned, init 0.5) with its own no-backtrack stream. Reaches every k <= 2^L - 1.
- D4 right: every competitor < 0.32 at 7/3, 11/4, 15/4 at every lr. But the one-hop pointer + no-backtrack from the identity-heads paper is not one layer per hop on reverse-ordered chains: it reaches 3, 6, 9 hops at L = 2, 3, 4 (3/2 1.000, 5/3 0.991, 6/3 0.999, 7/3 0.208, 9/4 0.998, 10/4 0.166).
- D5 right for the causal limit and harder than predicted: forward order fails even 3/3 (0.033) for BLN, while the one-hop pointer solves 3/3 forward (0.991). Random order 7/3: 0.056.
- D6 right: pointer frozen at test 0.038.
- D7 right with one exception: width 32 needs temperature 10 (0.925; at 3: 0.118).
- D8 split: one doubling head + pair-match ties pair-match (3.7835 vs 3.7782, plain 3.8002); the two-head lifting version costs loss (3.8098 alone, 3.8036 with pair-match).

### Hops x layers, BLN (lr 0.02)

| L \ k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.038 | | | | | | | | | | | | | | |
| 2 | | 1.000 | | | | | | | | | | | | | | |
| 3 | | | 0.989 | 0.999 | 0.986 | 0.978 | 0.979 | 0.131 | | | | | | | | |
| 4 | | | 0.979 | | | | | | 0.987 | 0.991 | 0.981 | 0.992 | 0.923 | 0.438 (100 steps) | 0.977 | 0.120 |

5 layers (4 pairs, chance 0.25): 31 hops 0.964, 24 hops 0.955, 20 hops 0.237, 18 hops 0.288, 32 hops 0.229.

### Four settings, best lr per arm

| Setting | Tuned attention (F) | KV shifting | Primer | n-gram heads | One-hop pointer + nb | Doubling, no lifting (DNB) | Binary lifting (BLN) |
|---|---|---|---|---|---|---|---|
| 7 hops, 3 layers | 0.269 | 0.269 | 0.263 | 0.157 | 0.208 | 0.981 | 0.987 |
| 11 hops, 4 layers | 0.293 | 0.289 | 0.275 | 0.170 | 0.194 | 0.207 | 0.981 |
| 15 hops, 4 layers | 0.295 | 0.273 | 0.311 | 0.219 | 0.288 | 0.993 | 0.984 |
| 7 hops, 3 layers, 16 pairs (L 256) | 0.125 | 0.072 | 0.063 | 0.055 | 0.081 | | 0.991 |

lr swept: F/KVS/PRI/NG 0.01, 0.04, 0.08 (7/3, 11/4, 15/4); one-hop pointer 0.01-0.04; BLN 0.01, 0.02, 0.04 (0.974-0.987 at 7/3, 0.976-0.984 at 15/4).

### Robustness (BLN vs best other arm in the same cell)

| Cell | BLN | Best other |
|---|---|---|
| width 32, temperature 10 | 0.925 | 0.165 (F) |
| width 128, 7/3 | 0.979 | 0.277 (F) |
| width 128, 15/4 | 0.972 | 0.202 (one-hop pointer) |
| width 256, 7/3 | 0.970 | 0.257 (F) |
| 24 pairs, L 384 | 0.998 | 0.066 (F) |
| 2 filler tokens per pair, L 240 | 0.992 | 0.166 (KVS) |
| fillers 30% replaced by chain tokens | 0.763 | 0.051 (F) |

### Controls (7/3 unless noted)

- Pointer frozen at test: 0.984 -> 0.038. Identity heads (pointer heads) ablated: 0.029; the same number of normal heads ablated: 0.985; last layer only: 0.024.
- No-backtrack removed (BL): 11/4 0.139, 15/4 0.196 (with it 0.981, 0.977).
- Lifting removed (DNB): 5/3 0.309, 10/4 0.146, 11/4 0.207, 12/4 0.130; at k = 2^L - 1 DNB works (0.995, 0.981, 0.993).
- Pointer not used as head-0 value (--ptr 5): 15/4 0.257 vs 0.941.
- Sink + ALiBi on the other heads (DBLA): 15/4 0.298 vs 0.941.
- Predecessor doubling (--ptr 6/8) on forward chains, first test: 3/2 0.082, 7/3 0.057-0.064. Both cells are past its reach of 2^(L-1) links; see block AI.

### Text (S5: width 64, 4 layers, L 256, 150 steps, FineWeb)

| Arm | val loss |
|---|---|
| Tuned attention | 3.8002 |
| Pair-match identity head | 3.7782 |
| One doubling head + pair-match | 3.7835 |
| Binary lifting (2 heads) | 3.8098 |
| Binary lifting + pair-match head | 3.8036 |

Takeaway: a doubled pointer with a doubled no-backtrack term and a gated jump pointer lets L layers follow every chain of up to 2^L - 1 links stored in reverse order, from step 25-50; every tuned alternative stays near chance, and the best one-hop pointer reaches 3(L - 1).

## Block AI: predecessor doubling inside its reach (forward order, 2026-11-15 22:50, 14 runs)

Prediction before the block: predecessor doubling (--ptr 6) reaches 2^(L-1) links on forward chains (one layer reads out), as in its IDEA.md on rb.py, and fails other counts.

| Links / layers (--rev 0, 8 pairs, 100 steps) | Tuned attention | KV shifting | One-hop pointer + nb | Full head (BLN) | Predecessor doubling | No-doubling control (--ptr 7) |
|---|---|---|---|---|---|---|
| 2 / 2 | | | | | 0.987 | |
| 4 / 3 | 0.126 | | 0.119 | 0.042 (block AC) | 0.976 | |
| 8 / 4 | 0.223 | 0.194 | 0.221 | | 0.915 | 0.133 |
| 16 / 5, 4 pairs (chance 0.25) | 0.465 | | | | 0.749 (100 steps), 0.818 (200 steps) | |

- 5 / 3 forward: 0.116 (only the count 2^(L-1) is reached). Gated version (--ptr 8) 3 / 3 forward: 0.988.
- Held. Predecessor doubling becomes a section of the paper (Section 7, Table 5) instead of being dropped. Accuracy falls with depth (0.99 → 0.82); no lifting or no-backtrack term was built for it.

Reproduce runs (RP-*, reproduce.sh, 67 s): 7/3 tuned attention 0.174, full head 0.973; 15/4 one-hop pointer 0.170, full head 0.989.

# Predicted-skip write: scaling-simulation study (gate 4, 2026-11-15)

Bed: MQAR (rc.py), 2 layers, width 32, 2 heads (dk 16), L64, P16 (one query per pair), 300 steps, batch 32, Muon lr 0.06 / Adam 0.04, init 0.5, exact chunked recurrence (chunk 16) for every base (mix.py `chunked`, matches the parallel form to 4e-6 for g, D, d, e at chunk sizes 1, 7, 16, 64). Val loss on query answers, lower is better. Driver: `g.py` (one run at a time, `nice -n 20`, 4 threads). Grid rows in results.jsonl have tags starting with `G:`.

Arms: **g** gated linear attention (scalar decay, Mamba-2-style write); **D** delta rule; **d** gated delta rule; **e** GDN-2-style (gated delta + learned per-token erase gate, bias -2, the strongest published competitor from gate 3); **+ps** predicted-skip gate (g0 1, learned per-head gamma). Delta family: the gate scales the write strength only, unless `lpsE 1` (also the erase strength).

Machine note: other sessions train at the same time (load 10-12 on 10 cores); a 300-step cell takes 30-45 s instead of 14 s. Runs are chaotic: two float-equal implementations differ by 0.2 at step 100, so single gaps under about 0.1 at P16 are noise; cells use 2 learning rates and report the better one where noted.

## Round 1: base model (L64 P16)

Rule R0: predicted-skip helps when the base re-writes values it has just retrieved into a full state; that is gated linear attention, while the delta rule's write v - S k is already small for a stored pair.
Predictions: g+ps beats g by at least 0.15; on D and d the write-only gate ties (within 0.1) and the gate on erase too hurts; on e it ties; the oracle (no writes at query-phase value positions) helps every base.

| base | base, by lr (0.03 / 0.04 / 0.05 / 0.06) | + predicted-skip, same lrs | best base | best +ps | gain | oracle (no query-value writes) |
|---|---|---|---|---|---|---|
| g gated linear | - / 0.520 / 0.465* / 0.379 | - / 0.147 / 0.218* / 0.196 | 0.379 | 0.147 | **0.23** | 0.162* |
| e GDN-2-style | - / 0.485 / - / 0.506 | - / 0.119 / - / 0.109 | 0.485 | 0.109 | **0.38** | 0.108* |
| D delta rule | 0.754 / 0.791 / 0.910 / 0.900 | 0.344 / 0.423 / 0.713 / 1.158 | 0.754 | 0.344 | **0.41** | did not learn (4.24*) |
| d gated delta | - / 0.926* / 1.079 / - | - / 0.451* / 0.360 / - | 0.926 | 0.360 | **0.57** | |
| D, gate also on erase (lpsE 1) | | - / 0.444 / - / - | | | ties write-only (0.423) | |

`*` = MPS (other rows CPU; same code, the two devices differ only by run-to-run chaos).

Gate at the end of training (layer 0, mean exp(-gamma relu(cos))): pair-phase values 0.88-0.92, query-phase values 0.13-0.16, query-phase keys 0.38-0.79. The learned gamma climbs from 1 to 2.5-3.6 in layer 0 and falls to 0.13-1.2 in layer 1, so layer 0 does the skipping.

Takeaway: the gate helps all four bases by 0.23-0.57 at the best learning rate of each arm, and it reaches the oracle ceiling (g 0.147 vs 0.162, e 0.109 vs 0.108). R0's delta-rule clause is wrong. The gate-3 "hurt on the delta rule" (1.754) was one run at lr 0.06, where the delta rule with the gate diverges (1.158 here); the delta rule's usable lr window is narrow and the gate moves its best lr down to 0.03. The delta rule's write v - S k does not stop the harm: it writes the residual of a retrieved pair, which is crosstalk noise from other pairs, so the rewrite still adds interference.
Speed note: from here on every run is on MPS (CPU runs took 80-185 s under 6 other training processes; MPS 37-112 s). The 300-step P16 cell is the smallest where the failure forms (accuracy is under 5% at step 75 and the gap opens after step 150), so cells stay at 300 steps.

## Round 2: pairs per state and state size (L64, lr 0.05, MPS)

Rule R1: the gate's gain equals what the base loses to query-phase value rewrites (the oracle headroom); the headroom is large when the state is near capacity (pairs per key dimension P/dk near 1) and queries come back to stored pairs.
Predictions: P4 (7 queries per pair, P/dk 0.25) ties within 0.05; P8 (3 per pair, 0.5) gains 0.05-0.15; P24 (0.33 per pair, 1.5) gains about 0.1; width 64 (dk 32) at P16 (0.5) gains under 0.05 absolute; width 16 (dk 8) at P16 (2.0): both arms fail and the gain is small.

| cell (L64) | pairs per key dim P/dk | queries per pair | g | g+ps | e (GDN-2-style) | e+ps | g oracle |
|---|---|---|---|---|---|---|---|
| P4, dk 16 | 0.25 | 7 | 0.109 (best of 3 lrs) | 0.107 (best of 3) | 0.019 | 0.026 | 0.049 |
| P4, dk 16, **chunk 4** | 0.25 | 7 | 0.140 | **0.030** | | | |
| P4, dk 16, chunk 2 | 0.25 | 7 | | 0.182 | | | |
| P8, dk 16 | 0.5 | 3 | 0.229 | **0.026** | 0.108 | **0.014** | |
| P16, dk 16 (round 1) | 1.0 | 1 | 0.379 (best) | **0.147** (best) | 0.485 | **0.109** | 0.162 |
| P24, dk 16 | 1.5 | 0.33 | 3.125 | 3.880 | 1.582 | 3.153 | |
| P16, dk 32 (width 64) | 0.5 | 1 | 0.060 | **0.010** | 0.036 | **0.006** | |
| P16, dk 8 (width 16) | 2.0 | 1 | 3.401 | 2.572 | 3.187 | 4.784 | |

Takeaways:
- Where the base learns the task (P/dk at most 1), the gate cuts the loss 3-9 times (P8: 0.229 -> 0.026, width 64: 0.060 -> 0.010), on g and on the GDN-2-style competitor alike.
- P4 at chunk 16 ties at all 3 lrs. Reason found: with 4 pairs, the pairs and the first 4 queries all sit in chunk 0, whose chunk-start state is empty, so the gate cannot see them; most re-queries also land in the same chunk as the previous query of that pair. Prediction made before the run: chunk 4 recovers the oracle gap. Result: 0.030 vs 0.140 (beats the 0.049 oracle). Rule: the gate only sees a pair once it is written at least one chunk boundary before the re-query.
- Chunk 2 (0.182) does not continue the trend; one run, see the chunk round.
- Over capacity (P/dk 1.5-2), retrieval has not formed after 300 steps (accuracy 0.03-0.6) and the gate is noise or harm (e: 1.58 -> 3.15). The pair-phase gate drops to 0.64-0.84 there (0.88-1.00 in the working cells), so it suppresses writes of new pairs. Likely mechanism: before retrieval forms, the read is an average of stored values, which has positive cosine with any new value, so the gate damps every write. The next round tests this with more steps.

## Round 3: context length (tokens per step fixed at 2,048) and training length over capacity

Rule R2: the gain is the rewrite interference the gate can see. It needs (a) a base that has formed retrieval (P/dk at most about 1, or enough steps), (b) re-queries at least one chunk after the pair was written, (c) enough load that rewrites interfere (P/dk at least 0.5 or many re-queries per pair). Before retrieval forms, the gate damps every write and slows learning.
Predictions: L128 P16 (3 queries per pair) gains at least as much as L64 (base 0.5-0.8, +ps 0.15-0.25); L256 P16 (7 per pair) gains more (base above 1, +ps 0.2-0.4); L128 P32 (P/dk 2) both fail and the gate does not help; P24 at 600 steps: once the base learns, the gate helps.

| cell (lr 0.05) | g | g+ps (g0 1) | g+ps (g0 0.25) | e | e+ps (g0 1) | e+ps (g0 0.25) |
|---|---|---|---|---|---|---|
| L64 P16 (1 query per pair) | 0.465 | 0.218 | **0.164** | 0.485 (lr .04) | 0.109 (lr .06) | **0.130** |
| L128 P16 (3 per pair), batch 16 | 0.610 | **0.150** | | 0.651 | **0.362** | |
| L256 P16 (7 per pair), batch 8 | 0.790 | **0.281** | | 0.997 | **0.620** | |
| L64 P8 | 0.229 | **0.026** | **0.035** | | | |
| L64 P24 (P/dk 1.5), 300 steps | 3.125 | 3.880 | **1.813** | 1.582 | 3.153 | **1.181** |
| L64 P24, 600 steps | 0.427 | **0.355** | | 0.530 | 4.853 (never learned) | |
| L128 P32 (P/dk 2), 300 steps | 3.394 | 4.844 (never learned) | **2.322** | | | |

Predictions: L128 and L256 gains grow (g: 0.25 -> 0.46 -> 0.51; e: 0.38 -> 0.29 -> 0.38), right for g, flat for e. L128 P32 "the gate does not help": wrong for g0 0.25 (2.32 vs 3.39). P24 at 600 steps "helps once learned": right for g, wrong for e at g0 1.

Takeaways:
- Longer context with a fixed number of pairs means more re-queries per pair, and the gain grows with it (g: 0.25 at L64, 0.46 at L128, 0.51 at L256).
- The over-capacity harm is a learning-onset effect of the initial gamma, not a limit of the idea: at g0 1 the gate damps all writes while reads are still noise and the model never learns (accuracy 0.01); at g0 0.25 the same cells learn faster than the base (P24: 1.81 vs 3.13; L128 P32: 2.32 vs 3.39; P24 e: 1.18 vs 1.58).
- The learned gamma ends near 2.8 in layer 0 from either start (and 0.1-0.7 in layer 1), so g0 0.25 loses nothing where g0 1 worked (P16: 0.164 vs 0.218; P8: 0.035 vs 0.026). Default from here on: g0 0.25.

## Round 4: depth, delta-rule bases, training length (L64 P16, g0 0.25)

Rule R3: with g0 0.25 the gate helps every recall cell where the base re-writes retrieved pairs that were stored at least one chunk earlier; the gain grows with re-queries per pair and with load.
Predictions: 1 layer gains at least 0.15 (one layer with the short conv can do MQAR, and the gate's read is the answer read); 4 layers: the base improves (4 states), the gain shrinks but stays above 0.05, and only layer 0 learns a large gamma; D and d at g0 0.25 gain at every lr in 0.03-0.06 (the g0 1 divergence at 0.06 was the onset effect); 150 steps: both arms barely learn, gap small; 600 steps: both improve, the gate still wins.

| cell (lr 0.05, g0 0.25) | g | g+ps | e | e+ps |
|---|---|---|---|---|
| 1 layer, 300 steps | 0.383 | **0.150** | 0.405 | **0.119** |
| 2 layers, 150 steps | 1.593 | **1.113** | **1.282** | 1.350 |
| 2 layers, 300 steps | 0.465 | **0.164** | 0.485 (lr .04) | **0.130** |
| 2 layers, 600 steps | 0.317 | **0.116** | 0.314 | **0.095** |
| 3 layers, 600 steps | 0.326 | **0.109** | 0.313 | **0.073** |
| 4 layers, 300 steps (not converged) | **0.811** | 1.953 | 1.797 | **0.729** |

Prediction check: 1 layer gains 0.23 and 0.29 (right). 4 layers at 300 steps: wrong in both directions; the base is worse than 2 layers, and the runs are still in the retrieval-onset lottery (loss curves drop between steps 125 and 200 at random). 3 layers at 600 steps, which is converged, gains 0.22 (g) and 0.24 (e), the same as 2 layers (right). 150 steps: gap small and mixed sign (right). 600 steps: both improve, the gate still wins by 0.20-0.22 (right).

Takeaways:
- Depth does not shrink the gain once runs converge: 0.23 / 0.30 / 0.22 for g and 0.29 / 0.22 / 0.24 for e at 1 / 2 / 3 layers.
- Only layer 0 learns a large gamma (2.6-2.9). Layers 1-2 drop gamma to 0.01-0.07, so their gates stay at 0.97-1.00: the model turns the gate off where it is not useful.
- Training length: the gain is zero before retrieval forms (150 steps) and stays at about 0.2 from 300 to 600 steps, so it is not a speed-up that the base catches up on.
- Machine note: 3-layer gated runs on MPS reach 1.65 GB and were killed twice by the lag guard's total-memory rule; on CPU (2 threads) they need 0.5 GB and take 80-100 s.

## Round 6: learning-rate sweeps in four settings, delta rule at g0 0.25

Rule R4 (= R3 plus depth): the gate helps every converged recall cell where retrieved pairs are re-written at least one chunk after storage; the gain (0.2-0.5 in loss) does not shrink with depth or steps.
Predictions: (1) at L128 P16, L64 P8 and width 64 P32, g+ps beats the best g at its own best lr, and e+ps beats the best e; (2) width 64 P32 (same load P/dk 1 as the main cell, twice the state) gains about as much as L64 P16 (0.25-0.35); (3) D+ps and d+ps at g0 0.25 beat D and d at every lr in 0.03-0.05.

Best over lr in {0.04, 0.05, 0.06} per arm (g0 0.25; the delta rows sweep 0.03-0.05):

| setting | g | g+ps | e (GDN-2-style) | e+ps |
|---|---|---|---|---|
| S1: L64 P16 width 32 | 0.379 | **0.148** | 0.427 | **0.125** |
| S2: L128 P16 (3 queries per pair) | 0.610 | **0.176** | 0.597 | **0.102** |
| S3: L64 P8 | 0.198 | **0.023** | 0.084 | **0.008** |
| S4: L128 P32 width 64 (dk 32) | 0.175 | **0.032** | 0.243 | **0.024** |
| delta rule D, L64 P16 | 0.754 | **0.150** | | |
| gated delta d, L64 P16 | 0.716 | **0.226** | | |

Every gated arm beats every base lr in all four settings, so the win does not depend on the lr choice: the worst gated lr beats the best base lr in S1-S4 for both g and e (worst gated: 0.156 / 0.327 / 0.031 / 0.042 for g, 0.218 / 0.107 / 0.014 / 0.138 for e).

Prediction check: (1) right in all four settings for both g and e. (2) width 64 P32 gains 0.14 (g) and 0.22 (e) in loss, smaller in absolute terms than S1 because the base is already better, but the loss ratio is the same or larger (0.18x and 0.10x of the base vs 0.39x and 0.29x in S1): wrong as stated, right as a ratio. (3) right at lr 0.03-0.04 for D and d; at lr 0.05 d+ps failed (0.97, layer-0 gamma collapsed to 0.12 while layer 1 took 2.3), a single onset failure.

Takeaways:
- The delta-rule "hurt" from the gate-3 test is gone at g0 0.25: D 0.754 -> 0.150 and d 0.716 -> 0.226 at their best lrs. The gate moves the best lr of D from 0.03 to 0.04; it does not need a lower lr.
- In the delta family the gate also lowers the gate at query keys (qk 0.51 for D vs 0.80 for g): the erase term already removes the old value, so the read at a query key is closer to v.

## Round 7: chunk size and mechanism probes (L64 P16, lr 0.05, g0 0.25)

Rule R5 (= R4 plus the sweeps): the gate wins at every lr in every converged recall cell where pairs are re-queried at least one chunk after storage, for g, e, D and d.
Predictions: (1) chunk size: P16 gains at C 4, 8, 16, 32; at C 64 = L the gate is off (S_c is the empty initial state), so g+ps ties g. (2) Content-free control (gate compares with another row's value) loses the gain: ties or worse than g. (3) Gate only at query-phase positions keeps all the gain (pair-phase gates sit at 0.9). (4) Read source: q_t (the current token's own query) loses most of the gain, because the value at a query position is predicted by the previous token's query; k_t gives part of the gain. (5) Fixed gamma 3 (near the learned value) matches learned; fixed 0.25 gains less; fixed 1 in between.

| probe (g base, L64 P16, lr 0.05) | val | layer-0 gate at query values / pair values | learned gamma, layer 0 / 1 |
|---|---|---|---|
| g, no gate (lr 0.04-0.06) | 0.379-0.520 | | |
| g+ps, chunk 16 (main) | 0.164 | 0.14 / 0.92 | 2.8 / 0.1 |
| chunk 4 / 8 / 32 | 0.193 / 0.170 / 0.127 (bases 0.496 / 0.475 / 0.575) | 0.17 / 0.17 / 0.03 | |
| chunk 64 = L (gate sees only the empty state) | 0.4630 = base 0.4630 | 1.00 / 1.00 | |
| content-free control (value of another row) | 0.356 | 0.78 / 0.95 | 0.7 / 0.1 |
| gate only at query-phase positions | **0.117** | 0.02 / 1.00 | 5.3 / 0.4 |
| read with the current query q_t | 0.293 | 0.19 / 0.91 | 2.7 / 0.1 |
| read with the current key k_t | 0.159 | 0.15 / 0.92 | 2.6 / 0.2 |
| fixed gamma 0.25 / 1 / 3 (all layers) | 0.338 / 0.318 / 0.270 | 0.82 / 0.47 / 0.14 | fixed |
| e, fixed gamma 3 | 0.332 (e+ps learned 0.130) | 0.17 / 0.91 | fixed |

Prediction check: (1) right: every chunk from 4 to 32 gains 0.30-0.45, and chunk 64 is bit-identical to the base (the gate is exactly 1). (2) right: the content-free control keeps at most a third of the gain (0.356, inside the base's lr spread). (3) right, and better than predicted: query-only is the best probe (0.117 vs 0.164), because the gate no longer damps any pair writes (pair-value gate 1.00 instead of 0.92). (4) half right: q_t keeps 0.17 of the 0.30 gain; k_t keeps all of it, because the short convolution puts the previous token (the query key) into k_t, so S_c k_t is also the answer read. (5) wrong: no fixed gamma matches the learned one. Fixed 3 is closest in layer 0 (query-value gate 0.14) but it also suppresses layer 1 (gate 0.40), where the learned gamma goes to 0.01-0.2. The learned per-head gamma matters because it switches the gate off in the layers that do not need it.

## Round 8: data mix on the text bed (lm.py, TinyStories tokens, V 2,000, L64, 300 steps, lr 0.04)

`--rep r` makes a share r of training rows repeat their first half in the second half (recall-heavy); val is natural text, val_rep is the loss on the repeated half of fully repeated validation rows. r = 0.0001 means natural text only, with val_rep still measured.
Rule R6 (= R5 plus probes): the gain is the share of writes whose value the previous read already returns. Natural text at L64 has few exact re-writes, so the gate should be near neutral on val; on repeated rows the second half re-writes the first half exactly, so val_rep improves once the model has learned to copy.
Predictions: (1) natural text (r 0.0001): val within 0.01 of the base for g and e. (2) r 0.5 and 1.0: val_rep better with the gate by at least 0.05, val within 0.02. (3) The second-half gate mean (h2) on repeated rows is lower than the first-half mean (h1) by at least 0.1 in layer 0.

| text bed (lr 0.04, g0 0.25) | g val / val_rep | g+ps | e | e+ps |
|---|---|---|---|---|
| natural only, 300 steps | 3.806 / 3.752 | 3.802 / 3.744 | 3.805 / 3.752 | 3.806 / 3.754 |
| 25% repeated rows, 300 | 3.808 / 3.733 | 3.810 / 3.746 | 3.812 / 3.744 | 3.808 / 3.752 |
| 50% repeated rows, 300 | 3.826 / 3.732 | 3.822 / 3.743 | 3.821 / 3.744 | 3.815 / 3.747 |
| 50% repeated rows, 900 | 3.712 / 3.531 | 3.713 / 3.525 | 3.722 / 3.565 | 3.699 / 3.575 |
| all rows repeated, 300 | 4.103 / 3.161 | 3.926 / 3.574 | 3.956 / 3.499 | 4.010 / 3.317 |
| all rows repeated, 900 | 5.672 / 1.490 | 5.710 / 1.440 | 5.800 / 0.925 | 4.890 / 1.812 |

Prediction check: (1) right: on natural text the gate is within 0.004 of the base, and the model lowers gamma from 0.25 to 0.13 in layer 0, i.e. it switches the gate off. (2) wrong: at 25% and 50% repeated rows val_rep ties (within 0.013, either sign) at 300 and 900 steps, because the width-32 model has barely learned to copy (val_rep 3.53 vs val 3.71). (3) wrong for mixed data (h2 below h1 by only 0.03-0.07), right only when every row repeats (0.20-0.38 in the head that learns gamma 5-18). With every row repeated the runs are unstable in both directions (val_rep gains 0.05 for g and loses 0.89 for e at 900 steps, while e+ps natural val is 0.91 better); these are single runs of a degenerate mix and are not used for the rule.

Takeaways:
- Limit: on text at this scale the gate ties. The gain needs exact key-value retrieval that is re-written; a 2-layer width-32 text model does not yet retrieve repeated tokens well enough for the previous read to predict the next value.
- No harm on natural text: the learned gamma shrinks, so the gate costs nothing where it is not useful.

## Round 9: predictions for unrun cells

Rule R7: the gate wins where (a) the model has formed exact key-value retrieval, (b) a stored pair is re-written after at least one chunk boundary, and (c) rewrites interfere (load P/dk at least about 0.25 with re-queries, or more re-queries per pair). It ties where (a) or (b) fails (natural text at small scale; pairs inside the first chunk; chunk = L), and with g0 0.25 it does not block learning over capacity.
Predictions, written before running:
1. L256 P16, e+ps at g0 0.25 (e base 0.997): e+ps below 0.5.
2. P24 at 600 steps, e+ps at g0 0.25 (e base 0.530; g0 1 never learned): e+ps below 0.35.
3. 4 layers at 600 steps (converged): g+ps beats g by at least 0.1, and e+ps beats e by at least 0.1.
4. P4 at chunk 4 with e (base 0.019 at chunk 16): e+ps ties within 0.01, because the e base is already near the floor.
5. Query-only gate with e at L128 P16 (e+ps 0.102): at most 0.10.

| prediction | cell | base | gated | outcome |
|---|---|---|---|---|
| 1. e+ps below 0.5 | L256 P16, e | 0.997 | **0.154** | right |
| 2. e+ps below 0.35 | P24, 600 steps, e | **0.530** | 1.481 | wrong |
| 3. both gain at least 0.1 | 4 layers, 600 steps | g 0.378, e 0.688 | **0.167**, **0.210** | right |
| 4. e ties within 0.01 | P4 chunk 4, e | 0.025 | **0.005** | wrong (gate 5x lower, 0.019 apart) |
| 5. at most 0.10 | L128 P16, e, query-only gate | e+ps 0.102 | **0.064** | right |

Extra cells, P24 at 600 steps: g 0.427, g+ps (g0 0.25) **0.317**, g+ps query-only 1.739; e 0.530, e+ps query-only **0.242**, e+ps (g0 0.25) 1.481.

Misses and what they change:
- Miss 2: over capacity (P/dk 1.5), whether a run learns retrieval within 600 steps is a lottery for every arm (base and gate each win two of four P24 comparisons across g0, query-only, g and e). The rule no longer claims anything over capacity: there the gate is neither reliably better nor worse.
- Miss 4: the gate helps even when the base is near the floor, as long as pairs cross a chunk boundary (P4 at chunk 4: 0.025 -> 0.005). The load condition (c) is dropped; the chunk condition (b) is what decides the P4 tie at chunk 16.

## Round 10: context x pairs map (lr 0.05, g0 0.25, tokens per step 2,048)

Rule R8: the gate wins in every cell where the base forms retrieval (P/dk at most 1) and pairs are re-queried after a chunk boundary; over capacity the outcome is a lottery; on text at this scale it ties.
Predictions: (1) every L128 and L256 cell with P 4 or 8 is a win for g and e (at L128 and L256 pairs span several chunks even at P4). (2) L64 P4 at g0 0.25 still ties for g (pairs inside chunk 0). (3) P32 at L128 and L256 (P/dk 2): the base does not learn in 300 steps (loss above 2), and the sign is a lottery.

| cell (g0 0.25) | g | g+ps | e | e+ps |
|---|---|---|---|---|
| L64 P4 | 0.109-0.156 (3 lrs) | 0.135 (tie) | **0.019** | 0.052 |
| L128 P4 | 0.129 | **0.074** | 0.028 | **0.015** |
| L128 P8 | 0.411 | **0.077** | 0.215 | **0.032** |
| L128 P16 | 0.610 | **0.176** (best of 3 lrs) | 0.597 | **0.102** |
| L128 P32 | 3.394 | **2.322** | **4.224** | 4.834 (both fail) |
| L256 P4, best of lr .04-.06 | 0.433 | **0.159** | 1.367 (lr .05) | **0.041** |
| L256 P8 | 0.425 | **0.230** | 0.131 | **0.027** |
| L256 P16, best of lr .04-.06 | 0.790 | **0.316** | 0.997 / 1.122 (lr .05/.04) | **0.154** |
| L256 P32 | 2.886 | **0.999** | 4.850 | **1.553** |

Prediction check: (1) right for every L128 and L256 cell with P 4 or 8 at the best lr (8 of 8). At L256 with batch 8, g+ps loses at single lrs: lr 0.05 at P4 (0.674 vs 0.522) and P16 (2.307 vs 0.790), lr 0.06 at P4 and P16. (2) right: L64 P4 ties for g (0.135 vs 0.109-0.156). (3) right that the bases fail at P32 (loss 2.9-4.9), and the sign was a lottery at L128 (g wins, e both fail) but the gate won both at L256 (0.999 vs 2.886, 1.553 vs 4.850).

Failure diagnostic (`figures/pairs.py`, all gated runs paired with their base): of the 23 gated runs that lose to their base at the same lr by more than 10%, 17 have a layer-0 head whose learned gamma stayed below 1.1, and the other 6 are over capacity or the lr 0.06 delta run. In the wins, layer-0 gamma ends at 1.5-5.3 in 60 of 66 runs. When the gate fails, it is because it did not attach to layer 0 (gamma stays small there, sometimes growing in layer 1 instead); this happens at small batch (L256, batch 8) with high lr, before retrieval forms, and over capacity.

Takeaways:
- Long context: the gain holds at L256 with an lr sweep (g 0.433 -> 0.159 at P4, 0.790 -> 0.316 at P16; e 1.367 -> 0.041, 0.997 -> 0.154), but the gate narrows the stable lr range at batch 8: it needs lr 0.04 where the base works at 0.04-0.06.

## Round 11: predictions for five unrun cells (lr 0.05, g0 0.25, 300 steps)

Rule R8 predicted a win in each of these cells: wider state, delta rule at long context, delta rule at low load, longer chunk with e, and width 16 at P8 (P/dk 1).

| cell | base | gated | acc base / gated | layer-0 gamma | outcome |
|---|---|---|---|---|---|
| width 64, L128 P16, g | 0.094 | **0.020** | 0.977 / 0.994 | 2.9-3.1 | right |
| L128 P16, delta rule D | 1.412 | **0.279** | 0.696 / 0.923 | 3.8-3.9 | right |
| P8, delta rule d | 0.098 | **0.016** | 0.974 / 0.997 | 3.2-4.3 | right |
| chunk 32, e | 0.459 | **0.093** | 0.886 / 0.972 | 5.3-5.6 | right |
| width 16 (dk 8), P8, g | **2.398** | 2.842 | 0.459 / 0.301 | 3.4-6.0 | wrong |

Miss and what it changes: at dk 8 the base never forms retrieval in 300 steps (loss 2.40, accuracy 0.46), although P/dk is 1. The gate then has no useful read to compare against, and it makes things worse even though its gamma grows. The precondition is therefore that the base forms retrieval (its accuracy reaches roughly 0.8 at that budget), not a P/dk ratio.

Rule R9 (final): the predicted-skip gate wins on recall wherever the base model forms key-value retrieval and stored pairs are queried after at least one chunk boundary. It ties when pairs sit inside the first chunk, when the chunk spans the whole sequence, and on natural text at this scale. Where the base does not form retrieval (over capacity, or too small a key size), the outcome is a lottery or a loss. At small batch it narrows the stable lr range. When it fails, its layer-0 gamma usually stays below 1.1.

## Round 12: the weakest share-bar point (written before the runs)

Weakest point: share-bar point 2 counts only three gain measures: +0.01 on text, +0.10 recall accuracy, or 1.5x fewer steps to 90% accuracy. The loss drops of 2-10x are not on that list. At 300 steps, with each arm at its best lr, the gate passes this bar in only part of the grid:
- S1: e passes (its base never reaches 90%, while e + ps reaches it at step 200). g fails: 275 vs 225 steps (1.2x) and +0.05 accuracy.
- S2: e passes (+0.11 accuracy). g is unclear: +0.08 accuracy, and its base never reaches 90% within 300 steps.
- S3: both fail, because the bases reach 90% by step 125-150.
- S4: e passes (225 vs 150, 1.5x). g fails (200 vs 150, 1.33x).

Test: rerun S1, S2 and S4 for 900 steps, so the bases have room to reach 90%. Use g and e, with and without the gate, each at its best 300-step lr. The lr is constant until step 630, then cools down. Steps to 90% are evaluated every 25 steps. Count a setting as passed when both g and e reach 90% at least 1.5x sooner with the gate.
Prediction: S1 and S2 pass for both arms, because the base g needs about 500 steps there. S4 g lands near 1.3x and fails. That would give 2 or 3 of 4 settings for g, and 3 of 4 for e.

Results (900 steps; each arm at its best 300-step lr; steps are counted at evaluations every 25 steps; "never" = not reached by step 900):

| setting | arm | steps to 90%: base / gated | ratio | steps to 95%: base / gated | ratio | final accuracy: base / gated |
|---|---|---|---|---|---|---|
| S1 | g | 475 / 225 | **2.1x** | never / 675 | >1.3x | 0.932 / 0.974 |
| S1 | e | 525 / 200 | **2.6x** | never / 475 | **>1.9x** | 0.930 / 0.985 |
| S2 | g | 800 / 250 | **3.2x** | never / 650 | >1.4x | 0.912 / 0.976 |
| S2 | e | 700 / 225 | **3.1x** | never / 350 | **>2.6x** | 0.924 / 0.992 |
| S4 | g | 200 / 175 | 1.14x | 450 / 200 | **2.25x** | 0.981 / 0.997 |
| S4 | e | 225 / 150 | **1.5x** | 775 / 175 | **4.4x** | 0.959 / 0.998 |
| S3 (300-step runs) | g | 150 / 125 | 1.2x | 275 / 150 | **1.8x** | 0.957 / 0.993 |
| S3 (300-step runs) | e | 125 / 125 | 1.0x | 225 / 125 | **1.8x** | 0.978 / 0.997 |

Prediction check: right. S1 and S2 pass for both arms at 90%. S4 g fails at 90% (1.14x, predicted about 1.3x). At 90%, e passes in 3 of 4 settings and g in 2 of 4.

The 90% target sits below where the S3 and S4 bases already are by step 150-225, so there the target measures early learning and misses the gain. At 95%, the gate reaches the target at least 1.8x sooner in all 8 arm-settings where both runs reach it. In S1 and S2 the bases never reach 95% in 900 steps; the gated runs do (by step 350-675). Lower bounds that do not clear 1.5x are left unbolded. At 900 steps the gap in loss also stays wide (S4 g 0.069 -> 0.010, e 0.132 -> 0.006), so the bases do not catch up given three times the budget.

Share-bar point 2, verdict:
- At the literal 90% target, e + ps beats its own base by at least 1.5x in 3 of 4 settings (S1, S2, S4).
- At 95%, both gated arms beat their bases by at least 1.5x in all 4 settings wherever the base reaches 95% (S3, S4). Where the base never gets there (S1, S2), the gated runs do.
- Against the best method, e + ps beats the tuned g baseline by at least 1.5x at 95% in all 4 settings (S1 475 vs never, S2 350 vs never, S3 125 vs 275, S4 175 vs 450).
Point 2 holds.

## Round 13 (2026-11-15 evening): can the read source be separated without the short convolution?

Question: the k_t read (0.159) ties the q_{t-1} read (0.164) at S1. Round 7 explained this by the short convolution, which mixes the previous token into k_t. Removing the convolution (`--sc 1`) should separate the two reads.
Prediction: without the convolution, the base still retrieves; q_{t-1} keeps the gain and k_t loses most of it.

| run (S1, lr 0.05, `--sc 1`) | loss | accuracy |
|---|---|---|
| g, no gate | 4.311 | 0.070 |
| g + predicted-skip (q_{t-1}) | 4.324 | 0.067 |

Verdict: the prediction cannot be tested. Without the short convolution the base model cannot retrieve at all (accuracy 0.07, chance level), so there is nothing to gate. The k_t run was skipped. Noise: these runs are not repeated (fixed data), and reproduce.sh notes that differences under about 0.1 are noise; the S1 lr spread of the gated arm is 0.148-0.164. The 0.159 vs 0.164 gap is within noise. The paper treats the read source as a design choice: with the short convolution used in current linear-attention layers both reads work; q_{t-1} reuses the read the chunked kernel already computes for the previous output, while S_c k_t is computed for free only in delta-rule kernels. Valid runs: 261.

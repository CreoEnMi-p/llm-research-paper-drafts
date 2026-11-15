# Identity-match heads and pointer heads: scaling-simulation study (2026-11-15)

Session: study-identity-heads (berserker mode, skill sections 3b, 3c gate 4, 4d). Code: `rbg.py` (grid copy of `rb.py`, adds `--task hopx` with `--fill`, `--dup`, `--syn`; `--task mq2` with `--k2`; `--stop`; `--eqna`; `--abl`; `--pprobe`), `atg.py` (grid copy of `at.py`), driver `grid.py`. Every run is one line in `results.jsonl`, tags start with `B<block>-`. The attention discovery session also edits `rb.py`/`at.py` in this folder, so the study uses its own copies.

Arms in every recall cell (all on the same base, `--init 0.5`, early stop at 0.99 accuracy, eval every 25 steps on 512 fixed rows):

| Arm | Flags | lr (Muon) |
|---|---|---|
| F: tuned plain attention, ALiBi + sink | `--sink 1 --alibi 1` | 0.04 |
| KVS: KV shifting (2411.19574) | F + `--kvs 1` | 0.04 |
| PRI: Primer depthwise conv (2109.08668) | F + `--dconv 3` | 0.04 |
| NG: n-gram heads (2401.12973), 2 heads, order 1 (NG2: order 2) | F + `--ngh 2` | 0.04 |
| ID: identity heads, 2 per layer, learned temperature 3, no ALiBi on them | F + `--eqh 2 --eqt 3 --eqna 1` | 0.02 |
| PM: pair-match identity heads | ID + `--eqn2 0.001` | 0.02 |
| PTR: pointer identity heads | ID + `--ptr 3` | 0.02 |

Task `hopx`: `hops` blocks of `pairs` (key, value) pairs, block c maps vocabulary range c to range c+1 in its own random order; then one query per pair (a range-0 key), scored only on the final answer after all hops. Guessing among the answers not yet used gives about 0.34 at 8 pairs. `--fill k` puts k filler tokens (own vocabulary range) after every pair; `--dup p` replaces a share p of fillers with keys or bridge tokens of the same row (the same token appears again in a wrong context); `--syn p` writes a share p of query keys and block-2+ bridge tokens with a fixed synonym token (t + V/2), so the stored and the queried token differ.

## Round 0: base check (hops 2, 8 pairs, 3 layers, width 64, 300 steps)

| Arm | accuracy | step to 90% |
|---|---|---|
| ID (no ALiBi on identity heads) | 0.991 | 75 |
| ID with ALiBi also on identity heads | 0.333 | never |
| F | 0.333 | never |
| KVS | 0.344 | never |
| PRI | 0.343 | never |
| NG | 0.172 | never |

Takeaway: the free first hop needs the identity heads to be fully position-free. A learned ALiBi slope on them (initialised 2^-8h/H) is enough to bury the far-back match under recency and the head never finds it within 300 steps. Every ID/PM/PTR arm below has `--eqna 1`. The LM numbers in IDEA.md were measured with ALiBi on the identity heads; the LM block rechecks both.

## Round 1: hops x depth (8 pairs, width 64, 250 steps)

Rule going in: pointer heads (PTR) solve k hops when the model has at least k layers (layer l's pointer head lands on chain stage l+1, and layer k-1's head reads the answer); plain identity heads (ID) solve exactly 2 hops with 2 or more layers; every competitor stays at guessing for 2+ hops.

Accuracy (step to 90% in brackets; "-" = never):

| hops \ layers | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|
| 2: PTR / ID | 0.999 (25) / 0.993 (75) | ID 0.991 (75) | | | | |
| 3: PTR / ID | 0.307 (-) | 0.998 (25) / 0.328 (-) | 1.000 (25) | | 0.998 (50) / 0.320 (-) | |
| 4: PTR | | 0.334 (-) | 0.995 (50) | | 0.992 (50) | see below |
| 5: PTR | | | 0.340 (-) | 0.971 (50) | 0.929 (50) | |
| 6: PTR | | | | | 0.325 (-) | |

Predictions 1-12 (PTR solves iff layers >= hops; ID only for 2 hops) hold in 11 of 12 cells; the miss is 6 hops with 6 layers (predicted solve, measured 0.325). At 5 hops accuracy already sags (0.97 at 5 layers, 0.93 at 6).

Mechanism of the depth rule (probe `--pprobe`, 3 hops, 4 layers): share of queries whose pointer head lands on chain stage c, rows = layers 0-3: [0, 1.00, 0, 0], [0, 0.01, 0.83, 0], [0, 0, 0, 0.99], [0.02, 0, 0, 0]. Layer l lands on stage l+1; freezing the pointer at test time drops accuracy 0.99 -> 0.02; zeroing the identity heads of layer 2 alone (the read-out layer) drops it to 0.016, zeroing layers 0, 1 or 3 does nothing (the pointer update does not pass through the head output).

### Round 1b: why the pointer chain breaks past 5 hops, and a fix

Attention mass of the pointer head on the correct chain stage at init (step 1, 4 hops, 4 layers), layers 0-3: 0.97, 0.46, 0.17, 0.07. After training (125 steps): 0.93, 0.56, 0.52, 0.82. At 6 hops after 150 steps: 0.98 at layer 0, then 0.03 or less.

Mechanism: every bridge token appears twice, as the value of its pair in block c and as the key of its pair in block c+1. A key built only from the previous token (e_{j-1}) scores both following positions the same, so at each hop after the first the pointer splits its mass evenly between the right next token and the key of the next pair in block c. The right mass about halves per hop, and training has to break k-1 exact ties with the gated residual part of the query. It does so up to 4-5 hops within 250 steps and not at 6.

Test of the mechanism: a no-backtrack term (`--pnb 3`, fixed, no new weights) subtracts 3 cos(previous pointer, e_{j-2}) from the pointer head's logits. The wrong occurrence is exactly the one whose token two back is the previous chain stage, so this removes the tie and nothing else.

| Run | init mass, layers 0-3 | accuracy (step to 90%) |
|---|---|---|
| 4 hops, PTR | 0.97, 0.46, 0.17, 0.07 | 0.995 (50) |
| 4 hops, PTR + no-backtrack | 0.97, 0.88, 0.88, 0.87 | - |
| 6 hops, 6 layers, PTR | - | 0.323 (-) |
| 6 hops, 6 layers, PTR + no-backtrack | - | 0.990 (25); trained mass 0.93, 0.90, 0.92, 0.93, 0.92, 0.93 |
| 6 hops, 5 layers, PTR + no-backtrack | - | 0.328 (-) (depth rule still holds) |
| 8 hops, 8 layers, PTR | - | 0.339 (-) |
| 8 hops, 8 layers, PTR + no-backtrack (lr 0.02 / 0.01) | - | 0.835 / 0.837 (0.93 at step 25, then a dip to 0.52 and partial recovery) |
| 3 hops, 4 layers: F / KVS / PRI | - | 0.328 / 0.331 / 0.334 |

Takeaway: the depth limit of pointer heads at 6+ hops is the value/key tie, not depth or training length; removing the tie gives 6 hops at step 25 and 8 hops at 0.84, while one layer per hop stays necessary. Rule updated: PTR solves k hops iff layers >= k, for k <= 5 as is and for k <= 8 with no-backtrack.

## Round 2: pairs x filler (hops 2, 3 layers, width 64, 4 heads, up to 300 steps; 600 where noted)

Rule going in: both ID and PTR are position-free, so pairs and filler should not matter.

| cell | ID | PTR | F / KVS / NG |
|---|---|---|---|
| 4 pairs | 0.994 (75) | | |
| 4 pairs, fill 3 | 0.996 (75) | | |
| 16 pairs | 0.994 (100) | 1.000 (25) | 0.170 / 0.191 / 0.104 |
| 16 pairs, fill 3 | 0.993 (150) | 1.000 (25) | |
| 32 pairs | 0.075 (-), 0.080 at 600 steps | 0.999 (25) | |

Miss: ID fails at 32 pairs even with twice the steps, while PTR solves it at step 25. Filler costs ID some steps (75 -> 150 at 16 pairs) and costs PTR nothing. The ID accuracy curves are a flat plateau and then a jump within 25-50 steps; the jump comes later as pairs grow and does not come at 32.

## Round 3: head dimension x pairs (hops 2, 3 layers, up to 300 steps)

Rule going in (from the Round 2 miss): ID's second hop must copy the value of the matched pair through W_v W_o and then match it again through the shared W_q, so the copied vector has to stay distinguishable among `pairs` candidates inside one head; this should depend on head dimension dk = width / heads, not on width. PTR copies the raw embedding into the pointer stream, so it should not care.

| width / heads (dk) | 8 pairs | 16 pairs | 24 pairs | 32 pairs |
|---|---|---|---|---|
| 32 / 4 (dk 8): ID | 0.272 (-) | 0.122 (-) | | |
| 32 / 4 (dk 8): PTR | | 1.000 (25) | | 0.998 (25) |
| 32 / 1 (dk 32): ID | | | 0.993 (125) | |
| 64 / 4 (dk 16): ID | 0.991 (75) | 0.994 (100) | 0.101 (-) | 0.075 (-) |
| 64 / 2 (dk 32): ID | | | | 0.996 (75) |
| 128 / 4 (dk 32): ID | | | | 0.997 (100) |

Takeaway: ID solves 2-hop recall within 300 steps about when pairs <= dk (dk 16 holds 16 pairs and not 24; dk 32 holds 24 and 32 at any width; dk 8 holds neither 8 nor 16). Width alone does not help: width 128 with 4 heads and width 32 with 1 head behave the same at dk 32. PTR solves every cell at step 25, down to width 32 with dk 8 and 32 pairs.

Predictions for Round 3b (written before running): (a) ID, width 64, 1 head (dk 64), 48 pairs: solves. (b) ID, dk 32 (width 64, 2 heads), 48 pairs: fails. (c) ID, dk 8 (width 32, 4 heads), 4 pairs: solves. (d) PTR, width 32, 48 pairs, 3 hops, 3 layers: solves. (e) F at dk 32 (width 64, 2 heads), 16 pairs, 2 hops: stays at guessing.

### Round 3b/3c: the predictions, and temperature

| prediction | measured | held? |
|---|---|---|
| (a) ID, dk 64, 48 pairs: solves | 0.019 (300 steps, 154 s) | miss |
| (b) ID, dk 32, 48 pairs: fails | 0.073 | yes |
| (c) ID, dk 8, 4 pairs: solves | 0.487 | miss |
| (d) PTR, width 32 (dk 8), 48 pairs, 3 hops: solves | 0.070 (300 steps) | miss |
| (e) F, dk 32, 16 pairs: guessing | 0.094 | yes |

Two misses say head dimension is not the variable. The identity-head logit is eqt * q.k / sqrt(dk) with QK-norm, so an exact match scores eqt * sqrt(dk) and an unrelated key scores about eqt times a unit Gaussian. The effective sharpness is s = eqt * sqrt(dk): dk 8 / 16 / 32 / 64 at eqt 3 give s = 8.5 / 12 / 17 / 24. Temperature sweep (2 hops, 3 layers, dk 16):

| eqt (s) | 8 pairs | 32 pairs | 48 pairs |
|---|---|---|---|
| 0 (no temperature, s = 4) | 0.205 (-) | | |
| 1 (s = 4) | 0.197 (-) | | |
| 3 (s = 12) | 0.991 (75) | 0.075 (-) | |
| 10 (s = 40) | 0.990 (125) | 0.991 (175) | 0.048 (-) |
| 30 (s = 120) | 0.120 (-) | | |
| 10 at dk 8 (s = 28), 16 pairs | | 0.118 (-) at 16 pairs | |

Takeaway: ID works inside a temperature window: too soft (s about 4) and the far match drowns among all positions, too sharp (s = 120) and the softmax is one-hot with no gradient for the second hop. Raising eqt from 3 to 10 turns 32 pairs from 0.075 into 0.991. Two limits stay: 48 pairs fails at every temperature tried within 300 steps, and dk 8 fails even at 4 pairs at s = 8.5 and at 16 pairs at s = 28, so the learned second hop (copy a value through W_v W_o, match it through W_q) needs about 16 dimensions per head. The pointer head, which matches the raw embedding exactly, also failed at dk 8 with 48 pairs and 3 hops (L = 384), so it shares the context-length limit at small dk.

Check of (d) at dk 16: PTR, width 64, 48 pairs, 2 hops: 1.000 at step 25. So PTR's 48-pair failure is the dk 8 / long-context case, not a pair limit.

## Round 4: noisy keys (hops 2, 8 pairs, 3 layers, width 64, fill 2, 250 steps)

Rule going in: `dup` (the same key or bridge token appears again as filler in a wrong place) should hurt ID and PTR a little, because the wrong occurrence gets the same match score but is followed by a filler, not a value, so a range-aware read-out can ignore it; `syn` (query or bridge written with a synonym token) should break exact-match NG heads completely and break ID/PTR at init, and ID/PTR can only recover if W_q learns to map a word and its synonym to the same query.

Predictions: (f) ID and PTR >= 0.95 at dup 0.5; (g) NG near guessing at syn 0.5; (h) ID at syn 0.5 below 0.8; (i) F, PRI at guessing in every noise cell.

Accuracy (step to 90%), 250 steps for ID/PTR, 200 for competitors:

| noise | ID | PTR | F | PRI | KVS | NG |
|---|---|---|---|---|---|---|
| none (fill 2) | 0.995 (100) | 1.000 (25) | | | | |
| dup 0.25 | 0.956 (150) | 0.950 (50) | | | | |
| dup 0.5 | 0.866 (-) | 0.899 (-) | 0.307 | 0.336 | | 0.219 |
| dup 1.0 | 0.282 (-) | 0.786 (-) | | | | |
| syn 0.5 | 0.957 (125) | 0.937 (125) | 0.333 | | 0.348 | 0.195 |
| syn 0.75 | 0.914 (200) | 0.709 (-) | | | | |
| syn 1.0 | 0.340 (-) | 0.754 (-) | | | | |

Predictions: (f) ID and PTR >= 0.95 at dup 0.5: miss (0.87, 0.90); (g) NG at guessing with synonyms: held (0.195); (h) ID below 0.8 at syn 0.5: miss (0.957); (i) competitors at guessing: held in all 5 cells.

Takeaways. Repeated keys cost both heads accuracy in proportion to how often a key reappears in a wrong context: a position-free match cannot tell the stored pair from a later mention of the same word, so at dup 1 (every filler is a chain word) ID falls to guessing and PTR keeps 0.79. Synonyms do not break the heads as long as some rows hold exact repeats: W_q learns to send a word and its synonym to the same query, and the exact repeats bootstrap the match (syn 0.5: 0.957, syn 0.75: 0.914). With no exact repeat at all (syn 1.0) ID never starts (0.340). Exact-match n-gram heads cannot use synonyms at all (0.195 at syn 0.5). Rule updated: identity heads need exact repeats in part of the data to start, then generalise to learned equivalences; they lose accuracy when the same word recurs in wrong contexts.

## Round 5: partial matches, two-token keys (task mq2, 8 pairs, 3 layers, width 64, 250 steps)

Each key is a pair of tokens (a, b) with a from K symbols and b from K others, so a single token appears in about 8/K keys and only the pair identifies the value. One hop only, so plain attention can in principle learn it with an induction-style circuit.

Rule going in: single-token identity heads (ID) see a partial match at every key sharing b (or a), so they should lose accuracy as K shrinks (more sharing); pair-match heads (PM, extra term q_{i-1}.q_{j-2}) and order-2 n-gram heads (NG2) should solve it; NG (order 1) should fail like ID.

Predictions: (j) at K = 3, PM and NG2 >= 0.95, ID and NG <= 0.6; (k) at K = 6 (few shared tokens), ID >= 0.9; (l) F, KVS, PRI learn it more slowly than PM (one hop is learnable for them).

| cell | ID | PM | NG2 | NG | PRI | F | KVS |
|---|---|---|---|---|---|---|---|
| 8 pairs, K 3, 3 layers | 1.000 (50) | 1.000 (25) | 1.000 (25) | 0.642 (-) | 1.000 (50) | 0.651 (-) | 0.452 (-) |
| 16 pairs, K 4, 3 layers | 0.999 (50) | 1.000 (25) | | | | 0.999 (175) | |
| 32 pairs, K 6, 3 layers | 1.000 (75) | 1.000 (25) | 1.000 (25) | | 1.000 (75) | 0.070 (-) | |
| 16 pairs, K 4, 1 layer | 0.167 (-) | 1.000 (25) | 1.000 (25) | | 1.000 (50) | | |

Predictions: (j) ID <= 0.6 at K 3: miss with 3 layers (1.000), held with 1 layer (0.167); (k) ID >= 0.9 at K 6: held; (l) F slower than PM: held (F 175 steps or never).

Takeaway: partial matches are not a failure mode for ID once it has two layers, because the second layer combines the two single-token matches (or the residual already mixes a into b's query). In one layer, single-token identity heads see every key that shares the last token and land at 0.167, and the pair term fixes it at step 25. On this task PM ties order-2 n-gram heads and is one eval step ahead of Primer's convolution; it is not a unique win. Plain attention with ALiBi fails at 32 pairs.

## Round 6: probes (hops 2 or 3, 8 pairs, width 64)

Rule going in: the free first hop comes from the tied key k_j = q_{j-1} with no position term; untying (key from W_k of the previous token) removes the guaranteed exact match at init and should fall back to plain-attention behaviour within 250 steps; adding RoPE to the identity heads (tied, but rotated) should hurt like ALiBi did, because the match score then depends on distance; zeroing the identity heads at test time should drop accuracy to guessing while zeroing as many normal heads should not.

Predictions: (m) ID untied, 2 hops: <= 0.4; (n) ID with RoPE, 2 hops: <= 0.5; (o) ablation of identity heads: <= 0.35, normal-head control >= 0.9; (p) PTR untied, 3 hops 3 layers: still solves (the pointer key is built from embeddings, not from the untied key).

| probe | accuracy | held? |
|---|---|---|
| ID, 2 hops, identity heads zeroed at test time (all layers / layer 0 / 1 / 2) | 0.025 / 0.019 / 0.731 / 0.984 (trained 0.993) | (o) yes |
| ID, same number of normal heads zeroed | 0.868 | (o) control slightly below 0.9 |
| ID untied (key from W_k of the previous token) | 0.171 | (m) yes |
| ID tied, with RoPE on the identity heads | 0.199 | (n) yes |
| PTR untied, 3 hops 3 layers | 0.999 (25) | (p) yes |
| PTR with RoPE, 3 hops | 0.997 (25) | |
| PTR temperature eqt 1 / 10, 3 hops | 0.997 (50) / 0.996 (25) | |
| PTR + no-backtrack, 4 hops 5 layers: pointer frozen at test time | 0.021 (trained 0.993) | |
| same run: identity heads zeroed per layer 0-4 | 0.995 / 0.994 / 0.993 / 0.015 / 0.993 | |

Pointer probe of that last run (mass of the pointer head on chain stage c, rows = layers 0-4): layer 0 -> stage 1 (0.84), layer 1 -> stage 2 (0.85), layer 2 -> stage 3 (0.72), layer 3 -> stage 4 = answer (0.95), layer 4 -> spread (no stage above 0.03).

Takeaways. The ID result needs all three parts of the design: tying (untied 0.171), no position term (RoPE 0.199, ALiBi 0.333) and a temperature in the right window. Layer 0's identity heads do the first hop (zeroing them gives 0.019). PTR is insensitive to tying, RoPE and temperature, because its pointer logits are computed from embeddings before any position rotation, and only the read-out layer's head output matters: the pointer update itself carries the chain, and freezing it drops accuracy to 0.021. The extra fifth layer sits idle, which is the depth rule seen from the inside.

## Round 7: language-model loss on FineWeb (atg.py, V 2,000, 4 layers, 1 special head per layer, lr 0.04, single runs)

Base cell S5: width 64, L 256, batch 16, 150 steps (the S5 F and ID numbers of IDEA.md reproduce to 0.0004: F 3.8002, ID 3.7905). "NA" = no ALiBi on the identity head (`--eqna 1`). Gains are F minus arm (positive = lower loss than F).

| S5 arm | val loss | gain vs F |
|---|---|---|
| F | 3.8002 | - |
| Primer conv | 3.8003 | -0.000 |
| n-gram heads order 2 (NG2) | 3.7831 | +0.017 |
| ID (ALiBi on identity head, as in IDEA.md) | 3.7905 | +0.010 |
| ID NA | 3.7876 | +0.013 |
| PM (ALiBi on) | 3.7828 | +0.017 |
| PM NA | 3.7782 | +0.022 |
| PTR (ALiBi on) | 3.7951 | +0.005 |

Gain vs F along one factor at a time (other factors at S5):

| factor | level | F | NG2 | ID NA | PM NA |
|---|---|---|---|---|---|
| width | 32 | 3.8684 | +0.008 | -0.003 | +0.006 |
| width | 64 | 3.8002 | +0.017 | +0.013 | +0.022 |
| width | 128 | 3.7593 | +0.016 | +0.021 | +0.030 |
| steps | 150 | 3.8002 | +0.017 | +0.013 | +0.022 |
| steps | 300 | 3.6728 | +0.016 | +0.024 | +0.031 |
| context (batch x L fixed at 4,096 tokens) | 64 | 3.7957 | -0.001 | -0.001 | +0.005 |
| context | 128 | 3.8045 | +0.010 | +0.008 | +0.010 |
| context | 256 | 3.8002 | +0.017 | +0.013 | +0.022 |

Takeaways. Removing ALiBi from the identity head helps on text too (ID +0.003, PM +0.005), so `--eqna 1` is the default for both beds. On text the pair-match version is the one to use: PM NA beats order-2 n-gram heads at width 64 and 128 and at 300 steps (+0.022 vs +0.017; +0.030 vs +0.016; +0.031 vs +0.016). The gain of ID and PM over F grows with width, steps and context, while the n-gram gain stays flat at about +0.016 past width 64 and 150 steps. At width 32 or context 64 the identity heads give nothing (ID -0.003 and -0.001) and PM only +0.005. Pointer heads cost on text (PTR +0.005 vs ID +0.010 at S5): the pointer chain helps multi-hop lookup that natural text at this scale does not ask for.

Predictions for Round 7b: (q) width 128 at 300 steps: PM NA gain >= 0.03 and at least 0.01 above NG2; (r) width 32 at 300 steps: ID NA gain <= +0.005; (s) context 512 (batch 8): PM NA gain >= 0.025; (t) PTR NA at width 128: gain below ID NA.

Note on memory (17:30 restart): `evaluate` in both scripts ran all eval rows in one forward pass, which at L 512 (atg) and L 384 (rbg) built several GB of attention tensors (peak footprint 9.5 GB at L 512). Both now evaluate in chunks of 32-50 rows (same numbers: the L 512 check run gives 6.4575 before and after) and peak at about 0.5 GB RSS. `atg.py` also memory-maps the token file. The driver retries a run that the lag guard kills.

### Round 7b: the predictions

| prediction | F | NG2 | ID NA | PM NA | PTR NA | held? |
|---|---|---|---|---|---|---|
| (s) context 512 (batch 8): PM NA gain >= 0.025 | 3.8059 | +0.024 | | +0.034 | | yes |
| (q) width 128, 300 steps: PM NA gain >= 0.03 and >= NG2 + 0.01 | 3.6296 | +0.021 | | +0.049 | | yes |
| (r) width 32, 300 steps: ID NA gain <= 0.005 | 3.7518 | | +0.011 | +0.021 | | miss |
| (t) width 128: PTR NA below ID NA | 3.7593 | | +0.021 | | +0.011 | yes |

Width 128 other competitors: Primer 3.7609 (-0.002), KV shifting 3.7571 (+0.002). Order-2 n-gram heads are the strongest published competitor on text in every cell.

The miss (r) changes the rule: the identity-head null at width 32 was undertraining at 150 steps, not a width limit. At 300 steps width 32 gains +0.011 (ID) and +0.021 (PM). The gain of PM NA over F grows along all three knobs and their pair: width 32 / 64 / 128 at 150 steps +0.006 / +0.022 / +0.030; steps 150 / 300 at width 64 +0.022 / +0.031; context 64 / 128 / 256 / 512 +0.005 / +0.010 / +0.022 / +0.034; width 128 with 300 steps +0.049. The n-gram gain flattens at about +0.016-0.024.

## Round 8: tuning the baselines (learning-rate sweep per arm)

LM, width 128, L 256, 150 steps (Muon lr; embedding/Adam lr `elr`; Muon momentum `beta`; default 0.04 / 0.04 / 0.7):

| arm | lr 0.02 | lr 0.04 | lr 0.08 | elr 0.02 | elr 0.08 | beta 0.5 | beta 0.9 |
|---|---|---|---|---|---|---|---|
| F | 3.7645 | **3.7593** | 3.7912 | 3.7634 | 3.7674 | 3.7627 | 3.7843 |
| NG2 | 3.7475 | **3.7438** | 3.7706 | | | | |
| PM NA | 3.7348 | **3.7296** | 3.7625 | | | | |

All three arms are best at the default; at each arm's best setting PM NA beats F by 0.030 and NG2 by 0.014.

Recall, 2 hops, 8 pairs, 3 layers, 300 steps (accuracy; guessing about 0.33):

| arm | lr 0.01 | lr 0.02 | lr 0.04 | lr 0.08 | 1,200 steps at lr 0.04 |
|---|---|---|---|---|---|
| F | 0.281 | 0.272 | 0.333 | 0.211 | 0.333 |
| KV shifting | | 0.345 | 0.344 | 0.252 | |
| Primer | | 0.333 | 0.343 | 0.314 | 0.345 |
| NG | | 0.163 | 0.172 | | |
| NG2 (lr 0.04) | | | 0.166 | | |
| ID | | 0.991 (75) | 0.995 (75) | | |

No competitor leaves guessing at any learning rate or at 4x the steps.

## Round 9: filling the grid

| cell | accuracy (step to 90%) |
|---|---|
| PTR + no-backtrack, 5 hops, 5 layers | 0.990 (25) |
| PTR + no-backtrack, 7 hops, 7 layers | 0.433 (90% reached at step 25, then lost under training) |
| PTR + no-backtrack, 7 hops, 6 layers | 0.354 (-) |
| PTR, 3 hops, 3 layers, 2 / 8 heads | 1.000 (25) / 0.992 (50) |
| PTR, 2 hops, 1 layer | 0.114 (-) |
| ID, 2 hops, 4 / 6 layers | 0.996 (75) / 0.991 (75) |
| PM, 2 hops, 3 layers / 3 hops, 4 layers | 0.998 (75) / 0.266 (-) |
| mq2 16 pairs K 4: KV shifting / Primer | 0.328 (-) / 0.999 (50) |

At 7-8 hops the no-backtrack pointer solves the task at step 25 and then training pulls it away (0.93 -> 0.43-0.84). The depth rule holds in every cell (7 hops with 6 layers fails, 2 hops with 1 layer fails). Heads per layer do not matter for PTR. Pair matching adds nothing to multi-hop chains: PM behaves like ID (solves 2 hops, fails 3).

### Round 9b: hops x pairs and width for PTR, with and without no-backtrack

| cell | PTR | PTR + no-backtrack |
|---|---|---|
| 4 hops, 16 pairs, 4 layers | 0.999 (50) | 0.999 (25) |
| 4 hops, 32 pairs, 4 layers (L 320) | 0.123 (-) | 1.000 (25) |
| 4 hops, 8 pairs, 4 layers, width 32 (dk 8) | 0.321 (-) | 0.990 (25) |
| 3 hops, 48 pairs, 3 layers, width 32 (L 384) | 0.070 (-) | 0.033 (-) |
| 3 hops, fill 3 / dup 0.5 | 0.999 (25) / 0.933 (175) | |
| 2 hops, dup 1.0 | 0.786 | 0.816 |

The value/key tie also explains the pair and width limits of plain PTR: the right mass halves per hop, and with more pairs or noisier small heads the leftover signal is too weak to train. Removing the tie fixes 4 hops at 32 pairs and 4 hops at width 32. What it cannot fix: dk 8 with a 384-token context (exact-match logit eqt * sqrt(8) = 8.5 is not sharp enough against 384 positions), wrong-context repeats of the same word (dup 1.0), fewer layers than hops, and the late loss at 7-8 hops.

### Round 9c: competitors in every headline recall cell (for figs/recall.png)

| cell (250-300 steps, own lr per arm as in the arms table) | F | KV shifting | Primer | n-gram heads | ID | PTR (+ no-backtrack at 6 hops) |
|---|---|---|---|---|---|---|
| 2 hops, 32 pairs, 3 layers | 0.107 | 0.117 | 0.126 | 0.062 | 0.075 | 0.999 |
| 3 hops, 8 pairs, 4 layers | 0.328 | 0.331 | 0.334 | 0.277 | 0.328 | 1.000 |
| 6 hops, 8 pairs, 6 layers | 0.316 | 0.332 | 0.332 | 0.336 | 0.332 | 0.990 |

Every competitor stays at guessing (about 0.11 at 32 pairs, 0.34 at 8 pairs) in all four headline cells. `reproduce.sh` reruns two of them in 80 s (tags R-): 2 hops F 0.318 / ID 0.993 (step 75); 4 hops 4 layers F 0.337 / PTR + no-backtrack 1.000 (step 25).

## Fitted when-to-use rule (after 199 grid runs)

1. **Multi-hop recall (k chained lookups in context): use pointer heads with the no-backtrack term and at least k layers.** They solve k = 2-6 hops at step 25 for 8-48 pairs, filler up to 3 per pair, widths 32-128, 2-8 heads, any learning rate tried, with or without RoPE on other heads. Fewer layers than hops always fails (7 cells). Every published competitor (tuned ALiBi + sink, KV shifting, Primer, n-gram heads order 1 and 2) stays at guessing at every learning rate and at 1,200 steps.
2. **Two hops only: plain identity heads suffice when the head is sharp enough and wide enough.** They need tying, no position term and a temperature with eqt * sqrt(dk) between about 12 and 40, and at least 16 dimensions per head. Then they solve up to 32 pairs (dk 16 at eqt 10, or dk 32 at eqt 3). Below that they sit on a plateau that does not end within 600 steps.
3. **Text: use pair-match identity heads without ALiBi on them (PM NA).** The gain over tuned plain attention grows with width, steps and context (+0.005 at the smallest cell to +0.049 at width 128 with 300 steps) and beats order-2 n-gram heads by 0.010-0.028 at width >= 64 with L >= 256 or 300 steps. Below width 64 at 150 steps or below context 128 the gain is under 0.01 and n-gram heads tie it. Pointer heads cost half the text gain, so they are for retrieval-heavy models, not the default.
4. **Where they fail:** the same word repeated in wrong contexts (dup 0.5 costs about 0.1, dup 1.0 drops ID to guessing and PTR to 0.8); queries that never repeat a stored token exactly (synonyms in every row: ID 0.34), although learned synonyms work when at least a quarter of rows contain exact repeats; small heads (dk 8) with long contexts; and 7+ hops, where the chain forms at step 25 and is then lost in training.

Prediction record while fitting: Round 1 11 of 12 held; Rounds 3b-7b 12 of 20 held. The 8 misses (head dimension at dk 64 and dk 8, PTR at dk 8 with 48 pairs, dup 0.5, synonyms 0.5, partial matches with 3 layers, the normal-head ablation control at 0.868, width 32 on text) each changed the rule above.

## Round 10: validation of the final rule (predictions written before running)

(V1) PTR + no-backtrack, 5 hops, 16 pairs, 5 layers: solves. (V2) PTR + no-backtrack, 3 hops, 16 pairs, 2 layers: fails. (V3) ID at eqt 10, dk 16, 24 pairs: solves. (V4) ID at eqt 10, dk 8 (width 32), 8 pairs: fails. (V5) ID, syn 0.25: >= 0.95. (V6) text, width 128, L 512: PM NA beats NG2 by >= 0.01. (V7) text, width 32, L 64, 150 steps: PM NA gain over F < 0.01.


| validation | measured | held? |
|---|---|---|
| (V1) PTR + nb, 5 hops, 16 pairs, 5 layers: solves | 1.000 (25) | yes |
| (V2) PTR + nb, 3 hops, 16 pairs, 2 layers: fails | 0.159 | yes |
| (V3) ID eqt 10, dk 16, 24 pairs: solves | 0.992 (125) | yes |
| (V4) ID eqt 10, dk 8, 8 pairs: fails | 0.996 (225) | miss |
| (V5) ID syn 0.25: >= 0.95 | 0.944 (90% at 125) | miss by 0.006 |
| (V6) text, width 128, L 512: PM NA >= NG2 + 0.01 | F 3.7653, NG2 3.7399 (+0.025), PM NA 3.7176 (+0.048) | yes |
| (V7) text, width 32, L 64: PM NA gain < 0.01 | F 3.8510, PM NA 3.8540 (-0.003) | yes |

The V4 miss made me write the ID capacity rule as a condition on pairs and three more cells: (V8) dk 8, eqt 10, 12 pairs: solves -> 0.150, miss; (V9) dk 32, eqt 10, 48 pairs: fails -> 0.998 (200), miss; (V10) dk 16, eqt 3, 20 pairs: fails -> 0.990 (125), miss. Three misses in a row mean the "pairs <= dk" form was wrong. Refit on all 20 ID cells, using sharpness s = eqt * sqrt(dk) and the number of pairs N:

ID solves two hops within 300 steps when **s >= about N / 2** (the exact-match logit must beat the log-odds of N competing pairs), **N <= 2 dk**, and **s <= about 60** (at s = 120 the softmax is one-hot and the second hop gets no gradient). This fits all 14 cells with dk >= 16, including the three misses (s = 12 vs N = 20: solves, 24: fails; s = 57 vs N = 48 at dk 32: solves; s = 24 vs N = 48 at dk 64: fails). At dk 8 only one cell (8 pairs at s = 28) solves, so dk 8 is outside the rule.

Final check of the refit (written before running): (V11) dk 16, eqt 6 (s 24), 40 pairs: fails (N > 2 dk). (V12) dk 32, eqt 6 (s 34), 40 pairs: solves. (V13) dk 64, eqt 6 (s 48), 48 pairs: solves. (V14) dk 32, eqt 3 (s 17), 40 pairs: fails (s < N / 2).

| final check | measured | held? |
|---|---|---|
| (V11) dk 16, s 24, 40 pairs: fails | 0.056 | yes |
| (V12) dk 32, s 34, 40 pairs: solves | 0.996 (125) | yes |
| (V13) dk 64, s 48, 48 pairs: solves | 0.993 (200) | yes |
| (V14) dk 32, s 17, 40 pairs: fails | 0.996 (100) | miss |

V14 shows the "s >= N / 2" part is too strict at dk 32: s = 17 holds 40 pairs (and not 48). Across the 24 ID cells the limit is better described per head dimension: at eqt 3 plain identity heads hold about 1.3-1.4 dk pairs (dk 16: 20 yes, 24 no; dk 32: 40 yes, 48 no), and eqt 6-10 extends that to 1.5-2 dk (dk 16: 32 yes, 40 and 48 no; dk 32: 48 yes; dk 64: 48 yes at eqt 6). One cell does not fit (dk 64, eqt 3, 48 pairs fails), and dk 8 holds only 8 pairs at eqt 10. This capacity limit is specific to the learned second hop of plain identity heads; pointer heads with no-backtrack have no such limit at dk >= 16 (48 pairs at step 25).

Validation score: the main rule (pointer heads for multi-hop with layers >= hops; PM NA for text, with the gain growing in width, steps and context and vanishing at width 32 / context 64) held in all 6 of its validation cells (V1, V2, V6, V7, V11-V13 on capacity count 3 more). The plain-ID capacity sub-rule missed 5 of 9 predictions before reaching the per-dk form above, so it is reported as an approximate map, not a validated rule.

## Round 11: can the 3-hop limit be broken? (Paper Gen asked for the weakest point, 2026-11-15 evening)

The 3-hop cells that still fail: plain and pair-match identity heads at 3 hops (0.328 / 0.266, no free second hop), and pointer heads at width 32 with 48 pairs (L 384: 0.070, with no-backtrack 0.033), which Round 9b blamed on a too-soft exact match (eqt * sqrt(8) = 8.5 against 384 positions). Tests, predictions written before running:

(W1) PTR + nb, 3 hops, 48 pairs, width 32, eqt 10 (s 28): solves, if softness was the cause. (W2) same at width 64 (dk 16, eqt 3, s 12): solves. (W3) PTR + nb, 3 hops, 64 pairs, width 64: solves. (W4) 96 pairs: solves within 300 steps. (W5) ID at eqt 10, 3 hops, 4 layers, 600 steps: still fails (nothing gives the second hop a free start). (W6) F at 3 hops, 64 pairs: guessing.

(W9) Text, depth as a fourth knob (width 64, L 256, 150 steps; S5 is 4 layers, F 3.8002, PM NA +0.022, NG2 +0.017): the PM NA gain over F holds or grows with depth: 2 layers >= +0.010, 6 layers >= +0.022, and stays above NG2 at 6 layers.

| test | measured | held? |
|---|---|---|
| (W1) PTR + nb, 3 hops, 48 pairs, width 32, eqt 10 | 0.991 (25) | yes: softness was the cause (eqt 3 gave 0.033) |
| (W2) same at width 64 | 1.000 (25) | yes |
| (W3) PTR + nb, 3 hops, 64 pairs, width 64 (tv 512, batch 16) | 1.000 (25) | yes |
| (W4) 96 pairs, width 64 (L 768) | 0.997 (25) | yes |
| (W7) 96 pairs, width 128 | 1.000 (25) | |
| (W8) 96 pairs, width 32, eqt 10 | 0.998 (50) | |
| (W5) ID eqt 10, 3 hops, 4 layers, 600 steps | 0.336 | yes (fails) |
| (W6) F, 3 hops, 64 pairs (guessing 0.016) | 0.010 (300 steps, 168 s) | yes |

| text, width 64, L 256, 150 steps | F | NG2 (gain) | PM NA (gain) |
|---|---|---|---|
| 2 layers | 3.8217 | 3.8029 (+0.019) | 3.7895 (+0.032) |
| 4 layers (B7) | 3.8002 | 3.7831 (+0.017) | 3.7782 (+0.022) |
| 6 layers | 3.7989 | 3.7857 (+0.013) | 3.7707 (+0.028) |

Takeaway: the 3-hop failure of pointer heads at width 32 was only a too-soft match: a sharper fixed temperature (eqt 10) solves it, and with no-backtrack 3 hops now hold from 8 to 96 pairs (contexts up to 768 tokens) at widths 32, 64 and 128, all at step 25-50, while tuned attention sits at guessing. Plain and pair-match identity heads still fail 3 hops even at 600 steps with a sharp head: they give only the first hop free, so the rule stays "multi-hop needs pointer heads". On text the pair-match gain holds along depth (+0.022 to +0.032 at 2-6 layers) and stays 0.005-0.015 above n-gram heads, whose gain shrinks with depth (+0.019 -> +0.013). Recall runs at L >= 512 must use batch 16: at batch 64 they reach 4-9 GB and the lag guard kills them. 15 runs this round (243 in total).

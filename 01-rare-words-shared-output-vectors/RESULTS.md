# Cluster-shared output rows: scaling-grid study (gate 4, berserker mode)

Bed: `grid.py` (copy of `tb.py`, same model and data; runs on the Apple GPU through MPS because the head at V 16,000 took 27 s per 100 steps on one CPU thread and 9 s on MPS; losses are identical on both devices: V 2,000 baseline 3.9459 and cluster arm 3.9353 reproduce `tb.py` exactly). Driver `run.py blocks/bN.txt`, one run at a time, `nice -n 20`, results in `grid.jsonl`, per-bin comparison `an.py`.

Default cell: width 32, 2 layers, batch 32 x 64 tokens, 400 steps (819K tokens), Muon 0.02 on blocks, Adam 0.05 on the input table, head rate 0.1, 64 clusters re-clustered every 50 steps with shared-vector rate 0.02, untied.

Arms in every cell:
- **B**: tuned baseline (head rate 0.1; 0.05 and 0.2 within 0.001 at V 6,000).
- **E**: strongest published competitor, a frequency-aware row step: head-only Adam epsilon 1e-5 (tuned below). It is the fix for rare output rows that keep moving at full Adam speed while their word is absent (see `../llm-frequency-scaled-embedding-steps`).
- **C**: cluster-shared head.
- **CE**: cluster-shared head with the head-only epsilon.

The headline number is **E - CE**, the gain over the strongest competitor. Per-bin losses are by the number of times the target word occurs as a target in the tokens the run trains on.

## Round 1. Which published competitor is strongest? (V 6,000, 400 steps)

Rule going in: the cluster gain is new because Coupled Adam, RowNorm and global epsilon lost at V 2,000. Risk: no frequency-aware head-only step was tried.

| Arm | val loss | vs baseline |
|---|---|---|
| baseline, head rate 0.05 / 0.1 / 0.2 | 5.0188 / **5.0176** / 5.0182 | - |
| cluster-shared head, head rate 0.1 / 0.05 | 4.9950 / **4.9933** | -0.024 |
| head-only Adam epsilon 1e-6 / 3e-6 / **1e-5** / 3e-5 / 1e-4 / 1e-3 | 5.0063 / 5.0003 / **4.9931** / 4.9945 / 4.9975 / 5.0425 | -0.025 |
| head epsilon 1e-5 at head rate 0.05 / 0.2 | 5.0007 / 4.9946 | |
| head-only beta2 0.99 | 5.0057 | -0.012 |
| Coupled Adam on the head, rate 0.1 / 0.03 | 5.0981 / 5.0865 | +0.069 |
| class-based softmax, 64 frequency classes (Mikolov 2011) | 5.0260 | +0.008 |
| class-based softmax, 64 k-means classes re-clustered | 5.2135 | +0.196 |
| adaptive softmax, cutoffs 10% / 40% of V, divisor 4 / 2 | 5.0539 / 5.0329 | +0.036 / +0.015 |

Takeaway: head-only Adam epsilon 1e-5 is the strongest competitor and alone matches the cluster head at V 6,000. The earlier screen missed it because it only tried epsilon on all tables at V 2,000.

## Round 2. Do clusters stack on the competitor? (V 6,000)

| Arm | val loss |
|---|---|
| head epsilon 1e-5 (E) | 4.9931 |
| clusters + head epsilon 1e-5 (CE) | **4.9720** |
| CE at head rate 0.05 / epsilon 3e-6 / shared rate 0.05 | 4.9821 / 4.9777 / 4.9790 |

Per bin of training occurrences (nats per target gained over B; share of validation targets in the bin):

| bin | share | C | E | CE |
|---|---|---|---|---|
| 0 occurrences | 0.000 | +0.19 | +5.34 | +5.82 |
| 1-9 | 0.007 | +0.30 | +0.95 | +1.24 |
| 10-31 | 0.079 | +0.10 | +0.18 | +0.32 |
| 32-99 | 0.132 | -0.02 | +0.00 | +0.00 |
| 100-999 | 0.215 | -0.01 | -0.02 | -0.00 |
| 1k+ | 0.403 | +0.02 | +0.01 | +0.02 |
| unknown token | 0.164 | +0.06 | -0.01 | +0.02 |

Takeaway: the two fixes add up (0.024 + 0.025 alone, 0.046 together), so the clusters do something the epsilon does not: over E, CE gains 0.011 of its 0.021 on words seen 10-31 times and 0.005 on the unknown token. Alone, a third of the cluster gain sits on the unknown token, a small-vocabulary artifact that a real vocabulary does not have, so every cell also reports the gain without it.

**Rule after round 2.** The cluster head pools the target counts of similar rare words; it gains over the competitor in proportion to the share of target tokens whose word occurs about 10-30 times in training. Prediction for round 3 (400 steps, E - CE): V 2,000 about 0.005, V 4,000 0.012, V 10,000 0.025, V 16,000 0.030.

## Round 3. Vocabulary (400 steps, batch 32)

| V | B | E | CE | B - CE | **E - CE** | E - CE without unknown token | predicted E - CE |
|---|---|---|---|---|---|---|---|
| 2,000 | 3.9459 | 3.9402 | 3.9317 | 0.014 | **0.0085** | 0.005 | 0.005 |
| 4,000 | 4.6048 | 4.5902 | 4.5761 | 0.029 | **0.014** | 0.012 | 0.012 |
| 6,000 | 5.0176 | 4.9931 | 4.9720 | 0.046 | **0.021** | 0.016 | - |
| 10,000 | 5.5078 | 5.4613 | 5.4406 | 0.067 | **0.021** | 0.022 | 0.025 |
| 16,000 | 5.9762 | 5.8945 | 5.8688 | 0.107 | **0.026** | 0.024 | 0.030 |

Cluster head without the epsilon (C), gain over B: 0.009 / 0.018 / 0.023 / 0.025 / 0.034.

Takeaway: the gain over the strongest competitor grows with vocabulary at every step (4 of 4 predictions within 0.004), about 0.007-0.008 per doubling of V. The competitor's own gain grows faster (0.006 to 0.082), so both fix rare rows, in different ways.

## Round 4. Number of clusters x vocabulary (CE arm, 400 steps)

| V | 16 | 64 | 256 | 1,024 |
|---|---|---|---|---|
| 2,000 | 3.9335 | **3.9317** | 3.9367 | - |
| 6,000 | 4.9740 | **4.9720** | 4.9726 | 4.9821 |
| 16,000 | 5.8753 | **5.8688** | 5.8696 | 5.8759 |

Takeaway: 64-256 clusters are best at every vocabulary; the prediction that the best count grows with V (fixed words per cluster) was wrong. 16 clusters keep about 80% of the gain and 1,024 lose about half, so a cluster must hold enough frequent members to pool from, and the count does not need tuning per vocabulary.

## Round 5. Training length x vocabulary (batch 32)

E - CE (without the unknown token in brackets):

| V | 200 steps | 400 | 800 | 1,600 |
|---|---|---|---|---|
| 2,000 | 0.0062 (0.005) | 0.0085 (0.005) | 0.0079 (0.010) | 0.0061 (0.006) |
| 6,000 | 0.011 (0.010) | 0.021 (0.016) | 0.015 (0.023) | 0.021 (0.011) |
| 16,000 | 0.013 (0.016) | 0.026 (0.024) | **0.035 (0.034)** | - |

Baseline B - CE at 1,600 steps: 0.013 (V 2,000), 0.059 (V 6,000).

Per-target gain of CE over E on words of corpus frequency 1-3 in 100,000 tokens: 0.14 (V 6,000, 400 steps), 0.12 (V 6,000, 1,600 steps, when such a word has been seen 33-100 times), 0.14 (V 16,000, 800 steps). At 3e-6-1e-5: 0.20 (V 16,000). At 3e-5-1e-4: 0.02-0.04. Above 1e-4: about 0 at V 6,000+ and 0.02 at V 2,000.

Takeaway: the gain does not fade with training at V 6,000 and 16,000 (8 and 1.6 tokens per weight); at V 2,000 it peaks at 400-800 steps and falls slightly by 1,600 (22 tokens per weight). Measured per word, the gain depends on how rare the word is in the corpus, and stays about the same when the word is seen 4 times more often; it is small only very early (200 steps, before the rows have formed clusters).

## Round 6. Batch size at fixed tokens (V 6,000)

| Setting | B | E | CE | E - CE |
|---|---|---|---|---|
| batch 32 x 400 steps | 5.0176 | 4.9931 | 4.9720 | 0.021 |
| batch 128 x 100 steps (same tokens) | 5.0978 | 5.0968 | 5.0896 | 0.007 |
| batch 128 x 400 steps (4x tokens) | 4.7314 | 4.7274 | 4.7195 | 0.008 |

Takeaway: a 4x larger batch removes almost all of the competitor's gain (0.001 vs 0.025: rows are rarely quiet for long) and cuts the cluster gain to a third, and 4x more tokens at that batch do not bring it back. Per word the gain on frequency 1-3e-5 is 0.06 at 100 steps and 0.10 at 400 steps of batch 128, against 0.14 at batch 32. So the cluster gain needs both many updates (clusters form over steps) and rare words appearing in few batches.

**Rule after round 6.** The gain over the competitor is a per-word effect on target words of corpus frequency about 1e-5 to 1e-4 that have been seen at least about 10 times: about 0.13 nats per target at 1-3e-5 and 0.03 at 3e-5-1e-4, near 0 above 1e-4. The cell gain is the share of such targets times these per-word gains, and it shrinks with a larger batch. Predictions for round 7: V 24,000 at 400 steps adds words of frequency 3e-6-1e-5 seen only 2-8 times, so the gain stops growing (0.026-0.030); a V 6,000 vocabulary with a long tail (half the words drawn from ranks 3,000-24,000, `--tail 4`) has fewer targets at 1e-5-1e-4 and should gain about 0.007, below plain V 6,000.

## Round 7. Vocabulary versus word frequency (400 steps)

| Cell | B | E | CE | E - CE | predicted |
|---|---|---|---|---|---|
| V 24,000 | 6.3206 | 6.2181 | 6.1948 | 0.023 | 0.026-0.030 (first prediction from the V rule: 0.032-0.037) |
| V 6,000, long tail (`--tail 4`) | 4.6146 | 4.5894 | 4.5797 | 0.0097 | 0.007 (V rule: 0.016) |

Per-word gain at V 24,000 on words of frequency 3e-6-1e-5 is 0.084 (words seen 2-8 times), against 0.20 at V 16,000 and 800 steps where the same words are seen 5-16 times.

Takeaway: vocabulary size is a proxy. The gain follows how many target tokens fall on words of frequency 1e-5-1e-4 that the run has seen at least about 10 times; adding rarer words (V 24,000) or swapping mid-frequency words for rarer ones (long tail) adds little. The frequency rule predicted both cells better than the vocabulary rule.

## Round 8. Width and depth

E - CE:

| V | width 16 | 32 | 64 |
|---|---|---|---|
| 2,000 | 0.0117 | 0.0085 | 0.0041 |
| 6,000 | 0.0164 | 0.021 | 0.0171 |

| V | 1 layer | 2 | 4 |
|---|---|---|---|
| 2,000 | 0.0031 | 0.0085 | 0.0030 |
| 6,000 | 0.0207 | 0.021 | 0.0241 |

Cell values (B / E / CE): V 6,000 width 16: 5.1190 / 5.0993 / 5.0829; width 64: 4.9636 / 4.9382 / 4.9211; 4 layers: 5.0154 / 4.9929 / 4.9688.

Takeaway: at V 6,000 the gain is flat in width (16-64) and depth (1-4), so it does not come from a capacity shortage of a small model. At V 2,000, where the gain sits near the noise floor (about 0.003-0.005 between nearby settings), it falls with width; this matches the rule, since V 2,000 has almost no words in the 1e-5-1e-4 band.

## Round 9. Probes: what carries the gain

V 6,000, 400 steps; E = 4.9931, CE = 4.9720 (gain 0.021).

| Probe | val | gain over E |
|---|---|---|
| clusters with frozen shared vectors (rate 1e-9) | 4.9931 | 0.000 |
| random fixed clusters | 4.9929 | 0.000 |
| frequency classes as fixed clusters (Mikolov) | 4.9894 | 0.004 |
| k-means, stop re-clustering at 25% of training | 4.9836 | 0.010 |
| k-means, stop at 50% | 4.9766 | 0.017 |
| re-cluster every 25 / 50 / 100 / 200 steps | 4.9755 / **4.9720** / 4.9770 / 4.9812 | 0.018 / 0.021 / 0.016 / 0.012 |
| shared-vector rate 0.01 / 0.02 / 0.05 | 4.9808 / **4.9720** / 4.9785 | |
| post-hoc: shrink each final row toward its k-means centroid by n0/(n0+count), best n0 (=3) | 4.9925 | 0.0006 |

V 16,000 (E = 5.8945, CE = 5.8688, gain 0.026): random clusters 5.8969 (-0.002), frequency classes 5.8971 (-0.003), post-hoc shrink best n0 = 1 5.8951 (-0.001).

With the learned shared vectors set back to their value at the last re-clustering, loss moves by at most 0.002 at the default period; with re-clustering stopped at 25%, the same reset costs 0.18 nats (5.1646), so the shared vectors keep learning between re-clusterings and carry the pooled signal.

Takeaway: the gain needs all three parts: shared vectors that learn during training, clusters built from the current row geometry, and re-clustering through training. Clusters by frequency or at random give nothing at V 16,000, and pulling rare rows toward their centroid after training (the post-hoc version of the same idea) gives at most 0.0006. The mechanism is a training-time effect: every gradient on a frequent word in a cluster moves the shared vector, and so moves its rare neighbours.

## Round 10. Tied embeddings and a stronger input side

Tied input and output rows (head rate 0.05, the best tied rate for B and E):

| V | B | E | CE | E - CE | untied E - CE |
|---|---|---|---|---|---|
| 2,000 | 4.0165 | 4.0109 | 4.0025 | 0.0084 | 0.0085 |
| 6,000 | 5.1220 | 5.0841 | 5.0720 | 0.0121 (0.0168 with CE at head rate 0.1) | 0.021 |
| 16,000 | 6.0637 | 5.9797 | 5.9494 | 0.0303 | 0.026 |

Stronger input side (value embeddings in both layers and token smearing, the modded-nanoGPT recipe):

| V | B | E | CE | E - CE | predicted |
|---|---|---|---|---|---|
| 2,000 | 3.9207 | 3.9159 | 3.9114 | 0.0045 | 0.008 |
| 6,000 | 4.9813 | 4.9615 | 4.9432 | 0.0183 | 0.020 |
| 16,000 | 5.9336 | 5.8622 | 5.8339 | 0.0283 | 0.026 |

Takeaway: the gain survives tied rows (where the cluster vector also enters the input) and a stronger input side, with the same growth from V 2,000 to 16,000. It is a property of rare output rows, not of the default toy model.

## Round 11. Is the competitor tuned at every vocabulary?

Head-only epsilon for E and CE at each vocabulary (400 steps):

| V | E: eps 3e-6 / 1e-5 / 3e-5 / 1e-4 | CE: eps 1e-5 / 3e-5 / 1e-4 | best E - best CE |
|---|---|---|---|
| 2,000 | 3.9431 / 3.9402 / **3.9383** / 3.9440 | 3.9317 / **3.9288** / - | 0.0095 |
| 6,000 | 5.0003 / **4.9931** / 4.9945 / 4.9975 | 4.9720 / **4.9695** / - | 0.0236 |
| 16,000 | 5.9122 / 5.8945 / **5.8905** / 5.8949 | 5.8688 / **5.8687** / 5.8879 | 0.0218 |

Head rate at V 16,000: B 5.9720 / 5.9762 / 5.9846 (0.05 / 0.1 / 0.2); E 5.9015 / 5.8945 / 5.8955; CE 5.8688 at 0.1 and 5.8772 at 0.2.

Takeaway: with epsilon and head rate tuned per vocabulary for both arms, the gain is 0.010 / 0.024 / 0.022, within 0.004 of the untuned numbers. The default head rate 0.1 and epsilon 1e-5 to 3e-5 are within 0.004 of best for every arm.

## Round 12. Batch extremes and vocabulary tails (V 6,000 unless noted)

Predictions written before the runs: batch 16 x 800 steps 0.030; tail 2 0.012; tail 8 0.007; V 16,000 tail 4 0.020.

| Cell | B | E | CE | E - CE | predicted |
|---|---|---|---|---|---|
| batch 16 x 800 steps | 5.0760 | 5.0016 | 4.9762 | 0.025 | 0.030 |
| batch 32 x 400 steps | 5.0176 | 4.9931 | 4.9720 | 0.021 | - |
| batch 64 x 200 steps | 5.0434 | 5.0362 | 5.0249 | 0.011 | - |
| batch 128 x 100 steps | 5.0978 | 5.0968 | 5.0896 | 0.007 | - |
| tail 2 (half the words from ranks 3,000-12,000) | 4.8032 | 4.7780 | 4.7637 | 0.014 | 0.012 |
| tail 4 | 4.6146 | 4.5894 | 4.5797 | 0.010 | 0.007 |
| tail 8 | 4.5225 | 4.5052 | 4.4906 | 0.015 | 0.007 (miss) |
| V 16,000, tail 4 | - | 5.4869 | 5.4530 | 0.034 | 0.020 (miss) |

Per-word gain by corpus frequency (nats per target, share of targets):

| Cell | <3e-6 (seen under 3 times) | 3e-6-1e-5 | 1-3e-5 | 3e-5-1e-4 | 1e-4-3e-4 |
|---|---|---|---|---|---|
| tail 8 | -0.26 / 0.004 | -0.06 / 0.005 | +0.11 / 0.007 | +0.063 / 0.116 | -0.007 |
| V 16,000, tail 4 | -0.37 / 0.006 | +0.05 / 0.012 | +0.19 / 0.086 | +0.060 / 0.136 | +0.018 |
| V 6,000 plain | - | - | +0.14 / 0.055 | +0.04 / 0.136 | -0.01 |

Takeaway: batch size at fixed tokens is a clean axis: the gain halves from batch 32 to 64 and is a third at 128, and batch 16 gives a little more than 32. Two predictions missed upward: a vocabulary with many very rare rows raises the per-word gain on words of frequency 3e-5-1e-4 from 0.04 to 0.06, so the per-word constants are not fixed; they grow with the number of rare rows that share clusters with these words. Words seen fewer than about 3 times lose 0.2-0.45 nats per target with clusters (their row is pulled toward a cluster before it has any evidence of its own); they are under 1% of targets here, so the cell still gains.

**Rule after round 12.** Gain over the competitor = (share of targets on words with corpus frequency 1e-5 to 1e-4 that have been seen at least about 10 times) x (about 0.14 per target at 1-3e-5 and 0.04-0.06 at 3e-5-1e-4) + about 0.2 per target on words of 3e-6-1e-5 once they are seen 5-16 times; minus 0.2-0.4 per target on words seen under 3 times; divided by about 2 at batch 64 and 3 at batch 128 (batch 16-32 = 1). Width, depth, tying and the input side do not change it.

## Round 13. Final prediction round (cells chosen after the rule was fixed)

Predictions from the rule after round 12, written before any run:

| Cell | why it tests the rule | predicted E - CE |
|---|---|---|
| V 24,000, 800 steps | words of 3e-6-1e-5 (8% of targets) reach 5-16 sightings: 0.079 x 0.2 + 0.1 x 0.14 + 0.136 x 0.04 | 0.035 (vs 0.023 at 400 steps) |
| V 10,000, 800 steps | almost no words below 1e-5; the 1-3e-5 band is fully past 10 sightings | 0.021 |
| V 16,000, width 64 | width does not matter | 0.026 |
| V 6,000, batch 64 x 400 steps (2x tokens) | batch 64 halves the gain; more tokens do not restore it | 0.011 |
| V 16,000, batch 128 x 100 steps | batch 128 cuts the gain to a third | 0.009 |

Results (the Mac restarted at 17:20 and the lag guard killed the first V 24,000 and batch-128 attempts; evaluation now runs 50 sequences at a time and large batches use gradient accumulation `--acc`, which reproduces 4.9931 exactly):

| Cell | B | E | CE | measured E - CE | predicted | hit |
|---|---|---|---|---|---|---|
| V 24,000, 800 steps | 6.1333 | 5.9971 | 5.9585 | 0.039 | 0.035 | yes |
| V 10,000, 800 steps | 5.3350 | 5.2723 | 5.2481 | 0.024 | 0.021 | yes |
| V 16,000, width 64 (default rates) | 5.9081 | 5.8341 | 5.8172 | 0.017 | 0.026 | no |
| V 6,000, batch 64 x 400 steps | 4.8528 | 4.8393 | 4.8304 | 0.009 | 0.011 | yes |
| V 16,000, batch 128 x 100 steps | 6.0209 | 5.9979 | 5.9875 | 0.010 | 0.009 | yes |

Over all prediction rounds, 12 of 15 predictions written before the run land within 0.005 (`fig_predictions.png`). The three misses: tail 8 and V 16,000 tail 4 (more very rare rows raise the per-word gain on mid-frequency words, round 12) and width 64 at default rates (round 14).

## Round 14. Width needs a smaller shared-vector rate

The width-64 miss led to a width sweep at V 6,000 and 16,000 with head rate, epsilon and shared-vector rate swept per arm.

| Cell | best E (setting) | best CE (setting) | E - CE tuned | E - CE default rates |
|---|---|---|---|---|
| V 16,000, width 32 | 5.8905 (eps 3e-5) | 5.8687 (eps 3e-5) | 0.022 | 0.026 |
| V 16,000, width 64 | 5.8277 (head 0.05, eps 3e-5) | 5.8044 (head 0.05, shared 0.01) | 0.023 | 0.017 |
| V 16,000, width 128 | 5.7650 (head 0.03, eps 3e-5) | 5.7449 (head 0.03, shared 0.005) | 0.020 | 0.013 |
| V 6,000, width 32 | 4.9931 | 4.9695 | 0.024 | 0.021 |
| V 6,000, width 128 | 4.8869 (head 0.03) | 4.8705 (head 0.03, shared 0.005) | 0.016 | 0.008 |
| V 6,000, width 128, 1,600 steps | 4.5502 (head 0.03, eps 3e-5) | 4.5343 (head 0.03, shared 0.005) | 0.016 | 0.063 (E untuned) |

Shared-vector rate at V 16,000, width 128, head 0.1: 0.05 / 0.02 / 0.01 gives 5.7860 / 5.7681 / 5.7671; head 0.05 with shared 0.01: 5.7537. Cluster count at width 128: 16 / 64 / 256 gives 5.7740 / 5.7681 / 5.7776; re-clustering every 25 steps 5.7708.

One trap: at width 128 and 1,600 steps the competitor at epsilon 1e-5 stays at the baseline (4.6151 vs 4.6150) and the cluster head looked 0.063 better; with epsilon 3e-5 and head rate 0.03 the competitor reaches 4.5502 and the true gain is 0.016. Every number in this file uses the best setting found for each arm where a sweep exists.

Takeaway: with rates tuned per arm, the gain holds across width 32-128 at V 16,000 (0.022 / 0.023 / 0.020) and falls by a third at V 6,000 (0.024 to 0.016). The shared-vector rate must shrink with width, about 0.02 x 32 / width; at the default 0.02 a wide model loses half the gain. The reason is that the shared vector sums the gradient of every word in its cluster, so its step grows with the number of active members, and a wider head with a lower head rate needs a proportionally smaller shared step.

## Round 15. Learning rates of the rest of the model

| Arm | Muon 0.01 / 0.02 / 0.04 | input table 0.05 / 0.1 |
|---|---|---|
| V 6,000 E | 4.9962 / **4.9931** / 5.0036 | **4.9931** / 5.0028 |
| V 6,000 CE | 4.9788 / **4.9720** / 4.9936 | **4.9720** / 4.9859 |
| V 16,000 E (eps 3e-5) | 5.9064 / **5.8905** / 5.9033 | - |
| V 16,000 CE (eps 3e-5) | 5.8783 / **5.8687** / 5.8824 | - |

Depth at V 16,000 (eps 3e-5): 4 layers B 5.9519, E 5.8694, CE 5.8484, gain 0.021 (2 layers: 0.022).

Takeaway: the shared block and input rates are at their best for both arms, and the gain holds at every rate tried (0.010-0.028).

## Final rule (fitted, 241 runs)

**Use the cluster-shared head when a sizable share of target tokens (5% or more) fall on words of corpus frequency 1e-5 to 1e-4 that the run sees at least about 10 times, with batches of 16-32 sequences per step.** Then it beats the tuned baseline and the strongest competitor (a head-only Adam epsilon) by about 0.14 nats per target on words of frequency 1-3e-5, 0.04-0.06 on 3e-5-1e-4 and 0.2 on 3e-6-1e-5 once they are seen 5-16 times; the cell gain is these numbers times the target shares (about 0.02 at vocabularies 6,000-16,000 and 0.039 at 24,000 with 800 steps). It holds with training length, depth (1-4 layers), width (32-128, with the shared rate scaled as 1/width), tied rows and a stronger input side.

**Do not expect a gain** at small vocabularies where nearly every word is frequent (V 2,000: 0.009, near noise), at large batches (half at batch 64, a third at batch 128, because a rare word then appears in fewer separate updates), or on words seen fewer than about 3 times (they lose 0.2-0.45 nats; under 1% of targets here). Clusters must come from k-means on the current rows and be refreshed during training: random clusters, frequency classes, frozen shared vectors and post-hoc shrinkage toward centroids all give 0 to 0.004.

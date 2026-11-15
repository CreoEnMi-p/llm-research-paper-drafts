# RESULTS: Muon that sets its own step size

**Current rule (after round 48, 792 runs in results.jsonl; round 38 adds six unseen settings and the fall-time rule for where it fails, round 39 starts 100x off):** one global multiplier s on Muon's learning rate, updated by log s += gain · mean over block matrices of cos(g_t, O_t−1) (the new gradient against the previous orthogonalized update), with weight decay 1 and warmup / flat / linear cooldown on top. **Safe recipe:** gain = the warmup factor (1 after warmup), adapt until the earlier of 30% of training and step 450, then hold s. It ties the swept constant (−0.008 to +0.006) from starts 10x too low to 10x too high at 4, 40 and 89 tokens per weight, at 450 steps of batch 16 (round 37; it still loses 0.007-0.014 at 225 steps of batch 32), at depth 8, and at widths 32-128 (gain 2 at width 128), with the early step peak at most 10x the tuned rate. Adapting to the cooldown (round 1-28 default) learns a decay that is 0.002-0.009 better at up to 24 tokens per weight and over-decays at 40 (round 35: +0.012-0.020). The rule is the only method in this slot that recovers from 10x too high (D-Adaptation, Prodigy and Mechanic do not; round 29). On Adam's embedding and head the same rule works from starts up to 40x low but not from 10x high (rounds 30, 32). **Rounds 40-43:** decrease-only or asymmetric versions fail (the held signal is flat and noisy, so asymmetry becomes drift); one multiplier per optimizer beats one shared multiplier (Adam started low); dropping the fixed warmup on both optimizers ties from 10x low to 10x high in 14 of 15 runs over five settings and repairs the width-64, batch-4 high-start break; dropping only Muon's warmup fails. **Rounds 45-48:** after the hold, a decrease-only correction ties only when its threshold is about 2 standard deviations of the cosine's own noise (`--psd 2`; 1 drifts, 3 is inert); the warmup-free recipe transfers with no retuning to widths 24, 48, 96 and depths 3, 5 (15 of 15 runs within 0.006); it beats Mechanic by 0.010-0.032 mean loss in 9 of 12 settings and ties in the 3 runs of 3,000-4,000 steps; one recipe run costs about 1.1 constant runs against 5-7 for a sweep.

Lab notebook. Every round: the rule being tested, the table, a one-line takeaway. Loss is cross-entropy on 400 held-out FineWeb windows of 64 tokens (`val`), lower is better.

Test bed `ss.py`: width 64, 4 layers, 4 heads, vocabulary 2,000 (rare tokens -> one unknown token), context 64, batch 16 windows, 200 steps, RMSNorm, RoPE, QK-norm, SwiGLU, no biases, untied head. Muon (momentum 0.95, Nesterov, 5 Newton-Schulz steps, shape factor sqrt(max(1, rows/cols))) on the 28 block matrices, Adam on embedding, head and norms. Schedule: 5% warmup, flat, 30% linear cooldown. Each run 6-15 s on one CPU thread under the shared lock.

## Round 1: test bed and baseline rates (consecutive text, later corrected)

Muon rate sweep at Adam rate 0.02, then Adam rate sweep at Muon rate 0.005.

| Muon lr | 0.0006 | 0.00125 | 0.0025 | 0.005 | 0.01 | 0.02 | 0.04 | 0.08 | 0.16 | 0.32 |
|---|---|---|---|---|---|---|---|---|---|---|
| val | 4.168 | 4.154 | 4.144 | **4.133** | 4.149 | 4.185 | 4.213 | 4.228 | 4.237 | 4.257 |

| Adam lr (Muon 0.005) | 0.005 | 0.01 | 0.02 | 0.04 | 0.08 | 0.16 |
|---|---|---|---|---|---|---|
| val | 4.235 | 4.180 | 4.133 | **4.119** | 4.116 | 4.140 |

Takeaway: Adam rate 0.04 is used from here on (0.04 and 0.08 tie). First naive rule (orthogonalized-update cosine, target 0) ran the step size to the upper bound in every matrix: consecutive Muon updates share 95% of their momentum, so their cosine is 0.8 at every rate.

## Round 2-3: which signal tracks the rate (consecutive text, later corrected)

Six cosines logged per matrix over the flat phase at each rate: `upd` = cos(O_t, O_t-1) of orthogonalized updates; `grad` = cos(g_t, g_t-1) of raw gradients; `mom` = cos(m_t, m_t-1); `hyper` = cos(g_t, O_t-1), the current gradient against the previous update (its dot product is minus the derivative of the loss with respect to the previous step size); `path` / `path100` = cos(g_t, P_t-1) with P a decayed sum of past updates (decay 0.9 / 0.99).

Takeaway: `upd` rises with the rate (0.81 -> 0.92) and `mom` sits at 0.95: both are useless or reversed as signals. **Correction:** the training windows were read in order, so consecutive batches came from the same documents and the raw gradient cosine was inflated to 0.3-0.5. All signal numbers in rounds 2-3 are dropped; the data are kept in `old/results_unshuffled.jsonl`. From round 4 the windows are shuffled once (fixed permutation), as in real training.

## Round 4: signals with shuffled data, across width, depth and batch

Rule under test: *the cosine between the current gradient and the previous Muon update (`hyper`) crosses zero at the tuned rate in every setting.*

| Setting | best lr (val) | hyper = 0 at lr | grad = 0 at lr |
|---|---|---|---|
| base (w64, L4, b16, 200 steps) | 0.005-0.01 (4.052 / 4.058) | 0.0075 | 0.011 |
| width 32 | 0.01 (4.103) | 0.0086 | 0.025 |
| width 128 | 0.005-0.01 (4.017 / 4.018) | 0.0074 | 0.0085 |
| depth 2 | 0.01 (4.045) | 0.0135 | 0.02 |
| depth 8 | >= 0.01 (4.057 at 0.01) | 0.006 | 0.009 |
| batch 64, 50 steps | 0.01-0.02 (4.194 / 4.197) | 0.016 | 0.014 |
| batch 4, 800 steps | 0.005 (4.067) | 0.0013 | none (0.08-0.12 at every lr) |

Base sweep: lr 0.0007 4.108, 0.0025 4.071, 0.005 4.052, 0.01 4.058, 0.02 4.069, 0.04 4.110, 0.07 4.140.

Takeaway: `hyper` = 0 lands inside the best-rate range in 6 of 7 settings, including the 4x move of the best rate with batch size. It fails for tiny batches with many steps (batch 4), where the noise the last step injected makes the next gradient oppose it at every rate, so the zero sits 4x too low. `grad` = 0 is 2x too high at width 32 and depth 2 and has no zero at batch 4.

## Round 5: first controller, no weight decay

Rule: log s += eta * (cos_hyper - 0) per matrix every step, adapting until the cooldown starts.

| Start lr (tuned 0.005-0.01) | 0.0007 | 0.007 | 0.07 |
|---|---|---|---|
| constant | 4.108 | ~4.054 | 4.140 |
| mult, eta 3 | 4.080 | 4.087 | 4.107 |
| mult, eta 1 | **4.057** | 4.082 | 4.137 |
| sign, eta 0.02 | 4.081 | 4.054 | 4.132 |

Takeaway: from 10x too low the rule recovers the tuned loss (4.057 vs 4.052); from 10x too high it barely moves (0.07 -> 0.045). Per-step probe (`probe.jsonl`): at the tuned rate the signal is +0.02-0.03 in steps 10-50, -0.01 mid-run and +0.02 in cooldown, so it averages to zero only over the run. At 0.07 the mid-run signal (-0.012) is no more negative than at 0.007 (-0.008): without weight decay, a too-high rate inflates the weight norms in the first steps, which shrinks the angular step, and the signal stops reporting the rate as too high. In the 0.07 run the value and output matrices of layers 1-3 even report "grow".

## Round 6: weight decay makes the signal monotone

Decoupled weight decay on Muon matrices, w *= 1 - wd * lr_t * s (wd 2, so lr * wd * steps is about 4, as in long runs that reach a norm equilibrium).

| Muon lr, wd 2 | 0.0015 | 0.0025 | 0.005 | 0.01 | 0.02 | 0.04 | 0.08 | 0.15 |
|---|---|---|---|---|---|---|---|---|
| val | 4.086 | 4.069 | 4.047 | 4.041 | **4.039** | 4.055 | 4.081 | 4.109 |
| hyper | | 0.022 | 0.012 | -0.004 | -0.030 | -0.043 | -0.044 | |

Controller from 10x off, wd 2 (tuned constant 4.039):

| Start lr | 0.0015 | 0.015 | 0.15 |
|---|---|---|---|
| constant | 4.086 | ~4.040 | 4.109 |
| mult, eta 1 | 4.047 | **4.037** | 4.048 |
| sign, eta 0.03 | 4.056 | **4.033** | 4.051 |

Takeaway: with weight decay the hyper signal falls monotonically with the rate and the rule closes 83-87% of the gap from both 10x starts (+0.008 / +0.009 vs +0.047 / +0.070); from a right start it slightly beats the tuned constant. The three starts end at 0.005 / 0.007 / 0.012: not yet converged when adaptation stops at the cooldown.

## Round 7: controller variants at wd 2 (base config)

Rule: log s += eta·(cos_hyper − 0), from three starts 10x apart (tuned constant 4.039).

| Variant | 0.0015 | 0.015 | 0.15 |
|---|---|---|---|
| mult eta 1 (round 6) | 4.047 | 4.037 | 4.048 |
| mult eta 2 | 4.045 | 4.0365 | 4.047 |
| mult eta 4 | 4.918 | 4.054 | diverged |
| global scale, mult eta 2 | **4.040** | **4.036** | 4.048 |
| sign, eta 0.05 | 4.0525 | 4.032 | 4.059 |
| EMA 0.9 of the cosine, eta 3 | diverged | diverged | diverged |
| per row, eta 2 | 4.215 | 5.018 | diverged |

Takeaway: eta 1-2 works; eta 4 and any smoothing diverge, because Muon's momentum 0.95 already delays the signal by about 20 steps and a smoothed signal adds more delay to a feedback loop. Per-row scales are unstable. One global scale (mean of the matrices' signals) is as good as per-matrix scales and more stable; it is the default from round 13.

## Round 8: weight-decay tuning

| wd | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| best constant val (lr 0.01 / 0.02) | 4.038 | **4.034** | 4.039 | 4.063 | 4.105 |

Base sweep at wd 1: lr 0.0025 4.069, 0.005 4.046, **0.01 4.034**, 0.02 4.038, 0.04 4.051, 0.08 4.088; 0.001 4.097, 0.1 4.102. Takeaway: wd 1 is used from here on, so the baseline gets its best weight decay.

## Round 9: per-matrix rule over a 100x range of starts (wd 1, base config, eta 2)

| Start lr | 0.001 | 0.002 | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 |
|---|---|---|---|---|---|---|---|
| constant | 4.097 | | 4.046 | **4.034** | 4.038 | | 4.102 |
| rule | 4.043 | 4.043 | 4.046 | 4.050 | 4.046 | 4.064 | 4.057 |

Takeaway: every start ends within 0.009-0.030 of the tuned constant, while a constant 10x off loses 0.063-0.068; from the tuned start the rule costs 0.016. At 200 steps (0.45 tokens per weight) the rule is a safety net, not a win.

## Round 10: gain schedules (eta falls linearly or as 1/sqrt(t), eta 3)

| Start lr | 0.001 | 0.01 | 0.1 |
|---|---|---|---|
| linear | 4.046 | 4.054 | 4.061 |
| sqrt | 4.046 | 4.058 | 4.061 |

Takeaway: all starts converge to about 0.005, half the tuned rate, and lose 0.012-0.027. Making the controller settle faster does not fix the level it settles at.

## Round 11: short runs at width 32 (vocabulary 1,000, 300 steps)

Constant: 0.005 3.398, **0.01 3.387**, 0.02 3.391, 0.04 3.411. Per-matrix rule with linear gain diverged from 0.02 and 0.2: matrices `b0.up` and the gate/up pairs ran to the 1000x bound, and the decay factor 1 − wd·lr·s went negative. Global rule: 3.417-3.438 from 0.002 / 0.02 / 0.2.

Takeaway: in short runs the global rule overshoots early (rate 0.03-0.07 in the first 50 steps) and then settles at 0.002-0.003, a quarter of the tuned rate. Per-matrix scales can run away on gate/up. **Correction to the plan:** per-matrix scales are dropped as the main rule; the global scale stays.

## Round 12: where each signal crosses zero (wd 1, base config)

| Signal | zero at lr | best constant |
|---|---|---|
| hyper cos(g_t, O_t−1) | 0.0075 | 0.01 |
| disp cos(g_t, W0 − W) (Mechanic-like) | 0.0065 | 0.01 |
| grad cos(g_t, g_t−1) | 0.006-0.007 | 0.01 |

Takeaway: every alignment signal crosses zero below the best constant rate. This is the short-horizon bias of greedy step sizes (Wu et al. 2018): the signal measures the next-step loss, which pays for gradient noise that the final cooldown later removes.

## Round 13: long training, small batch (the main setting)

Rule under test: *with many steps, one global scale driven by cos(g_t, O_t−1) beats the tuned constant from any start.* Config: width 32, 4 layers, vocabulary 1,000, batch 8, 900 steps, wd 1, 4.1 tokens per weight, 13-20 s per run.

| Start lr | 0.0004 | 0.001 | 0.002 | 0.0025 | 0.004 | 0.005 | 0.01 | 0.02 | 0.04 |
|---|---|---|---|---|---|---|---|---|---|
| constant, WSD | 3.4012 | 3.3702 | | 3.3546 | | **3.3534** | 3.3660 | 3.3752 | 3.4025 |
| constant, cosine decay | | | | 3.3594 | | **3.3503** | 3.3531 | | |
| global rule, eta 1 | **3.3448** | 3.3471 | **3.3440** | | 3.3476 | | 3.3510 | 3.3459 | 3.3511 |
| per-matrix rule, eta 1 | 3.3630 | | | | 3.3672 | | | | 3.3708 |

Takeaway: from every start across a 100x range the global rule beats the tuned WSD constant (by 0.002-0.009) and ties or beats the tuned cosine schedule. The logged rate rises to 0.011-0.017 by step 100 and then falls to about 0.001 by step 600: the rule learns a warmup and a decay schedule on its own. Per-matrix scales lose 0.01-0.02 to the global scale.

## Round 14: controls on the main setting

| Start lr | 0.0004 | 0.004 | 0.04 |
|---|---|---|---|
| global rule | **3.3448** | **3.3476** | **3.3511** |
| same rule, random sign of the signal | 3.4195 | 3.3612 | 3.3919 |
| same rule, frozen after step 100 | 3.3658 | 3.3644 | 3.3766 |
| tuned constant (one rate for all) | 3.3534 | 3.3534 | 3.3534 |

Takeaway: the sign of the alignment carries the information (random sign loses 0.014-0.075), and the adaptation must continue through training (freezing at step 100 loses 0.017-0.026): the rule's value is the schedule it keeps adjusting, not only the first 100 steps of rate search.

## Round 15-16: big batch, few steps (batch 32, 225 steps, same tokens)

| Start lr (tuned 0.01-0.02) | 0.0015 | 0.015 | 0.15 |
|---|---|---|---|
| constant | | 3.3590 (0.01) / 3.3600 (0.02) | |
| global rule, target 0 | 3.3712 | 3.3841 | 3.4002 |
| target −0.01 | 3.3648 | 3.3744 | 3.3897 |
| target −0.02 | **3.3600** | 3.3718 | 3.3848 |
| target −0.02, eta 2 | 3.3753 | 3.3993 | diverged |

Signal means over the flat phase: hyper 0.005 / −0.019 / −0.058 / −0.093 at lr 0.005 / 0.01 / 0.02 / 0.04; zero at about 0.0065.

Takeaway: the rule loses with few steps. The trajectory shows why: positive alignment early (the loss is falling fast) pushes the scale up for 50 steps (from 0.15 to 0.18), and then the rate falls to 0.003-0.004 by step 100, a quarter of the tuned rate. A negative target raises the settled rate and helps the low start, but cannot remove 50 steps spent at 10x in a 225-step run; a faster gain diverges.

## Round 17: batch size vs steps at fixed tokens (width 32, 4.1 tokens per weight)

| Batch / steps | tuned constant (lr) | rule from 0.1x / 1x / 10x of tuned | constant 10x off (low / high) |
|---|---|---|---|
| 4 / 1,800 | 3.3812 (0.0025) | 3.3813 / **3.3762** / **3.3773** | 3.40+ (not run) |
| 8 / 900 | 3.3534 (0.005) | **3.3448** / **3.3476** / **3.3511** | 3.4012 / 3.4025 |
| 16 / 450 | 3.3455 (0.005) | 3.3494 / 3.3596 / 3.3561 | 3.3818 / 3.4019 |
| 32 / 225 | 3.3590 (0.01) | 3.3712 / 3.3841 / 3.4002 | — |

Takeaway: the rule ties or beats the tuned constant at 900 steps or more and loses at 450 steps or fewer, at the same tokens. What matters is the number of steps compared with the controller's response time (log 10 / (eta·|cos|) ≈ 50-100 steps). Large models train for 10^4-10^5 steps, which puts them on the winning side.

## Round 18-20: width, depth and tokens per weight (batch 8, 900 steps unless noted)

| Setting | tokens/weight | tuned constant (lr) | rule from 0.1x / 1x / 10x |
|---|---|---|---|
| width 16 | 10.5 | 3.4021 (0.005) | **3.4010** / 3.4051 / 3.4101 |
| width 32 (main) | 4.1 | 3.3534 (0.005) | **3.3448** / **3.3476** / **3.3511** |
| width 64 | 1.4 | 3.3116 (0.0025) | **3.3074** / **3.3065** / **3.3084** |
| depth 2 | 5.3 | 3.3606 (0.0025) | **3.3531** / **3.3513** / **3.3558** |
| depth 6 | 3.4 | 3.3489 (0.0025) | **3.3483** / **3.3435** / **3.3433** |
| width 16, depth 2, 1,800 steps | 24.3 | 3.3579 (0.0025) | **3.3565** / **3.3577** / 3.3659 |

Depth 8: constants 3.3369 / 3.3385 / 3.3603 at 0.0025 / 0.005 / 0.01; each run took 35 s, over the run-length limit, so the rule was not run at depth 8.

Takeaway: across width 16-64, depth 2-6 and 1.4-24 tokens per weight, the rule beats or ties the tuned constant in 15 of 18 runs; the three losses are 10x-high or 1x starts at the two settings with the most tokens per weight (+0.003 to +0.008). At 24 tokens per weight the rule's rate decays 55x from its peak (0.011 at step 200 to 0.0002 at step 1,200).

## Round 21: can the rule replace the final cooldown?

Rule adapting to the last step with no cooldown (main setting): 3.4028 / 3.3955 / 3.3979 from 0.0004 / 0.004 / 0.04, against 3.345-3.351 with the cooldown.

Takeaway: no. The rule replaces the rate sweep and the mid-run decay; the final linear cooldown to zero stays. The rule's rate ends near 0.0005, not 0, and the last 30% of training at a rate falling to zero is worth 0.05.

## Round 22: sensitivity to the rule's own settings (main setting)

| Variant | start 0.0004 | start 0.04 |
|---|---|---|
| eta 0.5 | 3.3491 | 3.3499 |
| eta 1 (default) | 3.3448 | 3.3511 |
| eta 2 | 3.3476 | 3.3494 |
| target −0.005 | 3.3511 | 3.3508 |
| target +0.005 | 3.3614 | 3.3651 |

Takeaway: eta 0.5-2 and target −0.005 to 0 all beat the tuned constant (3.3534); the rule has no knob that needs tuning in this range. A positive target lowers the rate and loses 0.01.

## Round 23-25: which matrices want which rates, and is it worth using?

Per-matrix rule on the base config (round 9 runs), final scale of each matrix kind relative to the geometric mean, over 7 starts from 0.001 to 0.1: q 3.2-6.5x, k 2.0-4.8x, v 0.26-0.50x, o 0.43-1.47x, gate 0.53-0.99x, up 0.22-1.11x, down 0.38-0.77x. The order q, k high and v low holds at every start; layer 0 v and o are lowest (0.1-0.3x).

Direct test: fixed rate multipliers per matrix kind on top of the tuned constant (base config, lr 0.01).

| Multiplier | with QK-norm (tuned 4.0343) | without QK-norm (tuned 4.0591) |
|---|---|---|
| q, k × 16 | 4.0358 | **4.0361** |
| q, k × 4 | 4.0368 | |
| q, k × 0.25 | 4.0503 | 4.0913 |
| v, o × 4 | 4.0424 | |
| v, o × 0.25 | 4.0414 | |
| MLP × 4 | 4.0466 | |
| MLP × 0.25 | 4.0407 | |

Takeaway: the rule's q/k signal is real. Query and key matrices are hurt by a lower rate and not by a 16x higher one; without QK-norm a 16x rate on q and k gains 0.023, as much as adding QK-norm. Attention starts uniform and the logits grow with |W_q||W_k|; Muon's fixed-size update grows them slowly, and QK-norm's gain (trained by Adam) does the same job faster. Value, output and MLP matrices lose 0.006-0.012 at 4x or 0.25x.

Global scale plus a per-matrix offset (offset rate eta2, offsets bounded to 1/16-16x):

| Setting | constant | global rule | global + offsets |
|---|---|---|---|
| base, no QK-norm, start 0.01 | 4.0591 | 4.0617 | **4.0493** (eta2 0.5) / **4.0482** (eta2 2) |
| main setting with QK-norm, 0.0004 / 0.004 / 0.04 | 3.3534 | **3.3448 / 3.3476 / 3.3511** | 3.3502 / 3.3542 / 3.3559 |

Takeaway: per-matrix offsets recover half of the q/k gain without QK-norm and cost 0.005 with it. With QK-norm the global scale is enough.

## Round 26: which signal can drive the controller (main setting, start 0.004)

| Signal | val |
|---|---|
| hyper cos(g_t, O_t−1) | **3.3476** |
| grad cos(g_t, g_t−1) | 3.3744 |
| disp cos(g_t, W0 − W) | 3.3786 |
| upd cos(O_t, O_t−1) | diverged (scale to 1000x) |

Takeaway: only the gradient-against-last-update signal beats the tuned constant. The update cosine is always positive (consecutive updates share 95% of their momentum), so it drives the rate up without limit.

## Round 27: does adapting during warmup hurt?

The logged step size peaks at 0.03-0.07 at step 50 (6-14x the tuned rate, figure 3): during warmup the schedule factor is small, the signal is positive and the scale grows. Test: start adapting after warmup (step 45), main setting.

| Start lr | 0.0004 | 0.004 | 0.04 |
|---|---|---|---|
| adapt from step 0 (default) | **3.3448** | 3.3476 | 3.3511 |
| adapt from step 45 | 3.3555 | **3.3451** | **3.3463** |

Takeaway: mixed; from 10x too low the warmup is where the rule finds its rate (skipping it loses 0.011), from the right or a high start skipping it gains 0.002-0.005. The early overshoot is not harmful on average; adaptation from step 0 stays the default because it needs no knowledge of the right rate.

## Round 28 (after review): a tuned linear decay as baseline

Rule under test: *the rule beats every tuned schedule, not only the flat one.* Reviewer point: the rule learns a long decay, and the flat-then-cooldown baseline holds the rate flat for 65% of training, so a tuned linear decay from the end of warmup to zero may give the same gain. Main setting.

| Start or rate | 0.0004 | 0.0025 | 0.004-0.005 | 0.0075 | 0.01 | 0.04 |
|---|---|---|---|---|---|---|
| constant, linear decay to 0 | 3.4146 | 3.3539 | **3.3430** | 3.3441 | 3.3462 | 3.3765 |
| rule on flat schedule (round 13) | 3.3448 | 3.3440 (0.002) | 3.3476 | | 3.3510 | 3.3511 |
| rule on linear decay | 3.3467 | | 3.3469 | | | 3.3499 |

Takeaway: the rule broke. A tuned linear decay (3.3430) is 0.001-0.008 better than the rule from every start, and the rule on top of linear decay adds nothing. The rule's learned decay does the job of a tuned linear decay; its value is that it reaches that loss from a start 10x off with no sweep, while a linear decay 8-10x off loses 0.033-0.072. Claim 1 changes from "beats the tuned schedule" to "matches the best tuned schedule from any start".

## Round 29: learning-rate-free baselines in Muon's shared-multiplier slot (main setting)

Rule under test: *the rule is the only method in this slot that reaches the swept loss from a start 10x too high.* All three baselines replace the same global multiplier on Muon's rate (`--lrfree`); Adam parts unchanged.
- **D-Adaptation-Muon:** estimate d̂ = Σ λ_t ⟨g_t, W0 − W_t⟩ / ‖Σ λ_t g_t‖_* summed over matrices, with the nuclear norm (the dual of the spectral norm Muon descends in) instead of D-Adaptation's L1/L2 norm; d = max(d, d̂); step = d · schedule · O · shape factor; λ_t = d · schedule. Starts at d0 = the given start rate.
- **Prodigy-Muon:** the same with Prodigy's weights (β = sqrt(0.999) decay, weight (1 − β) · schedule · d²).
- **Mechanic:** W = W0 + S · Δ, where Δ is the sum of base Muon steps (base rate = the start rate, with weight decay) and S comes from Mechanic's tuner (6 betas, λ = 0.01, s_init = 1e-8).

| Method (WSD schedule unless noted) | start 0.0004 | start 0.004 | start 0.04 |
|---|---|---|---|
| tuned constant (swept) | 3.4012 | 3.3534 (0.005) | 3.4025 |
| **the rule** | **3.3448** | **3.3476** | **3.3511** |
| D-Adaptation-Muon | 3.3546 (d grows to 0.0048) | 3.3547 | 3.4002 (d cannot fall) |
| Prodigy-Muon | 3.3573 | 3.3547 | 3.4002 |
| Mechanic | 3.3605 | 3.3671 | 3.3680 |
| D-Adaptation-Muon, d0 = 1e-6 | 3.3611 | | |
| D-Adaptation-Muon on linear decay | 3.3469 | | 3.3756 |
| D-Adaptation-Muon, d0 = 1e-6, linear decay | 3.3607 | | |
| Prodigy-Muon, d0 = 1e-6, linear decay | 3.3581 | | |

Takeaway: D-Adaptation and Prodigy only raise their estimate (d = max(d, d̂)), so from a low start they find the rate (3.3546-3.3573, within 0.002-0.004 of the constant) and from a high start they stay there and lose 0.047. Mechanic's S loses 0.007-0.015 from every start. The rule is the only method that comes down from 10x too high. On linear decay D-Adaptation from a low start ties the rule (3.3469 vs 3.3467).

## Round 30: the same rule on the Adam-trained parts (embedding, head, norms)

Rule under test: *a second shared multiplier on Adam's rate, driven by cos(g_t, the step Adam just took), removes the Adam rate sweep too.*

Adam rate sweep at Muon 0.005: 0.004 3.3955, 0.01 3.3643, 0.02 3.3559, **0.04 3.3534**, 0.08 3.3579, 0.4 3.4290.

| Adam-side signal, starts (Muon, Adam) | (0.0004, 0.004) | (0.004, 0.04) | (0.04, 0.4) |
|---|---|---|---|
| mean over all 19 Adam tensors | 3.4032 | 3.4144 | 3.4085 |
| embedding and head only (`amat`), gain 1 | **3.3426** | **3.3528** | 3.3822 |
| embedding and head only, gain 5 | 3.3639 | 3.3674 | 3.3755 |

Diagnosis of the first row: 17 norm-gain vectors dominate the mean and their cosine crosses zero near Adam rate 0.01 (0.25x the tuned rate), so the rate settles at 0.007-0.03. The embedding and head alone cross zero near 0.5x. The best run of the whole paper is the two-rule run from (0.0004, 0.004): 3.3426, both rates started 10x low.

Takeaway: the Adam-side rule (embedding and head only) works from low and right starts and fails from 10x too high: there the embedding and head cosines are 10x smaller than Muon's (−0.003 vs −0.03), so the multiplier falls too slowly.

## Round 31: early overshoot, loss spike and two fixes (main setting)

Rule under test: *the early step-size peak (6-14x the tuned rate at step 50) causes a loss spike.* Measure: training-loss moving average of each rule run minus that of the constant run at 0.005 with the same data order and initial weights (`lc.py`), every 10 steps.

| Variant | start | val | peak rate (× tuned 0.005) | largest excess over constant | mean excess, steps 0-150 |
|---|---|---|---|---|---|
| rule (default) | 0.0004 | 3.3448 | 0.033 (6.5x) | +0.020 at step 20 (start too low) | −0.007 |
| | 0.004 | 3.3476 | 0.056 (11x) | +0.007 at step 240 | −0.041 |
| | 0.04 | 3.3511 | 0.108 (22x) | +0.026 at step 140 | −0.062 |
| cap 0.01 on the change of log s per step | 0.0004 | 3.3783 | 0.001 | +0.042 | +0.028 |
| | 0.004 | 3.3465 | 0.006 | +0.006 | +0.001 |
| | 0.04 | 3.3631 | 0.046 | +0.030 | −0.052 |
| cap 0.03 | 0.0004 | 3.3515 | 0.003 | +0.041 | +0.021 |
| gain × schedule factor (small in warmup) | 0.0004 | 3.3540 | 0.005 (1.1x) | +0.041 at step 20 (start too low) | +0.015 |
| | 0.004 | 3.3489 | 0.018 (3.5x) | +0.002 | −0.007 |
| | 0.04 | **3.3448** | 0.049 (10x) | +0.007 | −0.066 |

Takeaway: the overshoot causes no loss spike: the rule's training loss is below the constant run's over the first 150 steps from right and high starts, and the largest excess, +0.026, is the high start's first steps before it comes down. A per-step cap stops the low start from finding its rate (+0.025 at cap 0.01). Scaling the gain by the warmup factor cuts the peak 2-7x at a cost of +0.009 from 10x low and a gain of 0.006 from 10x high; it is the fix to use where a large step is a risk.

## Round 32: making the Adam-side rule recover from a high start

| Variant, starts (Muon, Adam) | (0.0004, 0.004) | (0.004, 0.04) | (0.04, 0.4) |
|---|---|---|---|
| `amat`, gain 3, gain × schedule factor | 3.3553 | 3.3567 | 3.3708 |
| Adam rate tied to Muon's multiplier (fixed ratio) | 3.3759 | 3.3725 | 3.3922 |
| `amat`, signal / running RMS × 0.01 | 3.3715 | 3.3468 | 3.4137 |
| `amat`, signal against a decayed sum of past steps (β 0.9) | 3.4031 | 3.3912 | 3.4033 |

Tied rates lose 0.02-0.04 from every start: Muon's learned 25-57x decay is wrong for Adam's parts, which want a near-constant rate. The RMS-normalized signal is mostly noise; the decayed-sum signal pushes the rate too low (rates end at 0.0014-0.006).

Low Adam starts with the plain `amat` rule (gain 1):

| Starts (Muon, Adam) | (0.04, 0.004) | (0.0004, 0.001) | (0.04, 0.001) | (0.004, 0.004) | (0.004, 0.01) |
|---|---|---|---|---|---|
| val | 3.3524 | 3.3585 | 3.3553 | 3.3486 | 3.3481 |
| final Adam multiplier | 2.6 | 12.2 | 10.1 | 3.3 | 1.8 |

Takeaway: with the Adam rate started anywhere from 40x to 1x below tuned and Muon anywhere from 10x low to 10x high, the two rules land at 3.3426-3.3585 against 3.3534 for both rates swept. A 10x-high Adam start is not recovered by any of five variants. Recipe: start Adam low.

## Round 33: depth 8 with the rule (width 32, batch 8, 700 steps; 20 s per run)

| Rate or start | 0.0004 | 0.0025 | 0.004 | 0.005 | 0.01 | 0.04 |
|---|---|---|---|---|---|---|
| constant | 3.4227 | 3.3697 | | **3.3672** | 3.3788 | 3.4142 |
| rule | 3.3685 | | 3.3698 | | | **3.3657** |

Takeaway: at depth 8 the rule from three starts spanning 100x lands at 3.3657-3.3698, within 0.003 of the tuned constant; a constant 10x off loses 0.047-0.055.

## Round 34: width 32 / 64 / 128 (2 layers, batch 8, 900 steps)

| Width | tuned constant (lr) | rule from 0.0003 / 0.003 / 0.03 | rule's mean flat-phase rate | rms of the signal |
|---|---|---|---|---|
| 32 | 3.3606 (0.0025) | **3.3525** / **3.3575** / **3.3558** | 0.0052 / 0.0045 / 0.0056 | 0.043 |
| 64 | 3.3202 (0.005) | **3.3165** / **3.3165** / 3.3238 | 0.0051 / 0.0055 / 0.0069 | 0.027 |
| 128 | 3.3036 (0.0025) | **3.3002** / **3.3010** / 3.3056 | 0.0028 / 0.0054 / 0.0088 | 0.018 |
| 128, gain 2 | | **3.2978** / **3.2998** / **3.3008** | 0.0036 / 0.0037 / 0.0062 | |
| 32, signal / running RMS × 0.04 | | 3.3685 / 3.3529 / 3.3565 | 0.0017 / 0.0047 / 0.0087 | |
| 128, signal / running RMS × 0.04 | | 3.3064 / 3.2999 / 3.3064 | 0.0014 / 0.0035 / 0.0080 | |

Constant sweeps (0.00125 / 0.0025 / 0.005 / 0.01): width 32 3.3789 / 3.3606 / 3.3615 / 3.3695; width 64 3.3360 / 3.3222 / 3.3202 / 3.3375; width 128 3.3060 / 3.3036 / 3.3125 / 3.3277.

Takeaway: the rule ties or beats the tuned constant at every width (within +0.002 to −0.008). Its mean rate sits at the tuned rate (0.0025-0.005 at every width), but the signal shrinks with width (rms 0.043 → 0.018, about width^−0.65), so at width 128 the three starts no longer converge (3.1x spread of the mean rate vs 1.2x at width 32). Gain 2 at width 128 (gain ∝ sqrt(width / 32)) restores it: spread 1.7x, all three starts 0.003-0.006 below the constant. Dividing by the signal's running RMS slows the controller and is dropped.

## Round 35: 40 tokens per weight breaks the full-run rule (width 16, 2 layers, 2 heads, 3,000 steps, 22 s per run)

Rule under test: *the rule matches the swept constant at any training length.* Round 20 showed a tie at 24 tokens per weight (1,800 steps).

| Variant | start 0.0003 | start 0.003 | start 0.03 | mean flat rate |
|---|---|---|---|---|
| constant (0.0003 / 0.00125 / **0.0025** / 0.005 / 0.01 / 0.03) | 3.3614 | 3.3321 / **3.3289** / 3.3327 / 3.3481 | 3.3670 | |
| rule, adapting to the cooldown (default) | 3.3488 | 3.3451 | 3.3406 | 0.0003-0.0005 |
| rule, target −0.01 | 3.3328 | 3.3330 | 3.3356 | 0.0038-0.0061 |
| rule, target −0.02 | 3.3505 | 3.3726 | diverged | 0.012-0.041 |
| rule, adapt until step 450 (15%) then hold | **3.3301** | 3.3319 | 3.3317 | 0.0021-0.0033 |
| rule, adapt until step 750 (25%) then hold | 3.3327 | 3.3315 | 3.3312 | 0.0009-0.0013 |
| rule, adapt until step 900 (30%) then hold | 3.3345 | 3.3320 | 3.3322 | 0.0007-0.0016 |

The rule's rate from all three starts is 0.0022-0.0033 at steps 320-480 (tuned 0.0025), then keeps falling to 4-8e-5 by step 1,600. Reason: the signal measured at the tuned constant rate (gain 0, `r35-tpw3k-eta0`) averages −0.01 to −0.02 after step 250, and the rule sums it: a mean of −0.01 over 2,000 steps cuts the rate by e^20. This is the short-horizon bias growing as training moves into the noise-dominated phase, where the next-step-optimal rate falls toward zero (Wu et al. 2018).

Same holds at 900 steps (main setting, starts 0.0004 / 0.004 / 0.04): adapt to 15% then hold 3.3598 / 3.3577 / 3.3638 (held rate 0.009-0.011, 2x too high: the early overshoot has not come down yet); adapt to 30% then hold 3.3527 / 3.3583 / 3.3591.

Takeaway: the full-run rule broke at 40 tokens per weight: it loses 0.012-0.020, as much as a constant 4x off. Its learned decay helps at up to 24 tokens per weight and over-decays beyond. A negative target is fragile (−0.01 is 0.004-0.007 behind, −0.02 runs away). Adapting for the first 30% of training then holding ties the tuned constant at both lengths (+0.003-0.006 at 3,000 steps where a 10x-off constant loses 0.033-0.038; −0.001 to +0.006 at 900 steps).

## Round 36: the safe recipe (gain × warmup factor, adapt for the first 30%, then hold)

| Setting | tuned constant | start 0.1x | start 1x | start 10x | peak rate |
|---|---|---|---|---|---|
| main (900 steps, 4.1 tokens per weight) | 3.3534 | 3.3529 | **3.3520** | 3.3537 | 0.005-0.049 |
| 3,000 steps, 40 tokens per weight | 3.3289 | 3.3339 | 3.3332 | **3.3293** | 0.012-0.047 |
| main, linear-decay schedule, adapt 30% (no warmup gain) | 3.3430 (tuned linear decay) | 3.3450 | 3.3455 | 3.3487 | 0.033-0.108 |
| 3,000 steps, linear-decay schedule, adapt 30% (no warmup gain) | 3.3256 (tuned linear decay; 0.0025 3.3267, 0.01 3.3304) | 3.3393 | 3.3344 | 3.3303 | 0.085-0.257 |

Takeaway: the safe recipe lands within −0.001 to +0.005 of the tuned constant at both training lengths from starts spanning 100x, with the peak step at most 10x the tuned rate. It gives up the full-run rule's 0.002-0.009 margin at short runs in exchange for holding at long ones. On linear decay the held version is 0.002-0.014 behind the tuned linear decay.

## Round 37: the safe recipe at few large batches and at 89 tokens per weight

Rule under test: *the safe recipe (warmup-scaled gain, adapt for the first 30%, hold) ties the swept constant from any start at any step count and training length.* Main model (width 32, 4 layers) for batch against steps; width 8, 2 layers, 2 heads, 3,000 steps (89 tokens per weight, 24 s per run) for training length. (Three first attempts ran with the script's defaults, eta 0.05 and weight decay 0; they were deleted and rerun with `--eta 1 --wd 1`.)

**Batch against steps (same tokens).**

| Batch × steps | tuned constant | full rule (rounds 15/17) | safe recipe, hold at 30% | held rate | other variants |
|---|---|---|---|---|---|
| 8 × 900 | 3.3534 | 3.3448 / 3.3476 / 3.3511 | 3.3529 / 3.3520 / 3.3537 | | |
| 16 × 450 (starts 0.001 / 0.01 / 0.1) | 3.3455 (0.005) | 3.3494 / 3.3596 / 3.3561 | **3.3427 / 3.3458** / 3.3507 | 0.0042-0.0057 | hold at step 300: 3.3502 / 3.3479 / 3.3581 (held 0.0014-0.0018) |
| 32 × 225 (starts 0.0015 / 0.015 / 0.15) | 3.3590 (0.01) | 3.3712 / 3.3841 / 3.4002 | 3.3663 / 3.3687 / 3.3733 | 0.0069-0.0164 | gain 2: 3.3634 / 3.3746 / 3.3788; hold at 50%: 3.3742 / 3.3784 / 3.3891 (held 0.004-0.0056) |

Takeaway: the safe recipe moves the step-count boundary of claim 4 from 900 down to 450 steps: at 16 × 450 it ties (−0.003 to +0.005) where the full rule lost 0.004-0.014. At 32 × 225 it still loses 0.007-0.014 (half the full rule's loss): the held rate from the low and right starts is 0.006-0.007, the greedy rate, and the best constant for a 225-step run is 0.01-0.02, twice that. Adapting longer lowers the held rate further and loses more, a faster gain does not help. The greedy rate is a fixed fraction below the run-optimal rate only when the run is long compared with the controller's 50-100-step response; at 225 steps it is not.

**89 tokens per weight (width 8, 2 layers, 3,000 steps).** Constants: 0.0003 3.4687, **0.0025 3.4353**, 0.005 3.4379, 0.01 3.4464, 0.02 3.4448, 0.03 3.4600 (10x off costs 0.025-0.033).

| Variant (gain × warmup factor) | start 0.0003 | start 0.003 | start 0.03 | held rate |
|---|---|---|---|---|
| full rule (gain 1, adapting to the cooldown) | 3.4577 | 3.4552 | 3.4490 | falls to 0.00012-0.00018 by step 2,000 |
| hold at step 300 (10%) | 3.4420 | 3.4446 | 3.4300 | 0.0033-0.0097 |
| **hold at step 450 (15%)** | **3.4403** | **3.4324** | **3.4274** | 0.0016-0.0040 |
| hold at step 900 (30%, safe recipe) | 3.4465 | 3.4337 | 3.4306 | 0.0005-0.0009 |

Takeaway: at 89 tokens per weight the full rule loses 0.014-0.022 and the 30% hold loses 0.011 from the low start, because the greedy bias has already pulled the rate 3-5x below the tuned one by step 900. Holding at step 450 ties from all three starts (+0.005, −0.003, −0.008). Holding at step 300 is too early: the high-start peak has not come down (held up to 0.0097).

**Hold point across all settings.** Good hold points: 900 steps → 270 (135 too early); 16 × 450 → 135 (300 too late); 3,000 steps at 40 tokens per weight → 450-900; 3,000 steps at 89 tokens per weight → 450 (300 too early, 900 too late). All are fitted by **hold at the earlier of 30% of training and step 450**: long enough for 5-10 controller response times after the warmup peak (each about 50-100 steps), short enough that the greedy bias has not yet accumulated. Only 32 × 225 has no good hold point.

Updated safe recipe: gain × warmup factor × sqrt(width/32), adapt until min(30% of steps, 450 steps), then hold, keep the cooldown.

## Round 38: the safe recipe on six settings it was never fitted on (no retuning)

Rule under test: *the safe recipe (gain = warmup factor × sqrt(width/32), adapt until min(30% of steps, 450), hold, keep the cooldown) lands within 0.005 of the tuned schedule from any start in a 100x range, and beats D-Adaptation, Prodigy and Mechanic.* Every setting below is new; the recipe's constants were not changed. Starts are 10x below, at and 10x above the tuned constant. Baselines: the swept constant (WSD schedule) and a swept linear decay to zero; the safe recipe is run on both schedules (`safe` on WSD, `safelin` on linear decay). The three tuning-free methods run on WSD in the same slot. Runs driven by `drive.py`; several runs took 40-100 s of wall time because three sibling paper sessions were training at the same time (CPU time about 20-35 s each).

| Setting (steps, tokens per weight) | tuned constant | 10x-off constants | safe, WSD (low / at / high) | tuned linear decay | safe, linear decay | D-Adaptation | Prodigy | Mechanic |
|---|---|---|---|---|---|---|---|---|
| depth 6, width 32 (900, 2.9) | 3.3489 (0.0025) | 3.4194 / 3.3902 | **3.3449 / 3.3469 / 3.3452** | 3.3355 (0.01) | 3.3438 / 3.3376 / 3.3424 | 3.3533 / 3.3499 / 3.3845 | 3.3496 / 3.3495 / 3.3845 | 3.3575 / 3.3578 / 3.3664 |
| depth 8, width 32 (700, 1.8) | 3.3672 (0.005) | 3.4227 / 3.4142 | **3.3704 / 3.3632 / 3.3673** | 3.3672 (0.01) | 3.3773 / 3.3677 / 3.3710 | 3.3747 / 3.3680 / 3.4089 | 3.3738 / 3.3680 / 3.4089 | 3.3928 / 3.3972 / 3.4062 |
| vocabulary 6,000 (900, 1.1) | 5.1214 (0.005) | 5.1910 / 5.1881 | 5.1189 / 5.1234 / 5.1291 | 5.1132 (0.01) | **5.1175 / 5.1099 / 5.1196** | 5.1234 / 5.1235 / 5.1872 | 5.1229 / 5.1256 / 5.1872 | 5.1324 / 5.1327 / 5.1430 |
| vocabulary 24,000, batch 4 (900, 0.03) | 6.6376 (0.005); 0.004 6.6482, 0.0055 6.6476, 0.0065 6.6524 | 6.7022 / 6.6835 | 6.6515 / 6.6514 / 6.6525 | 6.6327 (0.01) | 6.6495 / **6.6319** / 6.6404 | 6.6586 / 6.6519 / 6.6894 | 6.6575 / 6.6511 / 6.6894 | 6.6651 / 6.6609 / 6.6713 |
| width 128, depth 2, batch 4 (3,000, 1.2) | 3.2956 (0.00125) | 0.005 (4x): 3.3147 | 3.2942 / 3.2983 / 3.3121 | 3.2729 (0.0025) | 3.2750 / 3.2781 / 3.2842 | 3.2993 / 3.3039 / 3.3372 | 3.2983 / 3.3036 / 3.3388 | 3.2995 / 3.3011 / 3.2985 |
| width 8, depth 2 (4,000, 118) | 3.4180 (0.0025) | 3.4554 / 3.4488 | **3.4185 / 3.4188 / 3.4212** | 3.4174 (0.0025) | **3.4161 / 3.4175 / 3.4186** | 3.4219 / 3.4438 / 3.4547 | 3.4377 / 3.4504 / 3.4566 | 3.4159 / 3.4211 / 3.4271 |

Gaps to the tuned schedule of the same shape (safe on WSD vs constant; safe on linear decay vs linear decay):

| Setting | safe, WSD | safe, linear decay | best of the three tuning-free methods, per start |
|---|---|---|---|
| depth 6 | −0.004 / −0.002 / −0.004 | +0.008 / +0.002 / +0.007 | +0.001 / +0.001 / +0.018 |
| depth 8 | +0.003 / −0.004 / +0.000 | +0.010 / +0.001 / +0.004 | +0.007 / +0.001 / +0.039 |
| vocabulary 6,000 | −0.003 / +0.002 / +0.008 | +0.004 / −0.003 / +0.006 | +0.002 / +0.002 / +0.022 |
| vocabulary 24,000 | +0.014 / +0.014 / +0.015 (neighbouring constants +0.010-0.015) | +0.017 / −0.001 / +0.008 | +0.020 / +0.014 / +0.034 |
| width 128, batch 4 | −0.001 / +0.003 / +0.017 | +0.002 / +0.005 / +0.011 | +0.003 / +0.006 / +0.003 (Mechanic) |
| 118 tokens per weight | +0.000 / +0.001 / +0.003 | −0.001 / +0.000 / +0.001 | −0.002 / +0.003 / +0.009 |

Takeaways:
- **Within 0.005 of the tuned schedule of the same shape from all three starts:** depth 6 and 8, and 118 tokens per weight (both schedules). From two of three starts: vocabulary 6,000 (both schedules) and width 128 (the low and right starts). 10x-off constants lose 0.034-0.070 in the same settings.
- **Vocabulary 24,000 cannot resolve 0.005.** The best constant (0.005, 6.6376) is a 0.010 dip: the constants at 0.004, 0.0055 and 0.0065 give 6.6476-6.6524, and the safe recipe's 6.6514-6.6525 sits with them. Training loss at the end matches the constant run within 0.002.
- **The flat schedule against linear decay.** Where linear decay beats the flat constant by more than 0.005 (depth 6: 0.013; vocabulary 6,000: 0.008; width 128 at batch 4: 0.023), the safe recipe on the flat schedule trails it, and the safe recipe on the linear-decay schedule closes most of that gap. The recipe replaces the rate sweep, not the choice of schedule.
- **Against the tuning-free methods.** From the 10x-high start the safe recipe beats all three in five of the six settings (by 0.009-0.039 against the best of them). At width 128 with batch 4 Mechanic beats it from the high start (3.2985 vs 3.3121): Mechanic's scale can fall as well as rise and it adapts for the whole run, while the safe recipe holds after 300 steps of a 560-step fall. D-Adaptation and Prodigy lose 0.034-0.042 from that start. From the low and right starts D-Adaptation and Prodigy tie it at depth 6 and 8 and vocabulary 6,000; at 118 tokens per weight they overshoot from every start (3.4219-3.4566); Mechanic ties it there from the low start (3.4159).
- **The one break among the new settings: width 128 at batch 4, 10x-high start (+0.011-0.017, and behind Mechanic).** See the response-time rule below. The three width-128 Mechanic runs took 33-45 s of wall time under the sibling sessions' load; no more width-128 runs on this machine.

**Why the recipe fails where it fails: run length and window against the controller's response time** (`resp.py`, high start of each safe run).

| Run | steps | window after warmup | steps per tenfold fall (high start) | window / fall time | held rate / tuned | gap |
|---|---|---|---|---|---|---|
| batch 32 × 225 | 225 | 57 | 56 | 1.03 | 2.29 | +0.014 |
| batch 16 × 450 | 450 | 113 | 79 | 1.44 | 1.19 | +0.005 |
| main | 900 | 225 | 254 | 0.88 | 1.21 | +0.000 |
| depth 6 | 900 | 225 | 218 | 1.03 | 1.17 | −0.004 |
| depth 8 | 700 | 175 | 171 | 1.02 | 0.63 | +0.000 |
| vocabulary 6,000 | 900 | 225 | 245 | 0.92 | 1.15 | +0.008 |
| 40 tokens per weight | 3,000 | 750 | 471 | 1.59 | 0.30 | +0.000 |
| 89 tokens per weight | 3,000 | 300 | 225 | 1.33 | 0.88 | −0.008 |
| 118 tokens per weight | 4,000 | 250 | 273 | 0.92 | 1.20 | +0.003 |
| **width 128, batch 4** | 3,000 | 300 | **559** | **0.54** | **3.62** | **+0.017** |
| width 128, batch 4, hold at 750 | 3,000 | 600 | 682 | 0.88 | 0.83 | +0.003 (low start +0.008) |

Two rules explain the failures:
1. **The window must cover one tenfold fall.** A 10x-high start needs about one fall time (steps per tenfold drop of the step size, ln 10 / (gain · mean |c|) while overshooting) between the end of warmup and the hold step. Wherever window / fall time ≥ 0.9, the high start holds 0.6-1.2x the tuned rate and ties; at width 128 with batch 4 the fall takes 560 steps (the cosine is small there, and the multiplier rises from a low start in about 150 steps per tenfold), the window is 300, and it holds 3.6x the tuned rate. Holding later fixes the high start but lets the greedy drift pull the low start down (+0.008), so the robust fix is to start low: the upward response is 3-4x faster, and from 10x low the recipe is within 0.005 in every setting except batch 32 × 225 and the noisy vocabulary 24,000.
2. **The run must be long compared with the fall time.** At 225 steps (four fall times) the greedy rate the rule holds is half the best rate for so short a run (held 0.006-0.007 from the low and right starts; tuned 0.01-0.02), and all three starts lose 0.007-0.014. At 450+ steps the greedy rate and the tuned constant agree (held/tuned 0.6-1.2 from the right start in every setting).

Where not to use the recipe: runs under about 450 steps; and starts above the true rate when the signal is weak (wide matrices with small batches), unless the hold step is moved to at least one fall time after warmup, which can be read off the logged multiplier.

## Round 39: starts 100x off (a 10,000x range)

Rule under test: *the multiplier rises 3-4x faster than it falls, so the recipe should tolerate starts far below the tuned rate better than far above.* Prediction: from 100x low the rise takes about 100 steps after warmup and the run ties or loses a little; from 100x high it needs two fall times (about 450-500 steps), far more than the 113-225-step window, and loses. Safe recipe unchanged, WSD schedule, gap to the tuned constant.

| Setting | tuned constant | 100x low: held / tuned, gap | 10x low gap (rounds 36-38) | 10x high gap | 100x high: held / tuned, gap |
|---|---|---|---|---|---|
| main (900 steps) | 3.3534 (0.005) | 0.54, **−0.001** | −0.001 | +0.000 | 1.67, +0.008 |
| batch 16 × 450 | 3.3455 (0.005) | 0.46, +0.012 | −0.003 | +0.005 | 4.02, +0.022 |
| depth 6 | 3.3489 (0.0025) | 0.82, +0.009 | −0.004 | −0.004 | 1.89, **+0.004** |
| vocabulary 6,000 | 5.1214 (0.005) | 5.1226, **+0.001** | −0.003 | +0.008 | 5.1492, +0.028 |

The constant at 100x low loses 0.079 (main, 3.4321). From 100x low the rate reaches half the tuned value by step 100-120 in every setting (step 20-60 from 10x low); the loss comes from those slow early steps and is largest where the window is shortest (batch 16 × 450, window 113).

Takeaway: the recipe's 0.005 band covers starts from 10x low to 10x high (rounds 36-38) and extends to 100x low in two of four settings (losses at most 0.012 otherwise), but to 100x high in only one (losses up to 0.028). This confirms the start-low rule: when unsure, guess low.

## Round 40: an asymmetric rule, and why the recipe breaks at width 64, batch 4

Rule under test: *after the hold, letting the rule only lower s (driven by the same cosine) fixes 10-100x-high starts without hurting low starts.* Setting chosen to break the recipe in under 30 s: width 64, depth 2, batch 4, 3,000 steps (17-25 s per run). Constant sweep: 0.00018 3.3514, 0.0006 3.3243, 0.00125 3.3120, **0.0018 3.3101**, 0.0025 3.3129, 0.005 3.3253, 0.018 3.3638. Starts 0.01x-100x the tuned 0.0018. Gaps to 3.3101:

| Variant | 0.01x | 0.1x | 1x | 10x | 100x |
|---|---|---|---|---|---|
| safe recipe (gain 1.41, hold 450) | +0.002 | +0.001 | +0.014 | +0.034 | +0.029 |
| + after hold, decrease only (margin τ 0) | | +0.069 | +0.054 | +0.043 | +0.024 |
| + decrease only while c < −0.02 | | | +0.037 | | +0.013 |
| + decrease only while c < −0.05 | | +0.006 | +0.003 | +0.014 | +0.012 |
| + decrease only while c < −0.08 | | | +0.014 | | +0.029 |
| decreases 4x faster than increases during adaptation | | +0.087 | +0.084 | +0.062 | +0.035 |
| target −0.005 | | +0.030 | +0.054 | +0.062 | +0.073 |
| target +0.005 | | +0.028 | +0.002 | −0.004 | +0.005 |
| gain 2 | | +0.000 | +0.012 | +0.018 | +0.018 |
| gain 4 | +0.003 | −0.001 | +0.004 | +0.003 | +0.013 |
| gain 8 | | +0.002 | −0.001 | | +0.005 |
| Mechanic | | +0.008 | +0.005 | +0.006 | +0.007 |

Why the asymmetric versions fail: after the hold the mean cosine is between −0.004 and +0.001 at every held rate from 0.0012 to 0.0068 (0.7-3.8x tuned), while its per-step spread is 0.026. The signal cannot tell 1x from 4x here, so a decrease-only rule either never fires (τ 0.08), or fires on noise and becomes a fixed slow decay whose speed depends on τ and on the batch noise (τ 0-0.05: low starts lose up to 0.069). Scaling only the negative steps turns the same noise into a downward drift (every start collapses, +0.035 to +0.087). A target shift moves the zero but is a new constant to tune (+0.005 fixes high starts and breaks the low one).

Why the recipe breaks here: the round-38 fall-time rule predicts it. From the 10x start the rate peaks at 0.031 (step 60) and falls to 0.0068 by the hold (step 440): about 580 steps per tenfold, window 290, ratio 0.50. The held rate then depends on where the fall stopped: 0.0012 from 0.1x, 0.0041 from 1x, 0.0068 from 10x, 0.0054 from 100x (tuned 0.0018). A larger gain shortens the fall (gain 4-8 ties from 0.01x to 10x), but:
- the same boost breaks short runs: batch 16 × 450 with gain 5.66 loses +0.002 / +0.011 / +0.031 / +0.024 / +0.036 from 100x low to 100x high (peak 40x tuned, 113-step window); main with gain 5.66: −0.007 / −0.002 / +0.003 / +0.005 / +0.006;
- a gain law sqrt(width/32) × sqrt(steps/450), which fits these three settings, fails its first unseen test: at 118 tokens per weight (gain 1.49) the 1x start overshoots to 46x tuned and loses 0.017 (10x and 100x high: +0.006; recipe at 100x high: +0.004);
- no cheap noise measure separates the settings: the consecutive-gradient cosine at the tuned rate is 0.100 (main), 0.117 (batch 16), 0.116 (width 64, batch 4).

Takeaway: no asymmetric rule works on this signal, because after the hold it is flat over a fourfold range of rates and noisy; asymmetry converts noise into drift. Mechanic is steady (+0.005-0.008 from every start) but never within 0.005. From starts at or below the tuned rate the plain recipe ties (+0.001-0.002 from 0.01x and 0.1x). The working fix is round 42.

## Round 41: the whole model without a rate sweep: one multiplier or one per optimizer

Rule under test: *with the safe recipe, a second multiplier for the Adam-trained parts (signal from the embedding and head; norm gains follow) removes the Adam sweep; one multiplier shared by both optimizers (fixed Adam/Muon ratio) is simpler and may do as well.* Main setting, safe recipe on both, tuned 3.3534 with both rates swept (Muon 0.005, Adam 0.04). Starts (Muon, Adam):

| Starts (Muon, Adam) | (0.0004, 0.004) | (0.004, 0.04) | (0.04, 0.4) | (0.04, 0.004) | (0.0004, 0.4) |
|---|---|---|---|---|---|
| one shared multiplier | 3.3570 (+0.004) | 3.3546 (+0.001) | 3.3755 (+0.022) | 3.5742 (+0.221) | 3.4255 (+0.072) |
| one per optimizer | **3.3543 (+0.001)** | **3.3503 (−0.003)** | 3.3653 (+0.012) | **3.3578 (+0.004)** | 3.4129 (+0.060) |

Takeaway: one multiplier per optimizer wins from every start. A shared multiplier only works when the Adam/Muon ratio is already right; with Muon 10x high and Adam 10x low, Muon's falling multiplier drags Adam to 100x low (+0.221). Neither recovers from a 10x-high Adam rate: with Adam at 0.4 the Muon multiplier never rises (peak 0.00061 from 0.0004) and the run loses 0.060-0.072. Rule for the whole model: per-optimizer multipliers, Adam started at or below its guess (rounds 30-32 found the same with the rule run to the cooldown).

## Round 42: no fixed warmup; the multiplier's own rise is the warmup

Rule under test: *the fixed warmup is redundant: a multiplier started low rises at the rate the signal allows, which is a warmup whose length the signal sets; dropping the fixed ramp also gives the rule full gain from step 0 and lengthens the window before the hold by the warmup length.* `--warm 0` (no ramp on either optimizer), gain sqrt(width/32) from step 0, hold at min(30%, 450), cooldown kept. Gaps to the tuned constant with warmup:

| Setting | constant, no warmup | 100x low | 10x low | 1x | 10x high | 100x high |
|---|---|---|---|---|---|---|
| main | +0.004 | −0.003 | −0.001 | **−0.008** | 0.000 | +0.010 |
| batch 16 × 450 | −0.001 | +0.004 | −0.006 | −0.000 | −0.002 | +0.018 |
| width 64, batch 4 | +0.003 | | −0.001 | +0.005 | **+0.002** | **+0.006** |
| vocabulary 6,000 | +0.007 | | +0.010 | | −0.006 | |
| depth 8 (700 steps) | | | +0.002 | | +0.003 | |

With the fixed warmup the same starts gave: main −0.001 / −0.001 / −0.001 / +0.000 / +0.008; batch 16 × 450 100x low +0.012, 100x high +0.022; width 64 +0.001 / +0.014 / +0.034 / +0.029; vocabulary 6,000 −0.003 / +0.008; depth 8 +0.003 / +0.000. Loss-spike scores (largest rise of the smoothed training loss over its running minimum) equal the constant runs' in every setting (0.04, 0.10, 0.20), so the step-0 full-gain start causes no spike.

Takeaway: without the fixed warmup the recipe is within 0.005 from 10x low to 10x high (and 100x low) in 14 of 15 runs across five settings, and it repairs the width-64, batch-4 break (10x high: +0.034 → +0.002; 100x high: +0.029 → +0.006; held rates 0.0013-0.0023 from every start against 0.0012-0.0068 with warmup). The miss is vocabulary 6,000 from 10x low (+0.010, where the constant itself loses 0.007 without warmup: the 6,000-row embedding wants a warmup). 100x high still fails in runs of 450-900 steps (+0.010, +0.018).

## Round 43: Muon without warmup, Adam with warmup

Rule under test: *keeping Adam's warmup (for the embedding) and letting the multiplier be Muon's only warmup should fix the vocabulary-6,000 miss.* `--mwarm 0`.

| Setting | 10x low (0.1x at width 64) | 1x | 10x high | 100x high |
|---|---|---|---|---|
| vocabulary 6,000 | −0.001 | | +0.005 | |
| width 64, batch 4 | +0.018 | +0.028 | +0.018 | +0.022 |

Takeaway: it fixes vocabulary 6,000 and breaks width 64. With full gain on Muon while Adam is still ramping, Muon's rate peaks at 0.017-0.052 (9-29x tuned) by step 20-40 from the 0.1x-10x starts and is still 0.0037-0.0048 (2-2.7x tuned) at the hold, from every start. The warmups must match: either both fixed or neither. Recommendation: no fixed warmup on either, except with large vocabularies, where the embedding wants its own warmup and the fixed-warmup recipe is used.

## Round 44: the width-128 failure without warmup

Rule under test: *the no-warmup recipe also repairs the round-38 failure (width 128, batch 4, 3,000 steps, 10x-high start held at 3.6x tuned, +0.017).* One run (30.5 s), start 0.0125, tuned constant 0.00125: 3.2956.

| Version | val | gap | held rate / tuned | spike score |
|---|---|---|---|---|
| warmup recipe (round 38) | 3.3121 | +0.017 | 3.6 | 0.19 |
| **no-warmup recipe** | **3.3014** | **+0.006** | **1.4** | 0.19 |
| Mechanic (round 38) | 3.2985 | +0.003 | | |

Takeaway: without the fixed warmup the multiplier finishes its fall before the hold (held 1.4x tuned instead of 3.6x) and the gap drops from 0.017 to 0.006, of which about 0.003 is what a constant loses without warmup at width 64. Mechanic stays 0.003 ahead here.

## Round 45: a decrease-only correction after the hold, now without warmup

Rule under test: *with the fixed warmup gone (round 42), a post-hold rule that may only lower s while the cosine is below −τ closes the remaining high-start gaps without hurting low starts.* Same flags as round 42 plus `--post down --ptau τ`.

Width 64, batch 4, 3,000 steps (tuned 3.3101); τ dose:

| τ | 0.1x | 1x | 10x | 100x |
|---|---|---|---|---|
| none (round 42) | −0.001 | +0.005 | +0.002 | +0.006 |
| 0.02 | | +0.046 | | +0.019 |
| **0.05** | **+0.001** | **+0.001** | **−0.000** | **−0.000** |
| 0.08 | | +0.005 | | +0.007 |

τ = 0.05 elsewhere (gap to the tuned constant with warmup; no-warmup recipe without the correction in brackets):

| Setting | 100x low | 10x low | 1x | 10x high | 100x high |
|---|---|---|---|---|---|
| main | −0.000 (−0.003) | −0.003 (−0.001) | −0.009 (−0.008) | −0.006 (0.000) | **−0.001 (+0.010)** |
| batch 16 × 450 | +0.007 (+0.004) | −0.005 (−0.006) | +0.002 (−0.000) | −0.000 (−0.002) | +0.014 (+0.018) |
| depth 6 | | +0.005 | −0.008 | −0.002 (−0.001) | +0.003 (+0.006) |
| vocabulary 6,000 | | +0.007 (+0.010) | | −0.009 (−0.006) | |
| depth 8 | | +0.002 (+0.002) | | not run (load) | |

Loss-spike scores equal the constant runs' (0.04 / 0.09-0.10 / 0.19-0.22).

Takeaway: τ 0.05 ties from 0.1x to 100x at width 64 (spread 0.001 over a 1,000x range of starts) and repairs main's 100x-high start (+0.010 → −0.001); elsewhere it changes the no-warmup recipe by at most 0.006 either way. But τ is a knife edge: at width 64 the per-step spread of the cosine is 0.026, so τ 0.05 is about 2 standard deviations; τ 0.02 (under 1 sd) fires on noise and loses 0.019-0.046, and τ 0.08 (3 sd) never fires. τ must be set relative to the signal's own noise, so the correction stays optional and out of the recommended recipe until a noise-relative τ is tested. (Depth-8 high-start and width-128 runs were stopped by an app restart and machine load of 6-13.)

## Round 46: a decrease-only threshold set by the signal's own noise; Mechanic from 100x off

Rule under test: *round 45's τ 0.05 worked because it is about 2 standard deviations of the per-step cosine at width 64; a threshold of k × the running standard deviation (EMA 0.99 over the run, `--psd k`) with k = 2 should work in every setting with no constant to tune.*

| Setting, start | no correction | fixed τ 0.05 | k = 1 | **k = 2** | k = 3 |
|---|---|---|---|---|---|
| width 64, batch 4, 1x | +0.005 | +0.001 | +0.033 | **+0.003** | +0.006 |
| width 64, batch 4, 100x high | +0.006 | −0.000 | | **+0.001** | +0.007 |
| main, 1x | −0.008 | −0.009 | | **−0.009** | |
| main, 100x high | +0.010 | −0.001 | | **+0.005** | |

Takeaway: k = 2 ties in all four runs with no setting-specific constant. Below 2 the rule fires on noise and drifts down (k = 1: +0.033); at 3 it never fires and equals no correction, which is a safe failure. The correction is an optional add-on to the no-warmup recipe; it fixes main's 100x-high start (+0.010 → +0.005) and adds little elsewhere.

**Mechanic from 100x off (main setting, tuned 3.3534):** 0.00005 → 3.3809 (+0.028), 0.5 → 3.3650 (+0.012). With rounds 29 and 38, Mechanic's starts within 0.02 of the tuned constant span 0.1x-100x in main (1,000x), none at depth 8, 0.1x-1x at vocabulary 6,000; within 0.01 only the 0.1x start in main, 0.1x-1x at depth 6, none at depth 8 or vocabulary 6,000.

## Round 47: transfer to widths 24, 48, 96 and depths 3, 5 (never fitted)

Rule under test: *the no-warmup recipe (gain sqrt(width/32) from step 0, hold at min(30%, 450)), fitted on widths 8-128 at depth 2 and depths 4-8 at width 32, ties the swept constant on widths and depths it never saw, from starts set by the width-32 guess (0.0005 / 0.005 / 0.05), with no retuning.* Main configuration otherwise (batch 8, 900 steps, vocabulary 1,000). Constants swept with warmup over 3-4 rates around the best one.

| Setting | constant sweep (rate: val) | tuned | recipe from 0.0005 / 0.005 / 0.05 | gaps |
|---|---|---|---|---|
| width 24, depth 4 | 0.0025: 3.3684, **0.005: 3.3644**, 0.01: 3.3696 | 3.3644 | 3.3695 / 3.3633 / 3.3649 | +0.005 / −0.001 / +0.000 |
| width 48, depth 4 | 0.00125: 3.3366, **0.0025: 3.3255**, 0.005: 3.3315, 0.01: 3.3452 | 3.3255 | 3.3292 / 3.3218 / 3.3235 | +0.004 / **−0.004** / −0.002 |
| width 96, depth 4 | 0.00125: 3.3050, **0.0025: 3.3015**, 0.005: 3.3069 | 3.3015 | 3.2991 / 3.2952 / 3.2977 | −0.002 / **−0.006** / −0.004 |
| width 32, depth 3 | 0.00125: 3.3707, **0.0025: 3.3533**, 0.005: 3.3591, 0.01: 3.3676 | 3.3533 | 3.3597 / 3.3567 / 3.3523 | +0.006 / +0.003 / −0.001 |
| width 32, depth 5 | 0.0025: 3.3517, **0.005: 3.3471**, 0.01: 3.3658 | 3.3471 | 3.3455 / 3.3456 / 3.3477 | −0.002 / −0.002 / +0.001 |

Takeaway: 15 of 15 recipe runs within 0.006 of the swept constant and 13 within 0.005, with the start up to 20x off the setting's tuned rate (width 96: tuned 0.0025, start 0.05). The width-96 recipe beats its tuned constant from all three starts (−0.002 to −0.006). The two loosest runs are the 0.1x-0.2x low starts at depth 3 and width 24 (+0.006, +0.005), consistent with a low start spending its first steps too small. The sqrt(width/32) gain and the min(30%, 450) hold transfer to these five settings.

## Cost: one recipe run against a sweep

Compute per method, in units of one constant-rate training run, and the worst gap to the swept constant over starts from 10x low to 10x high in the settings where each was run.

| Method | runs needed | time per run vs constant (median wall time, matched settings) | total compute | worst gap from 10x low to 10x high |
|---|---|---|---|---|
| swept constant, 7 rates 2.15x apart over 100x | 7 | 1.00 | 7.0 | 0 by definition (grid resolution about ±0.003) |
| swept constant, 5 rates 3.16x apart over 100x | 5 | 1.00 | 5.0 | up to +0.008 (main: the grid can land 1.8x off) |
| one constant run at a guess 10x off | 1 | 1.00 | 1.0 | +0.031 to +0.071 |
| **no-warmup recipe** | **1** | **1.05-1.10** | **about 1.1** | **+0.006** (13 settings; vocabulary 6,000 from 10x low +0.010) |
| Mechanic | 1 | 0.9-2.2 | 0.9-2.2 | +0.039 (depth 8, 10x high) |
| D-Adaptation | 1 | 1.1-3.0 | 1.1-3.0 | +0.042 |
| Prodigy | 1 | 1.1-2.3 | 1.1-2.3 | +0.042 |

The rule adds one inner product and two norms per Muon matrix per step (3mn multiply-adds for an m × n matrix), against about 15mn·min(m, n) for five Newton-Schulz iterations: under 1% of Muon's optimizer cost at min(m, n) ≥ 16, and far less than the forward and backward passes. Wall-time ratios are medians over the runs of each kind in the same setting (rule / constant: width 64 batch 4 1.05, width 128 batch 4 1.07, depth 6 1.10, vocabulary 6,000 1.10), measured while sibling sessions loaded the machine, so they bound the cost from above; our D-Adaptation, Prodigy and Mechanic are unoptimized Python. Takeaway: the recipe reaches the swept loss for about one-sixth of the compute of a 7-rate sweep.

## Round 48: Mechanic on the five transfer settings; the 12-setting comparison

Rule under test: *the recipe beats Mechanic, the strongest published competitor that can lower its scale, by at least 0.01 mean loss over starts 10x low / 1x / 10x high in at least 3 of every 4 settings.* Mechanic with its default fixed warmup, same starts as round 47.

| Setting | tuned constant | Mechanic from 0.0005 / 0.005 / 0.05 | gaps | recipe gaps (round 47) |
|---|---|---|---|---|
| width 24 | 3.3644 | 3.3846 / 3.3920 / 3.3977 | +0.020 / +0.028 / +0.033 | +0.005 / −0.001 / +0.000 |
| width 48 | 3.3255 | 3.3331 / 3.3337 / 3.3386 | +0.008 / +0.008 / +0.013 | +0.004 / −0.004 / −0.002 |
| width 96 | 3.3015 | 3.3112 / 3.3075 / 3.3096 | +0.010 / +0.006 / +0.008 | −0.002 / −0.006 / −0.004 |
| depth 3 | 3.3533 | 3.3664 / 3.3749 / 3.3782 | +0.013 / +0.022 / +0.025 | +0.006 / +0.003 / −0.001 |
| depth 5 | 3.3471 | 3.3559 / 3.3630 / 3.3670 | +0.009 / +0.016 / +0.020 | −0.002 / −0.002 / +0.001 |

All 12 settings with Mechanic, mean gap over the three starts (recipe: the warmup-free version where it was run from all three starts; the fixed-warmup safe recipe at depths 6 and 8 and vocabulary 6,000; at width 128 the warmup recipe from low starts and the warmup-free one from the high start):

| Setting | steps | constant at a 10x-wrong rate (mean of both sides) | Mechanic | recipe | recipe better than Mechanic by |
|---|---|---|---|---|---|
| main | 900 | +0.049 | +0.012 | −0.003 | **0.015** |
| depth 3 | 900 | (not swept to 10x) | +0.020 | +0.003 | **0.017** |
| depth 5 | 900 | | +0.015 | −0.001 | **0.016** |
| depth 6 | 900 | +0.056 | +0.012 | −0.003 | **0.015** |
| depth 8 | 700 | +0.052 | +0.032 | +0.000 | **0.032** |
| width 24 | 900 | | +0.027 | +0.001 | **0.026** |
| width 48 | 900 | | +0.010 | −0.001 | **0.010** |
| width 96 | 900 | | +0.008 | −0.004 | **0.012** |
| vocabulary 6,000 | 900 | +0.069 | +0.015 | +0.002 | **0.012** |
| 118 tokens per weight | 4,000 | +0.034 | +0.003 | +0.002 | 0.002 |
| width 64, batch 4 | 3,000 | +0.048 | +0.006 | +0.002 | 0.004 |
| width 128, batch 4 | 3,000 | (+0.019 at 4x high) | +0.004 | +0.002 | 0.002 |

Takeaway: the recipe beats Mechanic by 0.010-0.032 mean loss in 9 of 12 settings and ties in the other 3; it never loses by more than 0.003 at any start. The 3 ties are exactly the 3,000-4,000-step runs: Mechanic keeps adapting for the whole run and needs about that long to settle, so the recipe's advantage over Mechanic is in runs of about 1,000 steps or fewer, while its advantage over a constant at a wrong guess (0.03-0.07) holds at every length. Mechanic's own gap grows with depth (+0.020 / +0.012 / +0.015 / +0.012 / +0.032 at depths 3-8) and shrinks with width (+0.027 / +0.012 / +0.010 / +0.008 at widths 24-96, depth 4), while the recipe stays within 0.006 along both.

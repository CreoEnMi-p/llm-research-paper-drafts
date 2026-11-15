# Literature: Muon that sets its own step size

Checked 2026-11-15 with extended web searches. Question: has anyone given each Muon weight matrix its own step size that grows when consecutive updates agree and shrinks when they flip?

Haiku check (2026-11-15): one Haiku agent listed the 5 closest published papers, and I read each abstract. (1) Baydin et al. 2018, hypergradient descent (arXiv 1703.04782): one global rate moved by the dot product of consecutive gradients, for SGD or Adam, all run. (2) GALA, Jiang/Kavis/Mokhtari 2025 (arXiv 2506.08419): rate from alignment of consecutive gradients plus a curvature estimate, for normalized SGD and Adam; no Muon, no hold. (3) AdamX, Caldas/Belo/Soares 2026 (arXiv 2609.11867): cosine similarity as a multiplier on an Adam-style update size; the abstract does not say which vectors, and Muon is not mentioned. (4) AdaGO, Zhang/Liu/Schaeffer 2025 (arXiv 2509.02981): Muon rate from summed gradient norms (AdaGrad-style). (5) Distance-Aware Muon, Demidovich/Chakraborty/Malinovsky/Nedić/Richtárik 2026 (arXiv 2605.18999): Muon step radius from distance travelled from the start, a descent certificate, or a 1-D search; no gradient-update cosine. Near misses: AdaMuon (2507.11005), Prodigy (2306.06101). Verdict: kept. No paper sets Muon's rate from the cosine of the new gradient with the last orthogonalized update and then holds it.

Correction (2026-11-15, after the prior-work check): partly published. Baydin et al. 2018, Eq. 8, the "multiplicative rule of HD", already uses the cosine of the gradient with the last update and calls it invariant up to rescaling (checked in the PDF). Funnel (Amid, Anil, Fifty, Warmuth 2022, arXiv 2202.00145), Eq. 12, is s ← s·exp(γ_s·cos(g_t, ν_t)), our exact update, and says the cosine lets one γ_s serve every layer size (checked in the PDF). MARTHE (Donini et al. 2020, arXiv 1910.08525, IJCAI) addresses greedy hypergradient schedules alongside Wu et al. 2018. New framing: the paper applies multiplicative HD / Funnel to Muon's orthogonalized update; it claims only the Muon result (replaces the sweep), adapt-then-hold with the measured drift, the sqrt(width/32) gain and the weight-decay requirement, and the measured comparisons plus the failed consecutive-update signal.

## Sources

| Work | What it does | Relation to this paper |
|---|---|---|
| Jacobs 1988, delta-bar-delta | Per-weight rate; add a constant when the gradient sign agrees with a running average, multiply down when it flips. | The rule's ancestor. Per scalar weight, plain gradient descent, no orthogonalized updates. |
| Riedmiller & Braun 1993, Rprop | Per-weight step from sign agreement of consecutive gradients, multiplicative 1.2 / 0.5, bounds. | Same sign-agreement idea per scalar; full-batch; the magnitude is set only by the step. |
| Schraudolph 1999, SMD; Baydin et al. 2018, hypergradient descent | Learning rate updated by the dot product of the current gradient and the previous update. | Hypergradient is the additive form of our rule on raw gradients for one global rate (or per parameter). Not applied to Muon, not per matrix with a scale-free signal. |
| Jiang, Kavis, Mokhtari 2025, GALA (arXiv 2506.08419) | One global rate for normalized SGD from consecutive-gradient alignment plus a curvature estimate, chosen by follow-the-regularized-leader. | Closest in spirit. Global rate, normalized SGD, needs curvature estimate; not Muon, not per matrix. |
| Defazio & Mishchenko 2023, D-Adaptation; Mishchenko & Defazio 2024, Prodigy | Estimate distance to solution D to set the rate of Adam / SGD. | Learning-rate-free, global; a 2025 survey notes any norm could be plugged in but no Muon version was found. |
| Ivgi et al. 2023, DoG | Rate = max distance from init / root of summed squared gradient norms. | Learning-rate-free, global or per layer for SGD. |
| Cutkosky, Defazio, Mehta 2023, Mechanic | Wraps any optimizer; one scalar scale from a 1-D online learner fed by gradient times displacement. | Could wrap Muon; one global scale, not per matrix; not tested on Muon in the paper. |
| Defazio et al. 2024, schedule-free; ScheduleFree+ 2026 (arXiv 2605.19095) | Removes the schedule; ScheduleFree+ adds a Polyak learning-rate-free variant for LLMs. | Removes the schedule, keeps or replaces a global rate. |
| AMUSE 2026 (arXiv 2605.22432) | Muon with schedule-free averaging; mild sensitivity across swept settings. | Lowers sensitivity, still has a global rate to set. |
| Zhang, Liu, Schaeffer 2025, AdaGO (arXiv 2509.02981) | AdaGrad-norm step size times Muon's orthogonal direction. | Per-matrix step from accumulated gradient norms (always shrinking), not from alignment. |
| Hao et al. 2025, LANTON (arXiv 2510.14009) | Per-layer rates for Muon/Scion from estimated gradient noise in the dual norm. | Per-layer Muon rates, but from noise variance, not update agreement; still has a base rate to tune. |
| You et al. 2017-2019, LARS / LAMB | Per-layer trust ratio = weight norm / update norm. | Per-layer, set by norms, not by agreement over time. |
| Liu et al. 2025, Moonlight ("Muon is scalable") | Weight decay plus matching Muon's update RMS to AdamW (0.2·sqrt(max dim)) so AdamW's rate can be reused. | Removes the Muon-specific sweep only if the AdamW rate is already tuned. |
| AdaMuon, NorMuon, MUON² (2025-2026) | Second-moment or per-neuron normalization on top of the orthogonalized update. | Change the update's shape, keep a global rate. |
| MuonClip (Kimi K2, 2025) | Rescales query/key weights when attention logits grow. | Stability fix, not a step-size rule. |
| "Muon sublates the edge of stability" 2026 (arXiv 2609.34915) | Logs per-matrix cosine of consecutive Muon updates across peak rates on a 130M LLM; key/value matrices go more negative. | Uses our signal as a diagnostic only, no controller. Supports the premise that the cosine tracks the rate. |
| Wu, Ren, Liao, Grosse 2018, "Understanding short-horizon bias in stochastic meta-optimization" | Greedy (one-step) learning-rate adaptation picks rates that are too small, because the next-step loss pays for noise that later steps remove. | Explains why every alignment signal here crosses zero below the best constant rate, and why the rule loses in few-step runs. |
| Pflug 1983-1990; Chee & Toulis 2018; Lang, Xiao, Zhang 2019 (SASA) | Detect when SGD reaches its noise-dominated phase from the sign of consecutive-gradient inner products, then cut the rate. | The same statistic, used as a step-down trigger for SGD; not a two-way multiplicative controller, not Muon. |
| Kosson, Messmer, Jaggi 2024, "Rotational equilibrium" | With weight decay, the angular update of normalized weights settles at a value set by lr times weight decay. | Why weight decay makes the signal fall monotonically with the rate (round 6). |
| Dehghani et al. 2023 (ViT-22B); Wortsman et al. 2023 | QK-norm keeps attention logits under control and lowers sensitivity to the learning rate. | Matches round 24: QK-norm does the job of a 16x query/key rate. |
| SignMuon 2026 | Sign compression for distributed Muon; logs sign agreement. | Unrelated mechanism. |

## Done and not done

| Question | Done? |
|---|---|
| Per-scalar rate from sign agreement of consecutive gradients (plain SGD) | Done (delta-bar-delta, Rprop) |
| One global rate from gradient alignment (hypergradient, GALA) | Done |
| Learning-rate-free global rate for Adam/SGD (D-Adaptation, Prodigy, DoG, Mechanic) | Done |
| Per-matrix Muon rate from noise (LANTON) or accumulated norms (AdaGO) | Done |
| Cosine of consecutive Muon updates as a diagnostic of the rate | Done (edge-of-stability paper) |
| **Per-matrix Muon step size driven by the cosine of consecutive updates** | **Not found** |
| **Does such a rule recover the tuned Muon rate from 0.1x-10x wrong starts, across width and depth** | **Not found** |
| **Which signal works for Muon: orthogonalized update, raw gradient or momentum** | **Not found** |
| **Which matrices (Q, K, V, O, MLP, depth) end at which rates, and why** | **Not found** |

The exact idea was not found. The closest are hypergradient descent and GALA (global rate from alignment) and LANTON (per-layer Muon rates from noise). This paper fills the bold rows.

## Note on the signal

Muon's orthogonalized update has a fixed size (all singular values near 1), so the cosine of two consecutive updates depends only on direction. That makes it a scale-free signal: the same target value can be used for every matrix shape and width. With momentum 0.95, consecutive momentum vectors are about 95% shared by construction, so the cosine of the momentum-based update is high at any rate; the raw gradient cosine does not have this offset. The rounds test which signal works.

Sources: arxiv.org/abs/2506.08419, arxiv.org/abs/2510.14009, arxiv.org/abs/2509.02981, arxiv.org/abs/2605.22432, arxiv.org/abs/2605.19095, arxiv.org/abs/2306.00144, arxiv.org/abs/2306.06101, arxiv.org/abs/2502.16982, arxiv.org/html/2609.34915.

# Research 03 — Autoregressive exposure falsification (2026-11-15)

**Status:** Completed independent three-seed *pilot*. Valid for stopping the current standalone previous-read architecture claim under this fixed protocol; **not** a statistical proof that all read-aware write controls fail.

- Independent CI evidence: [Actions run 43803573256](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43803573256), three successful seed-specific jobs with line-by-line per-query trajectories.
- Stacked review PR: [#3](https://github.com/CreoEnMi-p/llm-research-paper-drafts/pull/3), based on Gemini audit PR #2, not main.
- No duplicate of Gemini's chunk correctness / C64 / mqar2 / P4 / hop2 / D/M / compute-budget experiments.
- **Frozen trained checkpoint** intervention vs **trained Gate** effect are separately reported. Do not conflate them.

## Why this falsifier

Existing tests score MQAR query answers under teacher forcing: prior answer tokens, including the model's previously mispredicted answers, are supplied correctly by the dataset. An autoregressive system consumes its own outputs. The effect of write suppression on subsequent retrieval may reverse with exposure to generated answers.

## Precommitted experiment

- Train same synthetic 2-phase MQAR problem with fixed 300 steps, B32, L64, pairs16, width32, layers2, heads2, CPU, lr0.05 and elr0.04, 3 paired train/init seeds (0,1,2); evaluation uses the same fixed seed-12345 heldout examples.
- Compare g no extra gate; self-consistent sequential previous-read gate (lpsI1/lps0.25); self-consistent current-key gate (lpsR2); softmax as **diagnostic** only (not tuned, not a strong baseline).
- For 96 fixed validation rows and 16 answer slots each, recompute the full causal prefix before every answer; insert the *model's own argmax token* instead of the ground-truth answer; leave all original facts and query keys unchanged. Also report ordinary teacher-forced accuracy on exactly the same 96 rows.
- At frozen-checkpoint evaluation only, test the task-aware answer-position write suppression oracle (g-family; no retraining), in both teacher-forced and self-generated rollout.
- **Precommitted stop criterion:** if previous-read gate does not beat ordinary g autoregressive accuracy on at least 2/3 paired seeds, stop the general online-retrieval improvement claim for this configuration.

## Verified results: self-generated autoregressive answer accuracy, 96 rows × 16 queries

| variant | seed0 | seed1 | seed2 | mean |
|---|---:|---:|---:|---:|
| g baseline | 0.849609 | 0.867839 | 0.847656 | **0.855035** |
| g + exact previous-read gate | 0.939453 | 0.839193 | 0.789714 | **0.856120** |
| g + exact current-key gate | 0.919922 | 0.916016 | 0.811198 | **0.882379** |
| frozen g baseline + answer-write suppression oracle | 0.921224 | 0.936198 | 0.938151 | **0.931858** |
| untuned softmax (failed to learn task) | 0.065755 | 0.053385 | 0.076823 | 0.065321 |

Paired previous-read-minus-g differences: **+8.9844, −2.8646, −5.7942 percentage points**, seed 0,1,2. Paired current-key-minus-previous-read differences: **−1.9531, +7.6823, +2.1484 percentage points**.

Additional sanity:
- The first answer's prediction equals teacher-forced evaluation exactly by assertion, because no previous answer has been generated.
- At every later step, only answer positions are replaced; all facts and query-key positions remain verbatim correct (asserted).
- The oracle increases g baseline autoregressive accuracy on **3/3** seeds. Its task-aware phase labeling is privileged and is *not* a deployable learned controller or a controlled official GDN-2 comparison.
- All three GitHub jobs succeeded. The 300-step softmax baseline achieves only 5–8% autoregressive accuracy; **exclude** it as a strong baseline. This training recipe is not suitable for that model.

## Conclusions with epistemic status

**Observed:** Answer-position write suppression improves frozen ordinary g autoregressive accuracy consistently (~85.5% to ~93.2% aggregate mean). The trained exact previous-read gate does *not* consistently improve autoregressive retrieval (only 1/3 seeds) and its 3-seed mean is ~0.1 percentage points better than g. Current-key gate is better than previous-read in 2/3 seeds, without decisive superiority across more seeds.

**Mechanistic interpretation (hypothesis, narrower than fact):** Correct answer tokens written into recurrent state can interfere with subsequent retrieval; suppressing such updates can help even with exposure to wrong generated tokens. The failure of previous-read training can reflect an unreliable proxy and/or optimization sensitivity; this study does not isolate which.

**Unproven conjectures:** All wider domains, all learning-rate settings, all larger models, natural language, fast chunk kernels, actual memory-lifetime management, and true GDN-2/PH comparisons. Three seeds permit a falsification-oriented early decision, not a tight effect-size CI.

**Compute:** Sequential exact reference is not optimized; training ~24–39 s / 300 steps vs ~7–11 s for g on these runs, so equal-token training comparisons do NOT establish an efficient architecture. Gemini subagent is separately evaluating compute-matched baselines. Do not claim free compute or kernel equivalence.

## Strategic direction for PI meeting

- **Keep the narrower causal mechanism** as a reproducible observation: answer-update interference persists when the model consumes its own generated answers. This finding is interesting but the conceptual issue overlaps existing pretrained causal work (How Linear Attention Remembers, arXiv:2609.33093).
- **Stop independent development of the present previous-read gate** until it beats strong already-existing methods, content/position controls and equal-compute comparisons; the precommitted online behavior criterion failed.
- The next genuinely decisive question is not whether a different margin can overfit MQAR: **does a mature Gated DeltaNet-2 or PH/associability-style gate already suppress harmful answer writes at equal model and training budget?** If yes, this paper's novel method hypothesis is dead. Gemini PR #2 provides an exact reference but does NOT deliver an optimized kernel.
- Separate second missing falsifier is **same-value new-key association and overwriting old associations**. Gemini mqar2 only introduces disjoint fresh keys after queries; it does not test overwrite or alias. Demand ability to learn necessary new writes as well as preserve old recalls before any claims about general memory write control.
- Present to professor: the mechanistic observation, the original oracle bug, the chunk-partition inconsistency, and this autoregressive robustness failure, with both PR links and clear unproven parts. This is enough for a **go/no-go conversation**, not a preprint.

No main-branch changes in this PR.

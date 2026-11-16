# Paper 03: random read/write stream — one-baseline competence gate

Date: 2026-11-15 (PST). **Design preregistered before first training result.**

## Why not another GDN-2-like toy baseline?
- NVIDIA's [official implementation](https://github.com/NVlabs/GatedDeltaNet-2) requires CUDA, Triton, compiled Flash Linear Attention and a specific model and kernel recipe (see requirements and implementation). This chat runtime has CPU PyTorch only; the repo's standard ubuntu-latest Actions runners also have no GPU. GitHub larger GPU runners require an eligible org/enterprise configuration. We cannot honestly report a head-to-head with official GDN-2 without access to that environment.
- The previous equation-only CPU variant failed the basic learnability test on two seeds. That variant is **withdrawn as a strong baseline**; we will NOT sweep optimizer settings over the test split or invent GPU results.
- The published GDN-2 paper and official README already provide genuine *external* RULER and language-model benchmark results. They are important prior-art evidence, not scores on our new data.

## New task: genuinely mixed events
- Each event contains three tokens: `WRITE k v` updates a dictionary and `READ k v` requires predicting `v` after seeing only `READ k`.
- The first four events initialize four unique key/value facts. The remaining 28 event types are shuffled *independently for every sequence*, with six later writes (two new keys and four overwrites) and 22 read queries. Unlike old mqar2, there is no dedicated second write phase and no fixed query phase.
- Updated keys are required to return *latest* value, unchanged keys must retain their old value, and new keys may reuse an old value (same-value alias).
- Marker, key and value identities occupy disjoint vocabulary partitions. Replaying all visible writes is a deterministic 100%-accurate oracle; a unit test verifies all answers against the latest visible write and that no query label precedes its key.
- L=96 input tokens, V=256. No last-token leakage; only read-key positions contribute to training loss. Training uses full ground-truth inputs; **separate fresh answer-only autoregressive evaluation** uses model-produced previous answers and preserves externally supplied write facts.
- Test categories are **old untouched**, **after overwrite**, **new key**. Count each separately, including stale old-value predictions. Classifier uses only input prefix and checks every label.

## Single-seed learnability pilot — only experiment initially authorized
- First run only standard Delta Rule (`D`), seed0, width64, 2 layers, 4 heads, 24 examples per batch, Muon lr 0.05 / Adam lr0.04, init0.5, 1800 max steps. Exactly one CPU job.
- Every 100 updates, evaluate a *fixed* 512-row validation split (seed12345) and the three category accuracies.
- **Stop early on SUCCESS:** overall accuracy >=0.82 AND each category >=0.75, each category with >=100 answers, for **two successive evaluations**.
- **Stop as INVALID if not passed by 1800 updates.** Do not train any Gate or another seed in an unlearned setting. At most one predeclared easier diagnostic follow-up (N=3, four later writes incl one new key, same marker/value rules) may be run to establish task learnability; if that fails, abandon this testbed design.
- After PASS, evaluate 96 entirely fresh rows (seed92742) with greedy answer-only AR rollout. If AR retrieval collapses, suspend before full matrix.
- **No multi-seed architecture experiment, no margin search, and no claimed breakthrough** before this gate passes.

## Future official GDN-2 acceptance criteria (NOT fulfilled yet)
1. Run untouched NVIDIA official module/kernel with CUDA-capable GPU and published software environment; first prove forward recurrent==chunk numerical behavior on a deterministic short stream before any training.
2. Confirm official model learns *all three query groups* and initial facts under an appropriate training setup; train only after smoke test and validation, and compare with shared task/token budget, parameter counts, and hardware time.
3. Compare against a matched previous-read and **current-key** gate, and ordinary Delta Rule, on a heldout untouched randomized stream. If official strong baseline already closes advantage, stop separate PS gate architecture.
4. Do not label a mathematical imitation or failed-to-learn implementation as the official strong baseline.

All changes remain on a draft research branch. `main` unchanged.

# 03: second learnability pilot — higher-memory interleaved stream

**Why:** the precommitted easy mixed-stream Delta benchmark learned to 98.0% teacher-forced in 300 steps, and 96.45% fresh self-generated-answer accuracy (96 rows, 2112 scored answers). It is too close to ceiling to justify six more architectures or multiple random seeds. [Verified successful run #43832170838](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43832170838).

**Design change made transparently after easy pilot:** increase active keys from 6 to 12; 8 initial facts + 4 later new facts; 10 late writes (4 new, 6 genuine overwrites) and 30 randomly interleaved reads; 48 records, L=144. The same two marker IDs and disjoint value range, identical independent oracle replay, and identical per-group success criteria apply.

**One CPU job only** with unchanged standard Delta Rule model (width64, 2 layers/4heads, 24 examples/batch, same LR). At most 1600 steps; stop early if all queries >=82%, untouched/overwritten/new >=75% each (n>=100), on two consecutive checks separated by 100 steps.

**Hard stop:** If learning gate fails or fresh heldout answer-only autoregressive performance collapses, do NOT train prediction gates, no additional seeds, and do not salvage the baseline by tuning to the test set. Success licenses a later *one-seed* controlled Gate/Delta comparison on the **same** harder data, not a conference-level claim. No GPU official GDN2 run is attempted here because the current workspace and default GitHub runner cannot execute official CUDA/Triton stack.

The original frozen test split seed12345 is used only for early stopping; a *separate* fresh heldout seed92742 is evaluated on model-generated query answers after the stopping criterion passes.

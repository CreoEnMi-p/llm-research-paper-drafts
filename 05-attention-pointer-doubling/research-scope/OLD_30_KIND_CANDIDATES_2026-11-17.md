# Per-category candidate counts and source-only heuristic controls

**Protocol:** 128 CPython-executed programs per category. Old template data are exploratory and repeatedly inspected. The 16-way uniform-label prior (6.25%) does **not** represent the output-constrained pointer candidate set. Candidate-conditioned uniform is 1/U per program, where U is the number of different integer literals in the source. These are not universal lower or upper bounds. Five-seed models were only trained/evaluated for the first 20 categories; dashes mark unavailable predictions, not zero scores.

| Category | U literals | Conditional 1/U | Last source integer | Source modal integer | Transformer (5-seed) | Pointer + indent (5-seed) |
|---|---:|---:|---:|---:|---:|---:|
| direct_last | 2 | 50.0% | 100% | 52.3% | 100% | 100% |
| copy_snapshot | 2 | 50.0% | 0% | 50.8% | 100% | 100% |
| copy_chain | 2 | 50.0% | 0% | 44.5% | 100% | 100% |
| local_global | 2 | 50.0% | 0% | 51.6% | 57.3% | 67.5% |
| local_result | 2 | 50.0% | 100% | 53.1% | 45.6% | 68.0% |
| global_mutation | 2 | 50.0% | 100% | 57.0% | 100% | 100% |
| closure_late | 2 | 50.0% | 100% | 47.7% | 100% | 100% |
| closure_snapshot | 2 | 50.0% | 0% | 49.2% | 100% | 100% |
| param_global | 2 | 50.0% | 0% | 53.9% | 53.4% | 62.2% |
| param_result | 2 | 50.0% | 100% | 52.3% | 48.3% | 66.7% |
| nested_local | 2 | 50.0% | 100% | 50.8% | 49.2% | 85.9% |
| nested_global | 2 | 50.0% | 0% | 50.0% | 52.3% | 35.2% |
| nonlocal_mutation | 2 | 50.0% | 100% | 48.4% | 7.8% | 54.1% |
| nonlocal_untouched | 2 | 50.0% | 0% | 50.0% | 2.8% | 95.0% |
| closure_first | 2 | 50.0% | 0% | 56.2% | 23.4% | 50.3% |
| closure_second | 2 | 50.0% | 100% | 45.3% | 45.0% | 49.5% |
| long_reassign | 8 | 12.5% | 100% | 14.8% | 1.9% | 0% |
| long_alias | 3 | 33.3% | 0% | 41.4% | 94.2% | 71.2% |
| nested3 | 3 | 33.3% | 100% | 33.6% | 1.1% | 74.8% |
| global_from_nested | 2 | 50.0% | 100% | 46.1% | 1.4% | 60.3% |
| sibling_fun | 3 | 33.3% | 0% | 28.9% | — | — |
| sibling_outer | 3 | 33.3% | 100% | 29.7% | — | — |
| sibling_closure_fun | 2 | 50.0% | 0% | 53.1% | — | — |
| sibling_closure_outer | 2 | 50.0% | 100% | 48.4% | — | — |
| sibling_global | 3 | 33.3% | 0% | 35.9% | — | — |
| sibling_shadow_global | 3 | 33.3% | 100% | 21.9% | — | — |
| closure_many_first | 6 | 16.7% | 0% | 11.7% | — | — |
| closure_many_last | 6 | 16.7% | 100% | 15.6% | — | — |
| nested_late | 2 | 50.0% | 100% | 46.9% | — | — |
| nested_snapshot | 2 | 50.0% | 0% | 49.2% | — | — |

### Corrected aggregate reference
- 12 training-type families: mean U=2, candidate chance **50.0%**, last literal **50.0%**.
- 8 old OOD families: mean of per-program 1/U **41.15%**, last literal **62.50%**, fixed modal literal **41.99%**.
- 10 hard families: mean per-program 1/U **36.67%**, last literal **50.0%**, modal literal **34.14%**.
- Original mean 56.91% for pointer+depth over old 8 OOD does **not** beat the 62.5% last literal. The last-literal selection was observed on the exploratory test, not preregistered.
- Other script baselines available in the zipped original `old_30kinds_corrected.csv`: first literal, last constant assignment, scope-blind same-name most recent assignment. No guessing that all kinds have two candidates.
- Frozen new A+C split: **every example has U=4 and exactly four literal occurrences**; conditional source-literal prior **25%** per question. A pair has two distinct correct labels; independent-uniform pair both correct chance **6.25%**; uniformly drawing an ordered *distinct* pair chance **8.33%**. A location-unaware first-literal source heuristic gets **37.30% single/0% both**, and neural pointer+indent gets **37.89% single/0.08% both** (five seeds).
- Symbolic [restricted AST execution baseline](ast_reference.py) gets 100% local on 600 random AB + 600 random CD + 600 random AC plus 256 frozen A+C pairs. A separate [CI verification](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44231498354) passes 10 CPython lexical-scope regression examples; frozen JSONL was not committed in this PR, so these 512 tests are **local**, not CI reproduced.

The complete original source tables, raw per-example metadata, frozen test hash, and five-seed predictions are retained in downloadable `PBS_REVIEWER_REVISION_REPRO_WITH_AST_20261118.zip`, with the V2 frozen test SHA-256 unchanged.

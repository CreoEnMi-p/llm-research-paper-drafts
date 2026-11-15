#!/bin/sh
# Headline in under 2 minutes on an idle laptop (Apple GPU): vocabulary 6,000, 400 steps.
# competitor = head-only Adam epsilon 1e-5; ours = cluster-shared head + the same epsilon.
# Expected: competitor 4.9931, cluster-shared head 4.9720 (gain 0.021).
cd "$(dirname "$0")"
nice -n 20 python3 grid.py --V 6000 --heps 1e-5 --tag repro_competitor --out repro.jsonl
nice -n 20 python3 grid.py --V 6000 --heps 1e-5 --clus 64 --tag repro_cluster_rows --out repro.jsonl

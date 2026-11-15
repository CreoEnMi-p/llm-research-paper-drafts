#!/bin/bash
# Headline in about 1-2 minutes on a laptop CPU: MQAR, 16 pairs, context 64, 2 layers, width 32, 300 steps, Muon lr 0.06.
# Expected (runs are chaotic; differences under ~0.1 are noise): gated linear attention ~0.38, with predicted-skip ~0.15.
cd "$(dirname "$0")"
F="--width 32 --layers 2 --heads 2 --L 64 --batch 32 --steps 300 --pairs 16 --lpsX 1 --lpsC 16 --dev cpu --threads 2 --task mqar --lr 0.06 --elr 0.04 --init 0.5 --alibi 0 --sink 0 --mix g"
nice -n 20 python3 rc.py $F --tag repro-g | grep -E "^repro"
nice -n 20 python3 rc.py $F --lps 0.25 --tag repro-gps | grep -E "^repro|^gate"

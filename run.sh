#! /bin/bash
for logdir in logs/lvl{1,2,3,4,5,6}-QMix*; do
    echo "$logdir"
    nohup python3 src/main.py new run "$logdir" --n_runs=2 --n_tests=1 &
    sleep 5
done
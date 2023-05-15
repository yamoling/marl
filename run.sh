#! /bin/bash
for logdir in logs/*alternating*; do
    echo "$logdir"
    nohup python3 src/main.py new run "$logdir" --n_runs=3 --n_tests=1 &
    sleep 5
done
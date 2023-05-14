#! /bin/bash
for logdir in logs/StaticLaserEnv*lvl1*; do
    echo "$logdir"
    nohup python3 src/main.py new run --logdir "$logdir" --n_runs=3 --n_tests=1 &
    sleep 5
done
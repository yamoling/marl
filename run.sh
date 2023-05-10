#! /bin/bash
for logdir in logs/StaticLaserEnv*LinearVDN*Prioritized*; do
    echo "$logdir"
    nohup python3 src/main.py new run --logdir "$logdir" --n_runs=3 --n_tests=1 &
    sleep 2
done
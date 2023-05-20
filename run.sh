#! /bin/bash
for logdir in logs/*alpha*beta*; do
    echo "$logdir"
    nohup python3 src/main.py new run "$logdir" --n_steps=1500000 --n_runs=3 --n_tests=1 &
    sleep 5
done
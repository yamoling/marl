#! /bin/bash

# Sleep for 3 hours
sleep 10800
nohup python src/run.py vdn-baseline-256 --run --n-tests=1 --n-runs=10 --seed=10&

sleep 10800
nohup python src/run.py vdn-baseline-256 --run --n-tests=1 --n-runs=10 --seed=20&

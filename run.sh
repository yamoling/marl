#! /bin/bash

for i in 0 1 2 3 4; do
    echo "python src/create_experiments.py --delay=$i --logdir=LLE-tmp-$i-DQN-VDN --run --n-runs=10 &"
    python src/create_experiments.py --delay=$i --logdir=LLE-tmp-$i-DQN-VDN --run --n-runs=10 --seed=10&
    sleep 10
done

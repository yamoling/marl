#! /bin/bash

for i in 0 1 2 3; do
    echo "python src/create_experiments.py --delay=$i --logdir=LLE-tmp-$i-DQN-VDN --run --n-runs=20 &"
    python src/create_experiments.py --delay=$i --logdir=LLE-tmp-$i-DQN-VDN --run --n-runs=20 &
    sleep 10
done

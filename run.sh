#! /bin/bash

python src/run.py logs/haven-dqn --n-runs=8 &
python src/run.py logs/haven-ppo --n-runs=8
python src/run.py logs/haven-dqn-ir --n-runs=8 &
python src/run.py logs/haven-ppo-ir --n-runs=8

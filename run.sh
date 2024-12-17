#! /bin/bash

# This scripts start multiple jobs as background processes but only allows
# up to max_parallel_runs to run at the same time.
# Whenever a job finishes, a new job is started until all jobs are done.

# Files like 2b-1, 2b-2, etc
python src/run.py logs/unlimited-time-laser-random --n-tests=20 --n-runs=10 
python src/run.py logs/unlimited-time-laser-vdn-walkable-lasers --n-tests=20 --n-runs=10 
python src/run.py logs/unlimited-time-laser-random-unwalkable-lasers --n-tests=20 --n-runs=10 
python src/run.py logs/unlimited-time-laser-vdn-unwalkable-lasers --n-tests=20 --n-runs=10 

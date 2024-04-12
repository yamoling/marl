#! /bin/bash

# This scripts start multiple jobs as background processes but only allows
# up to max_parallel_runs to run at the same time.
# Whenever a job finishes, a new job is started until all jobs are done.

# Files like 2b-1, 2b-2, etc
map_files="2b-1 2b-2 2b-3 2b-4 2b-5"

# For each file
for map_file in $map_files
do
    # Start a job
    python src/create_experiments.py --logdir=logs/${map_file}-punish --n-runs=10 --n-tests=10 --map-file="maps/${map_file}" --run 
    # Wait for 10 minutes
    sleep 10m
done




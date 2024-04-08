#! /bin/bash

# This scripts start multiple jobs as background processes but only allows
# up to max_parallel_runs to run at the same time.
# Whenever a job finishes, a new job is started until all jobs are done.

max_parallel_runs=3
total_n_runs=30
run_num=10
n_tests=1
logdir=logs/vdn-baseline-256


while [ $run_num -lt $total_n_runs ]
do
    n_running=$(jobs | wc -l | xargs)
    if [ $n_running -lt $max_parallel_runs ]
    then
        python src/run.py ${logdir} --n-tests=${n_tests} --seed=${run_num} --quiet --fill-strategy=fill --estimated-memory-MB=3000&
        run_num=$((run_num+1))
    fi
    sleep 5
done


wait 




#! /bin/bash

# Send a SIGTERM to all processes in the process group of the current process
trap kill_all SIGINT SIGTERM

function kill_all() {
    echo "Killing all running processes"
    for pid in ${pids[*]}; do
        kill -9 $pid
    done
    exit 1
}



num_runs=10
num_simultaneous_runs=5

env="CartPole-v0"
declare -a maps
maps[0]=""

declare -a algos
algos[0]="dqn"

tags=("avg_score" "min_score" "max_score" "std_score") # "avg_battle_won" "avg_dead_allies" "avg_dead_enemies")
model="MLP"
batch_size=64
memory_size=50000
n_steps=10000
test_interval=500


function run_experiment() {
    for map in "${maps[@]}"; do
        for algo in "${algos[@]}"; do
            unset logdirs
            unset pids
            for ((i=0;i<${num_runs};i+=1)); do
                # Wait for previous jobs to finish
                while [ $(jobs | wc -l) -ge ${num_simultaneous_runs} ]; do 
                    sleep 1
                done

                # Build env + map name
                if [ ! -z "${map}" ]; then
                    env_map="${env}:${map}"
                else
                    env_map="${env}"
                fi
                logdir="logs/NO-PER2-${env_map}-${algo}-${model}-${i}"
                python3 src/main.py train \
                    --env="${env_map}" \
                    --algo="${algo}" \
                    --model=${model} \
                    --logdir="${logdir}" \
                    --batch_size=${batch_size} \
                    --memory_size=${memory_size} \
                    --training_steps=${n_steps} \
                    --test_interval=${test_interval} \
                    --quiet &
                pids[${i}]=$!
                logdirs[${i}]="${logdir}"
            done
            # Wait for all jobs to finish
            wait
            echo "Starting data analysis in the background"
            # Process the training data
            ./src/analysis/tensorboard_mean_std.py --input_dirs ${logdirs[*]} --output_dir="results/NO-PER2-${env}-${map}-${algo}" --tags ${tags[*]} &
        done
    done
}


run_experiment

#!/bin/bash

# Decision flag for experiment type
configs_path="configs/federatedML"
wandb_api_key=""

# Arguments sent to this script
current_task_num=$1
n_parallel_tasks=$2
n_allocated_gpus=$3
initial_port_num=$4
((initial_port_num+=current_task_num))

# Configuration variables
environment_path="/mimer/NOBACKUP/groups/naiss2023-22-904/env_containers/dl_flwr_v2.sif"

# Start experiment deployment phase
echo "[ArrayID: $current_task_num/$n_parallel_tasks] - Running federated experiments."  
exp_count=$(ls -1 $configs_path | wc -l)
index=1
for f in "$configs_path"/*
do
    if (( index == current_task_num )); then
        echo "[ArrayID: $current_task_num/$n_parallel_tasks] - Processing config file $f on port# $initial_port_num."
        # Use the apptainer to run the federated experiments.
        apptainer exec --env "WANDB_API_KEY=$wandb_api_key" --nv $environment_path python -u src/run_federated.py --num-gpus=$n_allocated_gpus --port=$initial_port_num --config-file="$f"
        # Update port number for current task ID
        ((initial_port_num+=n_parallel_tasks))
    fi
    ((index%=n_parallel_tasks))
    ((index+=1))
done    
#!/bin/bash
#SBATCH -A NAISS2023-22-904 -p alvis
#SBATCH -t 0-00:10:00 
#SBATCH --gpus-per-node=T4:2
#SBATCH --nodes=1 --ntasks=1
#SBATCH --array=1-3
#SBATCH -J test
#SBATCH --output "temp/logs/slurm-%x_%A_%a.out"
#SBATCH --error "temp/logs/slurm-%x_%A_%a.err"

hostnames=$SLURM_NODELIST
echo "[ArrayID: $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT] - Fetched Hostnames are $hostnames"
wait

./deploy.sh $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT 2 59995

echo "[ArrayID: $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT] - Finished running all experiments!!!"
printf "\n\n"

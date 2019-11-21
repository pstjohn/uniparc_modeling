#!/bin/bash

#SBATCH --job-name=debug_array
#SBATCH --output=/scratch/pstjohn/array_%A_%a.out
#SBATCH --array=0-1
#SBATCH --time=01:00:00
#SBATCH --partition=debug
#SBATCH --account=cooptimasoot
#SBATCH --ntasks=1

python run_training.py $SLURM_ARRAY_TASK_ID \
--hostlist=$hostfile \
--port=5050 \
--modelName='debug' &

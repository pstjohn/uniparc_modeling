#!/bin/bash
#SBATCH --account=cooptimasoot
#SBATCH --time=00:60:00
#SBATCH --partition=debug
#SBATCH --job-name=bert_debug
#SBATCH --nodes=2
#SBATCH -c 36
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/pstjohn/gpu.%j.out  # %j will be replaced with the job ID

srun -l hostname

source /home/pstjohn/.bashrc

module purge
module load cudnn/7.4.2/cuda-10.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nopt/nrel/apps/cuda/10.0.130/extras/CUPTI/lib64/:/home/pstjohn/Packages/nccl_2.4.8-1+cuda10.0_x86_64/lib/
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_ENABLE_AUTO_MIXED_PRECISION=1

conda activate tf2

hostfile=/scratch/pstjohn/gpuhosts.${SLURM_JOB_ID}.txt
scontrol show hostnames=$SLURM_JOB_NODELIST > $hostfile

for ((i = 0 ; i < $SLURM_JOB_NUM_NODES ; i++)); do
    srun -l -n 1 --gres=gpu:2 --nodes=1 python run_training.py $i \
        --hostlist=$hostfile \
        --port=5050 \
        --modelName='debug' &
done

wait

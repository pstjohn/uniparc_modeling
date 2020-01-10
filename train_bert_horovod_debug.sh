#!/bin/bash
#SBATCH --account=cooptimasoot
#SBATCH --time=00:60:00
#SBATCH --partition=debug
#SBATCH --job-name=albert_debug_1e5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/pstjohn/%x.%j.out

# srun -l hostname

source /home/pstjohn/.bashrc

module purge
module load gcc/7.3.0
module load cuda/10.0.130
module load openmpi/3.1.3/cuda-10.0.130

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nopt/nrel/apps/cuda/10.0.130/extras/CUPTI/lib64/:/home/pstjohn/Packages/nccl_2.4.8-1+cuda10.0_x86_64/lib/
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_ENABLE_AUTO_MIXED_PRECISION=1
# export HOROVOD_TIMELINE=/scratch/pstjohn/horovod_timeline_debug.json

conda activate /projects/bpms/pstjohn/envs/tf2

mpirun \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python training_horovod_single_aa.py --modelName /scratch/pstjohn/albert_single_aa_debug_1gpu --batchSize=5 --stepsPerEpoch=1000 --warmup=3000 --lr=0.00001


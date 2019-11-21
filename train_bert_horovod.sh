#!/bin/bash
#SBATCH --account=cooptimasoot
#SBATCH --time=10-00
#SBATCH --partition=long
#SBATCH --job-name=albert_base_8000
#SBATCH --nodes=22
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
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

conda activate tf2

mpirun \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python training_horovod.py --modelName /scratch/pstjohn/albert_base_8000 --batchSize=10 --stepsPerEpoch=10000 --warmup=30000

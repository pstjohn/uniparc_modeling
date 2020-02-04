#!/bin/bash
#SBATCH --account=invpoly
#SBATCH --time=2-00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --job-name=distribute_debug_large
#SBATCH --output=/scratch/pstjohn/%j.%x.out  # %j will be replaced with the job ID

module unload
unset LD_PRELOAD

module load openmpi/3.1.3/gcc-7.3.0
module load singularity-container/3.2.1

SIMG=/projects/bpms/pstjohn/containers/tf_19.11-tf2-py3.simg
SINGULARTY_CMD="singularity exec -B /scratch,/projects --nv $SIMG"
SINGULARITYENV_NCCL_DEBUG=INFO
SINGULARITYENV_TF_ENABLE_AUTO_MIXED_PRECISION=1

srun $SINGULARTY_CMD \
    python run_model_mirrored_strategy.py \
    --modelName=$SLURM_JOB_NAME \
    --scratchDir='/scratch/pstjohn/' \
    --dataDir='/projects/bpms/pstjohn/uniparc' \
    --batchSize=48 \
    --warmup=10000 \
    --totalSteps=400000 \
    --stepsPerEpoch=500 \
    --lr=1E-4 \
    --sequenceLength=512 \
    --initialEpoch=0

#!/bin/bash
#SBATCH --partition=GPU-AI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:volta16:8
#SBATCH --time=2-00
#SBATCH --job-name=dist_128_fixed_activation
#SBATCH --output=/pylon5/mc5plsp/pstjohn/job_output/%j.%x

env | grep ^SLURM | egrep 'CPU|TASKS'
echo "slurm ntasks" $SLURM_NTASKS

source /etc/profile.d/modules.sh

module load singularity
module load /opt/packages/openmpi/4.0.0/intel/modulefiles/4.0.0-intel

cd /pylon5/mc5plsp/pstjohn/uniparc_modeling/

SIMG=/pylon5/containers/ngc/tensorflow/19.11-tf2-py3.simg
SINGULARTY_CMD="singularity exec -B /local --nv $SIMG"
SINGULARITYENV_NCCL_DEBUG=INFO
SINGULARITYENV_TF_ENABLE_AUTO_MIXED_PRECISION=1

srun $SINGULARTY_CMD \
    python run_model_mirrored_strategy.py \
    --modelName=$SLURM_JOB_NAME \
    --scratchDir='/pylon5/mc5plsp/pstjohn/uniparc_checkpoints' \
    --dataDir='/pylon5/mc5plsp/pstjohn/uniparc_data' \
    --batchSize=1024 \
    --warmup=10000 \
    --totalSteps=400000 \
    --stepsPerEpoch=500 \
    --lr=1E-4 \
    --sequenceLength=128 \
    --initialEpoch=0

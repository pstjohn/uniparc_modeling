#!/bin/bash
#SBATCH --account=cooptimasoot
#SBATCH --time=2-00
#SBATCH --qos=high
#SBATCH --job-name=bde_new
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/pstjohn/gpu.%j.out

source ~/.bashrc
conda activate /projects/rlmolecule/pstjohn/envs/tf2_10

export PYTHONPATH=$HOME/Research/uniparc:$PYTHONPATH
export NCCL_DEBUG=INFO
export TF_ENABLE_AUTO_MIXED_PRECISION=1
export OMP_NUM_THREADS=4


srun python train_localization_model.py \
    --modelName='12_layer_small_batch' \
    --scratchDir="/scratch/pstjohn/uniparc" \
    --dataDir="/projects/bpms/pstjohn/swissprot" \
    --checkpointDir="/scratch/pstjohn/uniparc/12_layer_bs24_adamw_20200527" \
    --batchSize=24 \
    --warmup=250 \
    --lr=.00002 \
    --stepsPerEpoch=250 \
    --validationSteps=10 \
    --totalSteps=50000

    #--checkpoint="$MEMBERWORK/bie108/uniparc_checkpoints/multinode_test_mixed_precision.113288" \

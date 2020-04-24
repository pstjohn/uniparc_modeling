#!/bin/bash

#BSUB -P BIE108
#BSUB -W 24:00
#BSUB -nnodes 50
#BSUB -q killable
#BSUB -J multinode_12_layer
#BSUB -o job_output/%J.out
#BSUB -e job_output/%J.err

NODES=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)

module load ibm-wml-ce/1.7.0-1
conda activate tf2-ibm
export PYTHONPATH=$HOME/uniparc_modeling:$PYTHONPATH
export NCCL_DEBUG=INFO
export TF_ENABLE_AUTO_MIXED_PRECISION=1
export OMP_NUM_THREADS=4

cd $MEMBERWORK/bie108

jsrun -n ${NODES} -g 6 -c 42 -r1 -a1 -b none python3 $HOME/uniparc_modeling/run_model_multi_worker.py \
    --modelName=$LSB_JOBNAME.$LSB_JOBID \
    --scratchDir="$MEMBERWORK/bie108/uniparc_checkpoints" \
    --dataDir="$PROJWORK/bie108/split_uniref100" \
    --checkpoint="$MEMBERWORK/bie108/uniparc_checkpoints/$LSB_JOBNAME.$LSB_JOBID" \
    --batchSize=3000 \
    --warmup=16000 \
    --totalSteps=400000 \
    --stepsPerEpoch=1000 \
    --validationSteps=25 \
    --lr=1E-4 \
    --maskingFreq=0.15 \
    --sequenceLength=512 \
    --modelDimension=768 \
    --numberXformerLayers=12 \
    --initialEpoch=0

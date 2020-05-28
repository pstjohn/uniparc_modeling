#!/bin/bash

#BSUB -P BIE108
#BSUB -W 01:00
#BSUB -nnodes 2
#BSUB -J multinode_hvd_test
#BSUB -o /ccs/home/pstjohn/job_output/%J.out
#BSUB -e /ccs/home/pstjohn/job_output/%J.err

NODES=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)

module load ibm-wml-ce/1.7.0-3
conda activate tf21-ibm
export PYTHONPATH=$HOME/uniparc_modeling:$PYTHONPATH
export NCCL_DEBUG=INFO
export TF_ENABLE_AUTO_MIXED_PRECISION=1
export OMP_NUM_THREADS=4

# use NCCL only communication 
export HOROVOD_HIERARCHICAL_ALLGATHER=0
export HOROVOD_HIERARCHICAL_ALLREDUCE=0
# set number of communication group to 1
export HOROVOD_GROUPED_ALLREDUCES=1
# set cycle time to 1ms 
export HOROVOD_CYCLE_TIME=1
# set fusion buffer size to 8MB
export HOROVOD_FUSION_THRESHOLD=8388608

export HOROVOD_TIMELINE="$MEMBERWORK/bie108/uniparc_checkpoints/$LSB_JOBNAME.$LSB_JOBID.hvdtimeline.json"

cd $MEMBERWORK/bie108
cp $HOME/uniparc_modeling/run_model_horovod.py .

jsrun -n ${NODES} -g 6 -c 42 -r1 -a6 python3 run_model_horovod.py \
    --modelName=$LSB_JOBNAME.$LSB_JOBID \
    --scratchDir="$MEMBERWORK/bie108/uniparc_checkpoints" \
    --dataDir="$PROJWORK/bie108/split_uniref100" \
    --batchSize=24 \
    --warmup=10 \
    --totalSteps=100 \
    --stepsPerEpoch=10 \
    --validationSteps=10 \
    --lr=1E-4 \
    --maskingFreq=0.15 \
    --sequenceLength=512 \
    --modelDimension=768 \
    --numberXformerLayers=12 \
    --initialEpoch=0


    #--checkpoint="$MEMBERWORK/bie108/uniparc_checkpoints/multinode_test_mixed_precision.113288" \

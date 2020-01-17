#!/bin/bash
module load singularity 
SIMG=/pylon5/containers/ngc/tensorflow/19.11-tf2-py3.simg
SINGULARTY_CMD="singularity run -B /local --nv $SIMG"

$SINGULARTY_CMD mpirun -np 1 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    --launch-agent "$SINGULARTY_CMD orted" \
    python run_model.py --modelName='albert-xlarge-16' \
    --batchSize=16

#!/bin/bash
mpirun \
    -np 1 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    singularity exec --nv /pylon5/containers/ngc/tensorflow/19.11-tf2-py3.simg \
    python run_model.py --modelName $SCRATCH/models/test_1 --batchSize=4


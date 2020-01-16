#!/bin/bash
#SBATCH --partition=GPU-AI
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:volta16:2
#SBATCH --time=1:00:00
#SBATCH --job-name=hvd_test
#SBATCH --output=/pylon5/mc5plsp/pstjohn/job_output/%x.%j.%n  # %j will be replaced with the job ID

cd /pylon5/mc5plsp/pstjohn/hvd_test/
module load singularity
SIMG=/pylon5/containers/ngc/tensorflow/19.11-tf2-py3.simg
SINGULARTY_CMD="singularity run -B /local --nv $SIMG"

$SINGULARTY_CMD mpirun -np 2 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    --launch-agent "$SINGULARTY_CMD orted" \
    python tensorflow2_keras_mnist.py

#!/bin/bash
#SBATCH --partition=GPU-AI
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:volta32:2
#SBATCH --time=1:00:00
#SBATCH --job-name=albert_2gpu_test
#SBATCH --output=/pylon5/mc5plsp/pstjohn/job_output/%x.%j.%n  # %j will be replaced with the job ID

cd /pylon5/mc5plsp/pstjohn/uniparc_modeling/

module load singularity
module load /opt/packages/openmpi/4.0.0/intel/modulefiles/4.0.0-intel

SIMG=/pylon5/containers/ngc/tensorflow/19.11-tf2-py3.simg
SINGULARTY_CMD="singularity exec -B /local --nv $SIMG"

$SINGULARTY_CMD horovodrun -np 2 run_model.py \
	--modelName=albert_2gpu_test \
	--batchSize=4 \
	--stepsPerEpoch=10000 \
	--warmup=10000 \
	--lr=1E-4 \
	--weightDecay=0.001 \
	--sequenceLength=768

	# -bind-to none -map-by slot \
	# -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
	# -mca pml ob1 -mca btl ^openib \

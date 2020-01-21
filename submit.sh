#!/bin/bash
#SBATCH --partition=GPU-AI
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:volta32:2
#SBATCH --time=1:00:00
#SBATCH --job-name=gpu32_test
#SBATCH --output=/pylon5/mc5plsp/pstjohn/job_output/%j.%x

source /etc/profile.d/modules.sh

module load singularity
module load /opt/packages/openmpi/4.0.0/intel/modulefiles/4.0.0-intel

cd /pylon5/mc5plsp/pstjohn/uniparc_modeling/

SIMG=/pylon5/containers/ngc/tensorflow/19.11-tf2-py3.simg
SINGULARTY_CMD="singularity exec -B /local --nv $SIMG"

# $SINGULARTY_CMD horovodrun -np $SLURM_NTASKS run_model.py \

# mpirun -np $SLURM_NTASKS \
# 	-bind-to none -map-by slot \
# 	-x SINGULARITYENV_NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
# 	-x SINGULARITYENV_TF_ENABLE_AUTO_MIXED_PRECISION=1 \
# 	-mca pml ob1 -mca btl ^openib \

SINGULARITYENV_TF_ENABLE_AUTO_MIXED_PRECISION=1 $SINGULARTY_CMD horovodrun -np 2 \
	python run_model.py \
	--modelName=$SLURM_JOB_NAME \
	--batchSize=3 \
	--stepsPerEpoch=10000 \
	--warmup=10000 \
	--lr=1E-4 \
	--weightDecay=0.0 \
	--sequenceLength=1024

	# -bind-to none -map-by slot \
	# -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
	# -mca pml ob1 -mca btl ^openib \

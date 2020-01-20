#!/bin/bash
#SBATCH --partition=GPU-AI
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:volta16:2
#SBATCH --time=00:10:00
#SBATCH --job-name=albert_test
#SBATCH --output=/pylon5/mc5plsp/pstjohn/job_output/%x.%j.%n  # %j will be replaced with the job ID

cd /pylon5/mc5plsp/pstjohn/uniparc_modeling/

module load singularity
module load /opt/packages/openmpi/4.0.0/intel/modulefiles/4.0.0-intel

SIMG=/pylon5/containers/ngc/tensorflow/19.11-tf2-py3.simg
SINGULARTY_CMD="singularity exec -B /local --nv $SIMG"


mpirun -np $SLURM_NTASKS \
	-bind-to none -map-by slot \
	-x SINGULARITYENV_NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
	-mca pml ob1 -mca btl ^openib \
	$SINGULARTY_CMD \
	python run_model.py \
	--modelName=$SLURM_JOB_NAME \
	--batchSize=3 \
	--stepsPerEpoch=10000 \
	--warmup=10000 \
	--lr=1E-4 \
	--weightDecay=0.0001 \
	--sequenceLength=768


	# -x SINGULARITYENV_TF_ENABLE_AUTO_MIXED_PRECISION=1 \
	# -x SINGULARITYENV_TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" \

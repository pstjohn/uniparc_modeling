#!/bin/bash
#SBATCH --account=invpoly
#SBATCH --partition=gpu
#SBATCH --time=2-00
#SBATCH --nodes=5
#SBATCH -c 18
#SBATCH --gres=gpu:2
#SBATCH --job-name=transformer_base_512
#SBATCH --output=/scratch/pstjohn/%j.%x.out  # %j will be replaced with the job ID

module unload
unset LD_PRELOAD

module load openmpi/3.1.3/gcc-7.3.0
module load singularity-container/3.2.1

SIMG=/projects/bpms/pstjohn/containers/tf_19.11-tf2-py3.simg
SINGULARTY_CMD="singularity exec -B /scratch,/projects --nv $SIMG"


mpirun \
	-x SINGULARITYENV_NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
	-x SINGULARITYENV_TF_ENABLE_AUTO_MIXED_PRECISION=1 \
	-bind-to none -map-by slot \
	-mca pml ob1 -mca btl ^openib \
	$SINGULARTY_CMD \
	python run_model.py \
	--modelName=$SLURM_JOB_NAME \
	--batchSize=48 \
	--warmup=10000 \
	--lr=1E-4 \
	--weightDecay=0.00 \
	--sequenceLength=256 \
	--tensorboardDirectory=/scratch/pstjohn/tblogs/$SLURM_JOB_NAME

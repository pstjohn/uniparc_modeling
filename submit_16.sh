#!/bin/bash
#SBATCH --partition=GPU-AI
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:volta16:8
#SBATCH --time=2-00
#SBATCH --job-name=16_bs128
#SBATCH --output=/pylon5/mc5plsp/pstjohn/job_output/%j.%x

env | grep ^SLURM | egrep 'CPU|TASKS'
echo "slurm ntasks" $SLURM_NTASKS

sourcedir=$SCRATCH/uniparc_data/*

RC=1
n=0
rsync -aP $sourcedir $LOCAL/

source /etc/profile.d/modules.sh

module load singularity
module load /opt/packages/openmpi/4.0.0/intel/modulefiles/4.0.0-intel

cd /pylon5/mc5plsp/pstjohn/uniparc_modeling/

SIMG=/pylon5/containers/ngc/tensorflow/19.11-tf2-py3.simg
SINGULARTY_CMD="singularity exec -B /local --nv $SIMG"

mpirun -np $SLURM_NTASKS \
	-x SINGULARITYENV_NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
	-x SINGULARITYENV_TF_ENABLE_AUTO_MIXED_PRECISION=1 \
	-bind-to none -map-by slot \
	-mca pml ob1 -mca btl ^openib \
	$SINGULARTY_CMD \
	python run_model.py \
	--modelName=$SLURM_JOB_NAME \
	--scratchDir=/pylon5/mc5plsp/pstjohn/uniparc_checkpoints \
	--dataDir=$LOCAL \
	--batchSize=128 \
	--warmup=10000 \
	--lr=1E-4 \
	--weightDecay=0.0 \
	--sequenceLength=128 \
	--initialEpoch=0

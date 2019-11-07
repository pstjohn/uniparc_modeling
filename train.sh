#!/bin/bash
#SBATCH --account=bpms
#SBATCH --time=24:00:00
#SBATCH --mem=200000
#SBATCH --job-name=spm_5M
#SBATCH --output=/scratch/pstjohn/spm.%j.out  # %j will be replaced with the job ID

export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH
cd /scratch/pstjohn/uniparc/
srun spm_train --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --user_defined_symbols="[MASK]" --input=sequences_train.txt --model_prefix=uniparc_5M --vocab_size=32000 --character_coverage=1.0 --model_type=unigram --input_sentence_size=5000000
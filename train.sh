#!/bin/bash
#SBATCH --account=cooptimasoot
#SBATCH --qos=high
#SBATCH --time=48:00:00
#SBATCH --partition=bigmem
#SBATCH --mem=750000
#SBATCH --job-name=spm_8000
#SBATCH --output=/scratch/pstjohn/spm.%j.out  # %j will be replaced with the job ID

export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH
cd /scratch/pstjohn/uniparc/
srun spm_train --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --user_defined_symbols="[MASK]" --input=sequences_train_no_BZUX.txt --model_prefix=uniparc_10M_8000 --vocab_size=8000 --character_coverage=1.0 --model_type=unigram --input_sentence_size=10000000

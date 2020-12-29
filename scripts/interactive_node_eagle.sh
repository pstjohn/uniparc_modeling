#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --mem=200000
#SBATCH -A deepgreen
#SBATCH -p debug
#SBATCH --time 60

cd $HOME
source ~/.bashrc
conda activate /projects/rlmolecule/pstjohn/envs/tf2
/projects/rlmolecule/pstjohn/envs/tf2/bin/jupyter-lab --no-browser --ip=0.0.0.0 --port=8888

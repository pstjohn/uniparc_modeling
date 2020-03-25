#!/bin/bash

module unload
unset LD_PRELOAD

module load singularity-container/3.2.1
SIMG=/projects/bpms/pstjohn/containers/faiss.simg

singularity exec --nv -B /scratch,/projects $SIMG jupyter lab --no-browser --ip=0.0.0.0 --port=1234

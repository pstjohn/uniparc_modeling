#!/bin/bash

module unload
unset LD_PRELOAD

module load singularity-container/3.2.1
SIMG=/projects/bpms/pstjohn/containers/tf_19.11-tf2-py3.simg

singularity exec --nv -B /scratch,/projects $SIMG jupyter lab --no-browser --ip=0.0.0.0 --port=8889

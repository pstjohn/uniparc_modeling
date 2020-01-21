#!/bin/bash

module load singularity 
SIMG=/pylon5/containers/ngc/tensorflow/19.11-tf2-py3.simg

cd $SCRATCH
singularity exec --nv -B /local $SIMG jupyter lab --no-browser --ip=0.0.0.0

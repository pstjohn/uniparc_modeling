#!/bin/bash

module load ibm-wml-ce/1.7.0-1
conda activate tf2-ibm
jupyter lab --no-browser --ip=0.0.0.0

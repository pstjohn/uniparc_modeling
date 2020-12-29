#!/bin/bash

module load ibm-wml-ce/1.7.0-2
conda activate tf21-ibm
jupyter lab --no-browser --ip=0.0.0.0

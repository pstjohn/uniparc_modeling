#!/bin/bash

module load ibm-wml-ce/1.7.0-1
conda activate tf2-ibm
tensorboard --logdir $MEMBERWORK/bie108/uniparc_checkpoints/tblogs
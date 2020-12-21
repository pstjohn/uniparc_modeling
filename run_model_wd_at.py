import os
import re
import argparse
import shutil
import subprocess
import json

## Initialize the TF_CONFIG environment variable based on process rank
# From https://code.ornl.gov/olcf-analytics/summit/distributed-deep-learning-examples/tree/master/examples/tensorflow
# Get a list of compute nodes allocated for your job
get_cnodes = "echo $(cat {} | sort | uniq | grep -v batch | grep -v login)".format(os.environ['LSB_DJOB_HOSTFILE'])
cnodes = subprocess.check_output(get_cnodes, shell=True)
cnodes = str(cnodes)[2:-3].split(' ')
nodes_list = [c + ":2222" for c in cnodes] # Add a port number

# Get the rank of the compute node that is running on
index = int(os.environ['PMIX_RANK'])

# Set the TF_CONFIG environment variable to configure the cluster setting.
tf_config = json.dumps({
    'cluster': {
        'worker': nodes_list
    },
    'task': {'type': 'worker', 'index': index} 
})

print(tf_config, flush=True)

os.environ['TF_CONFIG'] = tf_config

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
    communication=tf.distribute.experimental.CollectiveCommunication.NCCL)

from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# Parse input arguments
parser = argparse.ArgumentParser(description='BERT model training')
parser.add_argument('--modelName', default='albert-xlarge',
                    help='model name for directory saving')
parser.add_argument('--batchSize', type=int, default=8, 
                    help='batch size per gpu')
parser.add_argument('--warmup', type=int, default=10000, 
                    help='warmup steps')
parser.add_argument('--totalSteps', type=int, default=100000, 
                    help='total steps')
parser.add_argument('--lr', type=float, default=1E-4, 
                    help='initial learning rate')
parser.add_argument('--sequenceLength', type=int, default=1024, 
                    help='Protein AA sequence length')
parser.add_argument('--scratchDir', default=None, 
                    help='Directory for tensorboard logs and checkpoints')
parser.add_argument('--dataDir', default=None, 
                    help='Directory for training and validation data')
parser.add_argument('--checkpoint', default=None, 
                    help='Restore model from checkpoint')
parser.add_argument('--initialEpoch', type=int, default=0, 
                    help='starting epoch')
parser.add_argument('--stepsPerEpoch', type=int, default=500, 
                    help='steps per epoch')
parser.add_argument('--validationSteps', type=int, default=25, 
                    help='validation steps')
parser.add_argument('--maskingFreq', type=float, default=.15, 
                    help='overall masking frequency')
parser.add_argument('--modelDimension', type=int, default=512, 
                    help='attention dimension')
parser.add_argument('--numberXformerLayers', type=int, default=6, 
                    help='number of tranformer layers')
parser.add_argument('--dropout', type=float, default=0.0, 
                    help='dropout')
parser.add_argument('--weightDecay', type=str, default='true', 
                    help='weightDecay')
parser.add_argument('--attentionType', type=str, default='relative', 
                    help='attentionType')
parser.add_argument('--restart', type=str, default='false', 
                    help='ignore checkpoint epoch')


arguments = parser.parse_args()
print(arguments, flush=True)

import numpy as np

from bert.losses import (ECE, masked_sparse_categorical_crossentropy,
                         masked_sparse_categorical_accuracy)
from bert.model import create_model, create_albert_model
from bert.dataset import create_masked_input_dataset

# Create the optimizer
from bert.optimization import create_optimizer

optimizer = create_optimizer(arguments.lr, arguments.totalSteps, arguments.warmup, optimizer="lamb")

# Training data path -- here the data's been sharded to allow multi-worker splits

with tf.device('/CPU:0'):
    
    training_data = create_masked_input_dataset(
        sequence_path=os.path.join(
            arguments.dataDir, 'train_uniref100_split/train_100_*.txt'),
        max_sequence_length=arguments.sequenceLength,
        batch_size=arguments.batchSize,
        masking_freq=arguments.maskingFreq,
        fix_sequence_length=True,
        sequence_compression=None,
        file_buffer_size=2048,
        buffer_size=10000,
        filter_bzux=False)
    
    valid_data = create_masked_input_dataset(
        sequence_path=os.path.join(
            arguments.dataDir, 'dev_uniref50_split/dev_50_*.txt'),
        max_sequence_length=arguments.sequenceLength,
        batch_size=arguments.batchSize,
        masking_freq=arguments.maskingFreq,
        fix_sequence_length=True,
        sequence_compression=None,
        filter_bzux=False)

from tensorflow.keras import layers
from bert.layers import DenseNoMask

with strategy.scope():   

#    model = create_albert_model(model_dimension=arguments.modelDimension,
#                                transformer_dimension=arguments.modelDimension * 4,
#                                num_attention_heads=arguments.modelDimension // 64,
#                                num_transformer_layers=arguments.numberXformerLayers,
#                                vocab_size=24,
#                                dropout_rate=arguments.dropout,
#                                max_relative_position=64,
#                                final_layernorm=False)
    
    model = create_model(model_dimension=arguments.modelDimension,
                         transformer_dimension=arguments.modelDimension * 4,
                         num_attention_heads=arguments.modelDimension // 64,
                         num_transformer_layers=arguments.numberXformerLayers,
                         vocab_size=24,
                         dropout_rate=arguments.dropout,
                         max_relative_position=64,
                         max_sequence_length=arguments.sequenceLength,
                         attention_type=arguments.attentionType)
    
    if arguments.checkpoint:
        checkpoint = tf.train.latest_checkpoint(arguments.checkpoint)

        print(f'loading checkpoint: {checkpoint}')

        # Support for 'killable' queue, where initially there's no checkpoint
        if checkpoint is not None:
            model.load_weights(checkpoint)
            print('loading checkpoint {} ...'.format(os.path.basename(checkpoint)))

            if arguments.restart == 'false':
                arguments.initialEpoch = int(
                    re.findall('.(\d{3})-', os.path.basename(checkpoint))[0])
                
    model.compile(
        loss=masked_sparse_categorical_crossentropy,
        metrics=[ECE],
        optimizer=optimizer)

#     if arguments.restart != 'false':
#         # Make sure we create the optimizer and recompile
#         optimizer = create_optimizer()  
#         model.compile(
#             loss=masked_sparse_categorical_crossentropy,
#             metrics=[ECE],
#             optimizer=optimizer)
#         print(optimizer.weights)

## Create keras callbacks
callbacks = []

model_name = arguments.modelName

# Only do these on the head node
if index == 0:

    checkpoint_dir = os.path.join(arguments.scratchDir, model_name)
    logdir = os.path.join(arguments.scratchDir, 'tblogs', model_name)
    
    # Print model structure to stdout
    model.summary()
    
    # Make the checkpoint directory
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Make sure this script is available later
    shutil.copy(__file__, checkpoint_dir)

    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()

else:
    
    checkpoint_dir = os.path.join('/tmp', model_name, f'worker_{index}')
    logdir = os.path.join('/tmp', 'tblogs', model_name, f'worker_{index}')
    from pathlib import Path
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(logdir).mkdir(parents=True, exist_ok=True)

    

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "weights.{epoch:03d}-{val_ECE:.2f}"),
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        monitor='val_ECE'),
    
    tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=0,
        write_graph=False,
        update_freq='epoch',
        profile_batch=0,
        embeddings_freq=0)
]

model.fit(training_data, steps_per_epoch=arguments.stepsPerEpoch,
          epochs=arguments.totalSteps//arguments.stepsPerEpoch,
          initial_epoch=arguments.initialEpoch,
          verbose=1 if index == 0 else 0, 
          validation_data=valid_data,
          validation_steps=arguments.validationSteps,
          callbacks=callbacks)

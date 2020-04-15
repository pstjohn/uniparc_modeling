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

print(tf_config)
os.environ['TF_CONFIG'] = tf_config

import tensorflow as tf
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

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

arguments = parser.parse_args()
print(arguments)

import numpy as np

from bert.losses import ECE, masked_sparse_categorical_crossentropy
from bert.model import create_albert_model, load_model_from_checkpoint
from bert.dataset import create_masked_input_dataset

## Create the optimizer
from bert.optimization import create_optimizer
optimizer = create_optimizer(arguments.lr, arguments.warmup, arguments.totalSteps)

# from bert.optimization import WarmUp

# learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
#     initial_learning_rate=arguments.lr,
#     decay_steps=arguments.totalSteps,
#     end_learning_rate=0.0)

# learning_rate_fn = WarmUp(initial_learning_rate=arguments.lr,
#                           decay_schedule_fn=learning_rate_fn,
#                           warmup_steps=arguments.warmup)

# optimizer = tf.keras.optimizers.Adam(
#     learning_rate=learning_rate_fn,
#     beta_1=0.9,
#     beta_2=0.999,
#     epsilon=1e-6)

# Training data path -- here the data's been sharded to allow multi-worker splits
train_data_dir = os.path.join(arguments.dataDir, 'train_uniref100_split')
train_data_files = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir)]

valid_data_dir = os.path.join(arguments.dataDir, 'dev_uniref50_split')
valid_data_files = [os.path.join(valid_data_dir, f) for f in os.listdir(valid_data_dir)]

with strategy.scope():
    
#    with tf.device('/CPU:0'):
    training_data = create_masked_input_dataset(
        sequence_path=train_data_files,
        max_sequence_length=arguments.sequenceLength,
        batch_size=arguments.batchSize,
        masking_freq=arguments.maskingFreq,
        fix_sequence_length=True)

    training_data = training_data.repeat().prefetch(tf.data.experimental.AUTOTUNE)    

    valid_data = create_masked_input_dataset(
        sequence_path=valid_data_files,        
        max_sequence_length=arguments.sequenceLength,
        batch_size=arguments.batchSize,
        masking_freq=arguments.maskingFreq,
        fix_sequence_length=True)

    valid_data = valid_data.repeat().prefetch(tf.data.experimental.AUTOTUNE)
    
    ## Create the model
    model = create_albert_model(model_dimension=arguments.modelDimension,
                                transformer_dimension=arguments.modelDimension * 4,
                                num_attention_heads=arguments.modelDimension // 64,
                                num_transformer_layers=arguments.numberXformerLayers,
                                vocab_size=24,
                                dropout_rate=arguments.dropout,
                                max_relative_position=64,
                                final_layernorm=False)
    
    if arguments.checkpoint:
        checkpoint = tf.train.latest_checkpoint(arguments.checkpoint)

        # Support for 'killable' queue, where initially there's no checkpoint
        if checkpoint is not None:
            model.load_weights(checkpoint)
            arguments.initialEpoch = int(
                re.findall('.(\d{3})-', os.path.basename(checkpoint))[0])
            print('loading checkpoint {} ...'.format(os.path.basename(checkpoint)))

    model.compile(
        loss=masked_sparse_categorical_crossentropy,
        metrics=[ECE],
        optimizer=optimizer)

## Create keras callbacks
callbacks = []

model_name = arguments.modelName
checkpoint_dir = f'{arguments.scratchDir}/{model_name}/'
logdir = f'{arguments.scratchDir}/tblogs/{model_name}'

# Only do these on the head node
if index == 0:
    
    # Print model structure to stdout
    model.summary()
    
    # Make the checkpoint directory
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Make sure this script is available later
    shutil.copy(__file__, checkpoint_dir)


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
          verbose=1, validation_data=valid_data,
          validation_steps=arguments.validationSteps,
          callbacks=callbacks)

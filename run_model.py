import os
import argparse
import shutil

parser = argparse.ArgumentParser(description='BERT model training')
parser.add_argument('--modelName', default='albert-xlarge',
                    help='model name for directory saving')
parser.add_argument('--weightShare', type=bool, default=False, 
                    help='Albert-style weight sharing')
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
parser.add_argument('--attentionType', type=str, default='relative', 
                    help='attention type')
parser.add_argument('--modelDimension', type=int, default=512, 
                    help='attention dimension')
parser.add_argument('--numberXformerLayers', type=int, default=6, 
                    help='number of tranformer layers')
parser.add_argument('--dropout', type=float, default=0.0, 
                    help='dropout')

arguments = parser.parse_args()
print(arguments)

import numpy as np
import tensorflow as tf

from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

from bert.losses import ECE, masked_sparse_categorical_crossentropy
from bert.model import create_albert_model, load_model_from_checkpoint
from bert.dataset import create_masked_input_dataset

## Create the model
# from bert.optimization import create_optimizer
# optimizer = create_optimizer(arguments.lr, arguments.warmup, arguments.totalSteps)

from bert.optimization import WarmUp
import tensorflow_addons.optimizers as tfa_optimizers

lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=arguments.lr,
    decay_steps=arguments.totalSteps,
    end_learning_rate=0.0)

lr_schedule = WarmUp(
    initial_learning_rate=arguments.lr,
    decay_schedule_fn=lr_schedule,
    warmup_steps=arguments.warmup)

optimizer = tfa_optimizers.LAMB(
    learning_rate=lr_schedule,
    weight_decay_rate=0.01,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-6,
    exclude_from_weight_decay=['layer_norm', 'bias'])

training_data = create_masked_input_dataset(
    sequence_path=os.path.join(
        arguments.dataDir, 'train_uniref100_split/train_100_*.txt.gz'),
    max_sequence_length=arguments.sequenceLength,
    batch_size=arguments.batchSize,
    masking_freq=arguments.maskingFreq,
    fix_sequence_length=True)

valid_data = create_masked_input_dataset(
    sequence_path=os.path.join(
        arguments.dataDir, 'dev_uniref50_split/dev_50_*.txt.gz'),
    max_sequence_length=arguments.sequenceLength,
    batch_size=arguments.batchSize,
    masking_freq=arguments.maskingFreq,
    fix_sequence_length=True)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    
    model = create_albert_model(model_dimension=arguments.modelDimension,
                                transformer_dimension=arguments.modelDimension * 4,
                                num_attention_heads=arguments.modelDimension // 64,
                                num_transformer_layers=arguments.numberXformerLayers,
                                vocab_size=24,
                                dropout_rate=arguments.dropout,
                                max_relative_position=64,
                                final_layernorm=False)
    
    if arguments.checkpoint:
        model.load_weights(arguments.checkpoint)

    model.compile(
        loss=masked_sparse_categorical_crossentropy,
        metrics=[ECE],
        optimizer=optimizer)

model.summary()

## Create keras callbacks
callbacks = []

model_name = arguments.modelName
checkpoint_dir = f'{arguments.scratchDir}/{model_name}/'
logdir = f'{arguments.scratchDir}/tblogs/{model_name}'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Make sure this script is available later
shutil.copy(__file__, checkpoint_dir)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "saved_weights"),
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        monitor='val_ECE'),
    tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=0,
        write_graph=False,
        update_freq='epoch',
        profile_batch=2,
        embeddings_freq=0)
]

model.fit(training_data, steps_per_epoch=arguments.stepsPerEpoch,
          epochs=arguments.totalSteps//arguments.stepsPerEpoch,
          initial_epoch=arguments.initialEpoch,
          verbose=1, validation_data=valid_data,
          validation_steps=arguments.validationSteps,
          callbacks=callbacks)

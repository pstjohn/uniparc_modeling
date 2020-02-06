import os
import argparse
import shutil

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
parser.add_argument('--maskingFreq', type=float, default=.15, 
                    help='overall masking frequency')
parser.add_argument('--attentionType', type=str, default='relative', 
                    help='attention type')
parser.add_argument('--numberXformerLayers', type=str, default='relative', 
                    help='attention type')
arguments = parser.parse_args()
print(arguments)

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from bert.losses import ECE, masked_sparse_categorical_crossentropy
from bert.model import create_albert_model, load_model_from_checkpoint


## Create the model
# from bert.optimization import create_optimizer
# optimizer = create_optimizer(arguments.lr, arguments.warmup, arguments.totalSteps)

from bert.optimization import WarmUp, create_optimizer

optimizer = create_optimizer(arguments.lr, arguments.warmup, arguments.totalSteps)

# learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
#     initial_learning_rate=arguments.lr,
#     decay_steps=arguments.totalSteps,
#     end_learning_rate=0.0)
# 
# learning_rate_fn = WarmUp(initial_learning_rate=arguments.lr,
#                           decay_schedule_fn=learning_rate_fn,
#                           warmup_steps=arguments.warmup)
# 
# optimizer = tf.keras.optimizers.Adam(
#     learning_rate=learning_rate_fn,
#     beta_1=0.9,
#     beta_2=0.999,
#     epsilon=1e-6)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():

    model = create_albert_model(model_dimension=768,
                                transformer_dimension=768 * 4,
                                num_attention_heads=768 // 64,
                                num_transformer_layers=24,
                                vocab_size=24,
                                dropout_rate=0.1,
                                max_relative_position=128,
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
        profile_batch=20,
        embeddings_freq=0)
]

from bert.dataset import create_masked_input_dataset

# Keep the data preprocessing steps on the CPU
with tf.device('/CPU:0'):
    
    training_data = create_masked_input_dataset(
        sequence_path=os.path.join(arguments.dataDir, 'train_uniref100.txt.gz'),
        max_sequence_length=arguments.sequenceLength,
        batch_size=arguments.batchSize,
        masking_freq=arguments.maskingFreq)

    valid_data = create_masked_input_dataset(
        sequence_path=os.path.join(arguments.dataDir, 'dev_uniref50.txt.gz'),
        max_sequence_length=arguments.sequenceLength,
        batch_size=arguments.batchSize,
        masking_freq=arguments.maskingFreq)


model.fit(training_data, steps_per_epoch=arguments.stepsPerEpoch,
          epochs=arguments.totalSteps//arguments.stepsPerEpoch,
          initial_epoch=arguments.initialEpoch,
          verbose=1, validation_data=valid_data, validation_steps=25,
          callbacks=callbacks)

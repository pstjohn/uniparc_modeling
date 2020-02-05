import os
import argparse

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

from bert.optimization import WarmUp

learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=arguments.lr,
    decay_steps=arguments.totalSteps,
    end_learning_rate=0.0)

learning_rate_fn = WarmUp(initial_learning_rate=arguments.lr,
                          decay_schedule_fn=learning_rate_fn,
                          warmup_steps=arguments.warmup)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate_fn,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-6)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    
    if not arguments.checkpoint:
        model = create_albert_model(model_dimension=512,
                                    transformer_dimension=512 * 4,
                                    num_attention_heads=512 // 64,
                                    num_transformer_layers=6,
                                    vocab_size=24,
                                    dropout_rate=0.,
                                    max_relative_position=64,
                                    weight_share=arguments.weightShare)
        
    else:
        model = load_model_from_checkpoint(arguments.checkpoint)

    model.compile(
        loss=masked_sparse_categorical_crossentropy,
        metrics=[ECE],
        optimizer=optimizer)

model.summary()

## Create keras callbacks
callbacks = []

model_name = arguments.modelName
checkpoint_dir = f'{arguments.scratchDir}/{model_name}_checkpoints'
logdir = f'{arguments.scratchDir}/tblogs/{model_name}'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "ckpt.h5"),
        save_best_only=True,
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
        batch_size=arguments.batchSize)

    valid_data = create_masked_input_dataset(
        sequence_path=os.path.join(arguments.dataDir, 'dev_uniref50.txt.gz'),
        max_sequence_length=arguments.sequenceLength,
        batch_size=arguments.batchSize)


model.fit(training_data, steps_per_epoch=arguments.stepsPerEpoch,
          epochs=arguments.totalSteps//arguments.stepsPerEpoch,
          initial_epoch=arguments.initialEpoch,
          verbose=1, validation_data=valid_data, validation_steps=25,
          callbacks=callbacks)

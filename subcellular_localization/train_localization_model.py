import os
import shutil
import argparse
from functools import partial

import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa

from bert.dataset import encode
from bert.optimization import WarmUp


parser = argparse.ArgumentParser(description='subcell localization model training')
parser.add_argument('--modelName', default='subcell-model',
                    help='model name for directory saving')
parser.add_argument('--batchSize', type=int, default=24, 
                    help='batch size')
parser.add_argument('--warmup', type=int, default=10000, 
                    help='warmup steps')
parser.add_argument('--epochs', type=int, default=10, 
                    help='total steps')
parser.add_argument('--lr', type=float, default=1E-4, 
                    help='initial learning rate')
parser.add_argument('--checkpointDir', default=None, 
                    help='directory for pretrained model')
parser.add_argument('--scratchDir', default=None, 
                    help='Directory for tensorboard logs and checkpoints')
parser.add_argument('--dataDir', default=None, 
                    help='Directory for training and validation data')
parser.add_argument('--stepsPerEpoch', type=int, default=500, 
                    help='steps per epoch')
parser.add_argument('--validationSteps', type=int, default=25, 
                    help='validation steps')
parser.add_argument('--totalSteps', type=int, default=10000, 
                    help='total steps')

arguments = parser.parse_args()
print(arguments)

strategy = tf.distribute.MirroredStrategy()

from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

data = pd.read_parquet(os.path.join(arguments.dataDir, 'parsed_swissprot.parquet')).set_index('accession')
train = pd.read_csv(os.path.join(arguments.dataDir, 'subcellular/train.csv.gz')).sample(frac=1.)
valid = pd.read_csv(os.path.join(arguments.dataDir, 'subcellular/valid.csv.gz')).sample(frac=1.)

num_targets = train.shape[1] - 1

# Create the datasets
max_seq_len=512
fix_seq_len=True
batch_size=arguments.batchSize

def create_dataset(sequences,
                   targets,
                   buffer_size=1000):
    
    encoded = tf.data.Dataset.from_tensor_slices(sequences.values)\
        .map(partial(encode, max_sequence_length=max_seq_len))
    target_ds = tf.data.Dataset.from_tensor_slices(
        targets.values.astype(np.int32))
    zipped = tf.data.Dataset.zip((encoded, target_ds))\
        .shuffle(buffer_size).repeat()
    return zipped

# Subsample the training data to get an even mix of all classes
def get_train_subsampled():
    for location in train.columns[1:]:
        train_subset = train[train[location] == 1]
        train_seq_subset = data.reindex(train_subset.accession).sequence
        yield create_dataset(train_seq_subset, train_subset.set_index('accession'))

training_data = tf.data.experimental.sample_from_datasets(list(get_train_subsampled()))\
    .shuffle(batch_size)\
    .padded_batch(batch_size=batch_size, padded_shapes=(
    [-1 if not fix_seq_len else max_seq_len], [num_targets]))\
    .prefetch(tf.data.experimental.AUTOTUNE)

valid_sequences = data.reindex(valid.accession).sequence
validation_data = create_dataset(valid_sequences, valid.set_index('accession'))\
    .padded_batch(batch_size=batch_size, padded_shapes=(
    [-1 if not fix_seq_len else max_seq_len], [num_targets]))\
    .prefetch(tf.data.experimental.AUTOTUNE)


# Create optimizer

lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=arguments.lr,
    decay_steps=arguments.totalSteps,
    end_learning_rate=0.0)

lr_schedule = WarmUp(
    initial_learning_rate=arguments.lr,
    decay_schedule_fn=lr_schedule,
    warmup_steps=arguments.warmup)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-6)


def create_model():
    from bert.model import create_albert_model

    dimension = 768

    model = create_albert_model(model_dimension=dimension,
                                transformer_dimension=dimension * 4,
                                num_attention_heads=dimension // 64,
                                num_transformer_layers=12,
                                dropout_rate=0.,
                                max_relative_position=64,
                                final_layernorm=False)
    
    if arguments.checkpointDir is not None:
        model.load_weights(tf.train.latest_checkpoint(
            arguments.checkpointDir)).expect_partial()

    model.trainable = True
    
    final_embedding = model.layers[-2].input
    residue_predictions = tf.keras.layers.Dense(num_targets)(final_embedding)
    protein_predictions = tf.keras.layers.GlobalMaxPooling1D()(residue_predictions)

    localization_model = tf.keras.Model(model.inputs, protein_predictions)
    
    return localization_model
    

with strategy.scope():   

    model = create_model()
    
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=optimizer)
    

## Create keras callbacks
callbacks = []

model_name = arguments.modelName

checkpoint_dir = os.path.join(arguments.scratchDir, model_name)
logdir = os.path.join(arguments.scratchDir, 'tblogs', model_name)

# Print model structure to stdout
model.summary()

# Make the checkpoint directory
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Make sure this script is available later
shutil.copy(__file__, checkpoint_dir)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "weights.{epoch:03d}-{val_loss:.3f}"),
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        monitor='val_loss'),
    
    tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=0,
        write_graph=False,
        update_freq='epoch',
        profile_batch=0,
        embeddings_freq=0)
]

model.fit(training_data, 
          validation_data=validation_data,
          verbose=1,
          callbacks=callbacks,
          steps_per_epoch=arguments.stepsPerEpoch,
          epochs=arguments.totalSteps//arguments.stepsPerEpoch,
          validation_steps=arguments.validationSteps)

import os
import argparse

parser = argparse.ArgumentParser(description='BERT model training')
parser.add_argument('--modelName', default='albert-xlarge', help='model name for directory saving')
parser.add_argument('--batchSize', type=int, default=8, help='batch size per gpu')
parser.add_argument('--stepsPerEpoch', type=int, default=10000, help='steps per epoch')
parser.add_argument('--warmup', type=int, default=10000, help='warmup steps')
parser.add_argument('--lr', type=float, default=1E-4, help='initial learning rate')
parser.add_argument('--weightDecay', type=float, default=0.01, help='AdamW weight decay')
parser.add_argument('--sequenceLength', type=int, default=1024, help='Protein AA sequence length')
arguments = parser.parse_args()

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

tf.compat.v1.disable_eager_execution()

import horovod.tensorflow.keras as hvd
from bert.hvd_utils import is_using_hvd

# Horovod: initialize Horovod.
if is_using_hvd():
    print('Initializing Horovod')
    hvd.init()

# Print runtime config on head node
if not is_using_hvd() or hvd.rank() == 0:
    print(arguments)

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus and is_using_hvd():
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

from bert.dataset import create_masked_input_dataset

hvd_size = hvd.size() if is_using_hvd() else None
hvd_rank = hvd.rank() if is_using_hvd() else None

training_data = create_masked_input_dataset(
    sequence_path='../uniparc_data/sequences_train.txt',
    max_sequence_length=arguments.sequenceLength,
    batch_size=arguments.batchSize,
    fix_sequence_length=False,
    shard_num_workers=hvd_size,
    shard_worker_index=hvd_rank)

training_data.repeat().prefetch(tf.data.experimental.AUTOTUNE)

valid_data = create_masked_input_dataset(
    sequence_path='../uniparc_data/sequences_valid.txt',
    max_sequence_length=arguments.sequenceLength,
    batch_size=arguments.batchSize,
    fix_sequence_length=False,
    shard_num_workers=hvd_size,
    shard_worker_index=hvd_rank)

valid_data.repeat().prefetch(tf.data.experimental.AUTOTUNE)

# Create the model
from bert.model import create_albert_model
model = create_albert_model(embedding_dimension=128,
                            max_embedding_sequence_length=1024,
                            model_dimension=2048,
                            transformer_dimension=2048 * 4,
                            num_attention_heads=2048 // 64,
                            num_transformer_layers=24,
                            vocab_size=22,
                            dropout_rate=0.)

if hvd_rank or not is_using_hvd() == 0:
    model.summary()
    
from bert.optimizers import ECE, masked_sparse_categorical_crossentropy, BertLinearSchedule
    
opt = tfa.optimizers.AdamW(learning_rate=arguments.lr, weight_decay=arguments.weightDecay)
opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

# Horovod: add Horovod DistributedOptimizer.
if is_using_hvd():
    opt = hvd.DistributedOptimizer(opt)

# Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
# uses hvd.DistributedOptimizer() to compute gradients.
true_labels = tf.keras.layers.Input(
    shape=(None,), dtype=tf.int32, batch_size=None)

model.compile(
    target_tensors=true_labels,    
    loss=masked_sparse_categorical_crossentropy,
    metrics=[ECE],
    optimizer=opt,
    experimental_run_tf_function=True)

model_name = arguments.modelName
checkpoint_dir = f'{model_name}_checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.h5")

callbacks = [    
    # Add warmup and learning rate decay
    BertLinearSchedule(arguments.lr, arguments.warmup, int(1E7)),
]

if is_using_hvd():
    callbacks += [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if (not is_using_hvd() or hvd_rank == 0):
    callbacks.append(tf.keras.callbacks.CSVLogger(f'{checkpoint_dir}/log.csv'))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix))
    
# Horovod: write logs on worker 0.
verbose = 1 if (hvd_rank == 0 or not is_using_hvd()) else 0

model.fit(training_data, steps_per_epoch=arguments.stepsPerEpoch, epochs=1000,
          verbose=verbose, validation_data=valid_data, validation_steps=100,
          callbacks=callbacks)

import os
import argparse

parser = argparse.ArgumentParser(description='BERT model training')
parser.add_argument('--modelName', default='albert-xlarge',
                    help='model name for directory saving')
parser.add_argument('--batchSize', type=int, default=8, 
                    help='batch size per gpu')
parser.add_argument('--warmup', type=int, default=10000, 
                    help='warmup steps')
parser.add_argument('--lr', type=float, default=1E-4, 
                    help='initial learning rate')
parser.add_argument('--weightDecay', type=float, default=0.01, 
                    help='AdamW weight decay')
parser.add_argument('--sequenceLength', type=int, default=1024, 
                    help='Protein AA sequence length')
parser.add_argument('--scratchDir', default=None, 
                    help='Directory for tensorboard logs and checkpoints')
parser.add_argument('--checkpoint', default=None, 
                    help='Restore model from checkpoint')
parser.add_argument('--initialEpoch', type=int, default=0, 
                    help='starting epoch')

arguments = parser.parse_args()

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

tf.compat.v1.disable_eager_execution()

import horovod.tensorflow.keras as hvd
# from bert.hvd_utils import is_using_hvd
is_using_hvd = lambda: True

# Horovod: initialize Horovod.
if is_using_hvd():
    print('Initializing Horovod')
    hvd.init()

# Print runtime config on head node
if hvd.rank() == 0:
    print(arguments)

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"HVD rank: {hvd.rank()}, GPUs: {gpus}")
if gpus and is_using_hvd():
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

hvd_size = hvd.size() if is_using_hvd() else None
hvd_rank = hvd.rank() if is_using_hvd() else None

# Create the model
if not arguments.checkpoint:
    from bert.model import create_albert_model
    model = create_albert_model(model_dimension=512,
                                transformer_dimension=512 * 4,
                                num_attention_heads=512 // 64,
                                num_transformer_layers=6,
                                vocab_size=22,
                                dropout_rate=0.)
else:
    from bert.model import load_model_from_checkpoint
    model = load_model_from_checkpoint(arguments.checkpoint)


if hvd_rank == 0:
    model.summary()
    
from bert.optimizers import (ECE, masked_sparse_categorical_crossentropy,
                             BertLinearSchedule)

# opt = tf.optimizers.Adam(learning_rate=1E-4,
#                         beta_2=0.98,
#                         epsilon=1E-6)

opt = tfa.optimizers.AdamW(learning_rate=arguments.lr,
                           beta_2=0.98,
                           epsilon=1E-6,
                           weight_decay=arguments.weightDecay)

# opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

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
checkpoint_dir = f'{arguments.scratchDir}/{model_name}_checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

callbacks = [    
    # Add warmup and learning rate decay
    BertLinearSchedule(arguments.lr, arguments.warmup, int(1E6)),
]

if is_using_hvd():
    callbacks += [
    # Horovod: broadcast initial variable states from rank 0 to all other
    # processes.  This is necessary to ensure consistent initialization of all
    # workers when training is started with random weights or restored from a
    # checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from
# corrupting them.
if hvd.rank() == 0:
    callbacks += [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "ckpt.h5"),
            save_best_only=True,
            mode='min',
            monitor='val_ECE'),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'{arguments.scratchDir}/tblogs/{model_name}',
            histogram_freq=0,
            write_graph=False,
            update_freq='epoch',
            profile_batch=0,
            embeddings_freq=0)
    ]

    
# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0

from bert.dataset import create_masked_input_dataset

training_data = create_masked_input_dataset(
    sequence_path='../uniparc_data/sequences_train.txt',
    max_sequence_length=arguments.sequenceLength,
    batch_size=arguments.batchSize,
    shard_num_workers=hvd_size,
    shard_worker_index=hvd_rank)

training_data = training_data.repeat().prefetch(tf.data.experimental.AUTOTUNE)

valid_data = create_masked_input_dataset(
    sequence_path='../uniparc_data/sequences_valid.txt',
    max_sequence_length=arguments.sequenceLength,
    batch_size=arguments.batchSize,
    shard_num_workers=hvd_size,
    shard_worker_index=hvd_rank)

valid_data = valid_data.repeat().prefetch(tf.data.experimental.AUTOTUNE)

model.fit(training_data, steps_per_epoch=200, epochs=500, 
          initial_epoch=arguments.initialEpoch,
          verbose=verbose, validation_data=valid_data, validation_steps=20,
          callbacks=callbacks)
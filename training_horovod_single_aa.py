import os
import argparse

parser = argparse.ArgumentParser(description='BERT model training')
parser.add_argument('--modelName', default='bert', help='model name for directory saving')
parser.add_argument('--batchSize', type=int, default=20, help='batch size per gpu')
parser.add_argument('--stepsPerEpoch', type=int, default=10000, help='steps per epoch')
parser.add_argument('--warmup', type=int, default=16000, help='warmup steps')
parser.add_argument('--lr', type=float, default=1E-4, help='initial learning rate')
arguments = parser.parse_args()

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from bert.dataset import create_masked_input_dataset
from bert.layers import (PositionEmbedding, Attention, Transformer, TokenEmbedding, Bias,
                         gelu, masked_sparse_categorical_crossentropy, ECE,
                         InverseSquareRootSchedule, initializer, Projection)

import horovod.tensorflow.keras as hvd

# Horovod: initialize Horovod.
hvd.init()

# Print runtime config on head node
if hvd.rank() == 0:
    print(arguments)

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


# import tensorflow_addons as tfa
from tensorflow.keras import layers

vocab_size = 22
max_seq_len = 1024

def encode(line_tensor):
    line = line_tensor.numpy().decode('utf8')

    if len(line) > max_seq_len:
        offset = np.random.randint(
            low=0, high=len(line) - max_seq_len + 1)
        line = line[offset:(offset + max_seq_len)]

    vocab = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
             'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 
             'W', 'Y']

    replacement_dict = {key: i + 2 for i, key in enumerate(vocab)}
    return np.asarray([replacement_dict[item] for item in line])

def encode_tf(line_tensor):
    return tf.py_function(encode, inp=[line_tensor], Tout=[tf.int32,])

training_data = create_masked_input_dataset(
    encode_fn=encode_tf,
    sequence_path='/projects/bpms/pstjohn/uniparc/sequences_train.txt',
    max_sequence_length=max_seq_len,
    batch_size=arguments.batchSize,
    buffer_size=1024,
    vocab_size=vocab_size,
    mask_index=1,
    vocab_start=2,
    fix_sequence_length=True,
    shard_num_workers=hvd.size(),
    shard_worker_index=hvd.rank())

training_data.repeat().prefetch(tf.data.experimental.AUTOTUNE)

valid_data = create_masked_input_dataset(
    encode_fn=encode_tf,
    sequence_path='/projects/bpms/pstjohn/uniparc/sequences_valid.txt',
    max_sequence_length=max_seq_len,
    batch_size=arguments.batchSize,
    buffer_size=1024,
    vocab_size=vocab_size,
    mask_index=1,
    vocab_start=2,
    fix_sequence_length=True,
    shard_num_workers=hvd.size(),
    shard_worker_index=hvd.rank())

valid_data.repeat().prefetch(tf.data.experimental.AUTOTUNE)

embedding_dimension = 128
model_dimension = 768
transformer_dimension = 4 * model_dimension
num_attention_heads = model_dimension // 64
num_transformer_layers = 12

# embedding_dimension = 32
# model_dimension = 64
# num_attention_heads = model_dimension // 16
# num_transformer_layers = 4

dropout_rate = 0.

# Horovod: adjust learning rate based on number of GPUs.
learning_rate = 1E-4

inputs = layers.Input(shape=(max_seq_len,), dtype=tf.int32, batch_size=None)

token_embedding_layer = TokenEmbedding(
    vocab_size, embedding_dimension, embeddings_initializer=initializer(), mask_zero=True)
token_embeddings = token_embedding_layer(inputs)
position_embeddings = PositionEmbedding(
    max_seq_len + 1, embedding_dimension, embeddings_initializer=initializer(),
    mask_zero=True)(inputs)

embeddings = layers.Add()([token_embeddings, position_embeddings])
embeddings = Projection(model_dimension, dropout_rate, use_residual=False)(embeddings)

transformer = Transformer(num_attention_heads, transformer_dimension, dropout=dropout_rate)
for i in range(num_transformer_layers):
    embeddings = transformer(embeddings)

out = layers.Dense(embedding_dimension, activation=gelu, kernel_initializer=initializer())(embeddings)
out = token_embedding_layer(out, transpose=True)
out = Bias()(out)

model = tf.keras.Model(inputs, out, name='model')

if hvd.rank() == 0:
    model.summary()
    
# Horovod: add Horovod DistributedOptimizer.
# opt = tfa.optimizers.AdamW(weight_decay=0.01, learning_rate=learning_rate)

opt = tf.optimizers.Adam(learning_rate=arguments.lr)
opt = hvd.DistributedOptimizer(opt)

# Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
# uses hvd.DistributedOptimizer() to compute gradients.

true_labels = layers.Input(shape=(None,), dtype=tf.int32, batch_size=None)
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
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),
    
    # Add warmup and learning rate decay
    InverseSquareRootSchedule(arguments.lr, arguments.warmup),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.CSVLogger(f'{checkpoint_dir}/log.csv'))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix))
    
# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0

model.fit(training_data, steps_per_epoch=arguments.stepsPerEpoch, epochs=500,
          verbose=verbose, validation_data=valid_data, validation_steps=100,
          callbacks=callbacks)

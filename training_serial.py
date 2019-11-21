import os
import argparse

parser = argparse.ArgumentParser(description='BERT model training')
parser.add_argument('--modelName', default='bert', help='model name for directory saving')
arguments = parser.parse_args()

import tensorflow as tf

from bert.dataset import create_masked_input_dataset
from bert.layers import (PositionEmbedding, Attention, Transformer, TokenEmbedding, Bias,
                         gelu, masked_sparse_cross_entropy_loss, InverseSquareRootSchedule,
                         initializer)

import tensorflow_addons as tfa
from tensorflow.keras import layers

vocab_size = 8000
max_seq_len = 512

training_data = create_masked_input_dataset(
    language_model_path='sentencepiece_models/uniparc_10M_8000.model',
    sequence_path='/projects/bpms/pstjohn/uniparc/sequences_train.txt',
    max_sequence_length=max_seq_len,
    batch_size=20,
    buffer_size=1024,
    vocab_size=vocab_size,
    mask_index=4,
    vocab_start=5,
    fix_sequence_length=True)

training_data.repeat().prefetch(tf.data.experimental.AUTOTUNE)

valid_data = create_masked_input_dataset(
    language_model_path='sentencepiece_models/uniparc_10M_8000.model',
    sequence_path='/projects/bpms/pstjohn/uniparc/sequences_valid.txt',
    max_sequence_length=max_seq_len,
    batch_size=20,
    buffer_size=1024,
    vocab_size=vocab_size,
    mask_index=4,
    vocab_start=5,
    fix_sequence_length=True)

valid_data.prefetch(tf.data.experimental.AUTOTUNE)

# embedding_dimension = 128
# model_dimension = 512
# num_attention_heads = model_dimension // 64
# num_transformer_layers = 12

embedding_dimension = 32
model_dimension = 64
num_attention_heads = model_dimension // 16
num_transformer_layers = 4

dropout_rate = 0.1

# Horovod: adjust learning rate based on number of GPUs.
learning_rate = 1E-4
warmup_updates = 3000


inputs = layers.Input(shape=(max_seq_len,), dtype=tf.int32, batch_size=None)
input_mask = layers.Input(shape=(max_seq_len,), dtype=tf.bool, batch_size=None)

token_embedding_layer = TokenEmbedding(
    vocab_size, embedding_dimension, embeddings_initializer=initializer(), mask_zero=True)
token_embeddings = token_embedding_layer(inputs)
position_embeddings = PositionEmbedding(
    max_seq_len + 1, embedding_dimension, embeddings_initializer=initializer(),
    mask_zero=True)(inputs)

embeddings = layers.Add()([token_embeddings, position_embeddings])
embeddings = layers.Dense(model_dimension)(embeddings)

transformer = Transformer(num_attention_heads, dropout=dropout_rate)
for i in range(num_transformer_layers):
    embeddings = transformer(embeddings)

out = layers.Dense(embedding_dimension, activation=gelu, kernel_initializer=initializer())(embeddings)
out = token_embedding_layer(out, transpose=True)
out = Bias()([out, input_mask])

model = tf.keras.Model([inputs, input_mask], [out], name='model')
model.summary()


# Horovod: add Horovod DistributedOptimizer.
opt = tfa.optimizers.AdamW(weight_decay=0.01, learning_rate=learning_rate)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    optimizer=opt)

model_name = arguments.modelName
checkpoint_dir = f'{model_name}_checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.h5")

callbacks = [
    InverseSquareRootSchedule(learning_rate=learning_rate, warmup_updates=warmup_updates),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
callbacks.append(tf.keras.callbacks.CSVLogger(f'{checkpoint_dir}/log.csv'))
callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix))
    

model.fit(training_data, steps_per_epoch=100, epochs=10, verbose=1,
          validation_data=valid_data, validation_steps=10,
          callbacks=callbacks)

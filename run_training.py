import os
import json
import argparse

parser = argparse.ArgumentParser(description='BERT model training')
parser.add_argument('n', type=int, help='worker id')
parser.add_argument('--hostlist', help='file containing list of hosts')
parser.add_argument('--port', '-p', default=22834, help='communication port')
parser.add_argument('--modelName', default='bert', help='model name for directory saving')

arguments = parser.parse_args()

with open(arguments.hostlist) as file:
    hosts = file.readlines()

num_hosts = len(hosts)
n = arguments.n
    
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": [f"{worker.strip()}:{arguments.port}" for worker in hosts],
    },
   "task": {"type": "chief" if n is 0 else "worker", "index": n}
})

print(os.environ["TF_CONFIG"], flush=True)

# import tensorflow as tf
# import tensorflow_addons as tfa
# from tensorflow.keras import layers

# mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

# from bert.dataset import create_masked_input_dataset
# from bert.layers import (PositionEmbedding, Attention, Transformer, TokenEmbedding, Bias,
#                          gelu, masked_sparse_cross_entropy_loss, InverseSquareRootSchedule,
#                          initializer)

# vocab_size = 8000
# max_seq_len = 512

# training_data = create_masked_input_dataset(
#     language_model_path='sentencepiece_models/uniparc_10M_8000.model',
#     sequence_path='/projects/bpms/pstjohn/uniparc/sequences_train.txt',
#     max_sequence_length=max_seq_len,
#     batch_size=20 * num_hosts,
#     buffer_size=1024,
#     vocab_size=vocab_size,
#     mask_index=4,
#     vocab_start=5,
#     fix_sequence_length=True)

# training_data.repeat().prefetch(tf.data.experimental.AUTOTUNE)

# valid_data = create_masked_input_dataset(
#     language_model_path='sentencepiece_models/uniparc_10M_8000.model',
#     sequence_path='/projects/bpms/pstjohn/uniparc/sequences_valid.txt',
#     max_sequence_length=max_seq_len,
#     batch_size=20 * num_hosts,
#     buffer_size=1024,
#     vocab_size=vocab_size,
#     mask_index=4,
#     vocab_start=5,
#     fix_sequence_length=True)

# valid_data.prefetch(tf.data.experimental.AUTOTUNE)

# embedding_dimension = 128
# model_dimension = 512
# num_attention_heads = model_dimension // 64
# num_transformer_layers = 12
# dropout_rate = 0.1

# learning_rate = 1E-4
# warmup_updates = 1000

# with mirrored_strategy.scope():

#     inputs = layers.Input(shape=(max_seq_len,), dtype=tf.int32, batch_size=None)
#     input_mask = layers.Input(shape=(max_seq_len,), dtype=tf.bool, batch_size=None)

#     token_embedding_layer = TokenEmbedding(
#         vocab_size, embedding_dimension, embeddings_initializer=initializer(), mask_zero=True)
#     token_embeddings = token_embedding_layer(inputs)
#     position_embeddings = PositionEmbedding(
#         max_seq_len + 1, embedding_dimension, embeddings_initializer=initializer(),
#         mask_zero=True)(inputs)

#     embeddings = layers.Add()([token_embeddings, position_embeddings])
#     embeddings = layers.Dense(model_dimension)(embeddings)

#     transformer = Transformer(num_attention_heads, dropout=dropout_rate)
#     for i in range(num_transformer_layers):
#         embeddings = transformer(embeddings)

#     out = layers.Dense(embedding_dimension, activation=gelu, kernel_initializer=initializer())(embeddings)
#     out = token_embedding_layer(out, transpose=True)
#     out = Bias()([out, input_mask])

#     model = tf.keras.Model([inputs, input_mask], [out], name='model')
#     model.summary()

#     model.compile(
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(
#             from_logits=True, reduction=tf.keras.losses.Reduction.NONE),
#         metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
#         optimizer=tfa.optimizers.AdamW(weight_decay=0.01, learning_rate=learning_rate))

# model_name = arguments.modelName
# checkpoint_dir = f'{model_name}_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# callbacks = [
# #    tf.keras.callbacks.TensorBoard(log_dir=f'./{model_name}_logs'),
#     tf.keras.callbacks.CSVLogger(f'{checkpoint_dir}/log.csv'),    
#     tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix),
#     InverseSquareRootSchedule(learning_rate=learning_rate, warmup_updates=warmup_updates)
# ]

# model.fit(training_data, steps_per_epoch=100, epochs=10, verbose=1,
#                     validation_data=valid_data, validation_steps=10,
#                     callbacks=callbacks)
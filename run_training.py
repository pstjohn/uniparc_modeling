import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers

from bert.dataset import create_masked_input_dataset
from bert.layers import (PositionEmbedding, Attention, Transformer, TokenEmbedding, Bias,
                         gelu, masked_sparse_cross_entropy_loss, BERTLearningRateScheduler,
                         initializer)

model_name = 'bert'

training_data = create_masked_input_dataset(
    language_model_path='uniparc_10M.model',
    sequence_path='/projects/bpms/pstjohn/uniparc/sequences_train.txt',
    max_sequence_length=512,
    batch_size=40,
    buffer_size=1024,
    vocab_size=32000,
    mask_index=4,
    vocab_start=5,
    fix_sequence_length=True)

training_data.repeat().prefetch(tf.data.experimental.AUTOTUNE)


valid_data = create_masked_input_dataset(
    language_model_path='uniparc_10M.model',
    sequence_path='/projects/bpms/pstjohn/uniparc/sequences_valid.txt',
    max_sequence_length=512,
    batch_size=40,
    buffer_size=1024,
    vocab_size=32000,
    mask_index=4,
    vocab_start=5,
    fix_sequence_length=True)

valid_data.prefetch(tf.data.experimental.AUTOTUNE)


vocab_size = 32000
max_seq_len = 512
embedding_dimension = 128
model_dimension = 512
num_attention_heads = model_dimension // 64
num_transformer_layers = 12
dropout_rate = 0.1

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():

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

    transformer = Transformer(num_attention_heads)
    for i in range(num_transformer_layers):
        embeddings = transformer(embeddings)

    out = layers.Dense(embedding_dimension, activation=gelu, kernel_initializer=initializer())(embeddings)
    out = token_embedding_layer(out, transpose=True)
    out = Bias()([out, input_mask])

    model = tf.keras.Model([inputs, input_mask], [out], name='model')
    model.summary()

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        optimizer=tfa.optimizers.AdamW(weight_decay=0.01, learning_rate=1E-3))


checkpoint_dir = f'./{model_name}_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=f'./{model_name}_logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix),
    BERTLearningRateScheduler(initial_learning_rate=1E-3, num_warmup_steps=1000)
]

model.fit_generator(training_data, steps_per_epoch=100, epochs=10, verbose=1,
                    validation_data=valid_data, validation_steps=10,
                    callbacks=callbacks)
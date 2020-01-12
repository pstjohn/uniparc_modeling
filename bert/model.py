import tensorflow as tf
from tensorflow.keras import layers

from bert.layers import (PositionEmbedding, Attention, Transformer, TokenEmbedding, Bias,
                         gelu, initializer, Projection)

def create_albert_model(embedding_dimension=128,
                        max_embedding_sequence_length=1024,
                        model_dimension=768,
                        transformer_dimension=3072,
                        num_attention_heads=12,
                        num_transformer_layers=12,
                        vocab_size=22,
                        dropout_rate=0.):
    
    inputs = layers.Input(shape=(None,), dtype=tf.int32, batch_size=None)

    token_embedding_layer = TokenEmbedding(
        vocab_size, embedding_dimension, embeddings_initializer=initializer(),
        mask_zero=True)
    token_embeddings = token_embedding_layer(inputs)
    position_embeddings = PositionEmbedding(
        max_embedding_sequence_length + 1, embedding_dimension,
        embeddings_initializer=initializer(),
        mask_zero=True)(inputs)

    embeddings = layers.Add()([token_embeddings, position_embeddings])
    embeddings = Projection(model_dimension, dropout_rate,
                            use_residual=False)(embeddings)

    transformer = Transformer(num_attention_heads, transformer_dimension,
                              dropout=dropout_rate)
    for i in range(num_transformer_layers):
        embeddings = transformer(embeddings)

    out = layers.Dense(embedding_dimension, activation=gelu,
                       kernel_initializer=initializer())(embeddings)
    out = token_embedding_layer(out, transpose=True)
    out = Bias()(out)

    model = tf.keras.Model(inputs, out, name='model')
    
    return model
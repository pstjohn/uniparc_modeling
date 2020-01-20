import tensorflow as tf
from tensorflow.keras import layers

from bert.layers import (PositionEmbedding, Attention, Transformer, TokenEmbedding, Bias,
                         gelu, initializer, Projection, DenseNoMask)

def create_albert_model(model_dimension=768,
                        transformer_dimension=3072,
                        num_attention_heads=12,
                        num_transformer_layers=12,
                        vocab_size=22,
                        dropout_rate=0.):
    
    inputs = layers.Input(shape=(None,), dtype=tf.int32, batch_size=None)

    # Amino-acid level embeddings
    embeddings = TokenEmbedding(
        vocab_size, model_dimension, embeddings_initializer=initializer(),
        mask_zero=True)(inputs)
    
    # Initialize transformer, use ALBERT-style weight sharing
    transformer = Transformer(num_attention_heads, transformer_dimension,
                              attention_type='relative', max_relative_position=10,
                              dropout=dropout_rate)
    
    # Stack transformers together
    for i in range(num_transformer_layers):
        embeddings = transformer(embeddings)

    # Project back to original embedding dimension
    out = DenseNoMask(vocab_size, activation=gelu,
                      kernel_initializer=initializer())(embeddings)

    model = tf.keras.Model(inputs, out, name='model')
    
    return model
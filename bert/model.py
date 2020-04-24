import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

from bert.layers import (Attention, Transformer,
                         gelu, initializer, Projection, DenseNoMask)

from bert.losses import masked_sparse_categorical_crossentropy, ECE
from bert.optimization import WarmUp

def create_albert_model(model_dimension=768,
                        transformer_dimension=3072,
                        num_attention_heads=12,
                        num_transformer_layers=12,
                        vocab_size=24,
                        dropout_rate=0.,
                        max_relative_position=64,
                        final_layernorm=True,
                        attention_type='relative'):
    
    inputs = layers.Input(shape=(None,), dtype=tf.int32, batch_size=None)

    # Amino-acid level embeddings
    embeddings = layers.Embedding(
        vocab_size, model_dimension, embeddings_initializer=initializer(),
        mask_zero=True)(inputs)
        
    # Stack transformers together
    for i in range(num_transformer_layers):
        
        # Whether to use layernorm on the final layer
        if not final_layernorm and i == (num_transformer_layers - 1):            
            use_layernorm=False
        else:
            use_layernorm=True
            
        transformer = Transformer(
            num_attention_heads, transformer_dimension,
            attention_type=attention_type,
            max_relative_position=max_relative_position,
            dropout=dropout_rate,
            use_layernorm=use_layernorm)
            
        embeddings = transformer(embeddings)

    # Project to the 20 AA labels (and zero 'pad' label)
    outputs = DenseNoMask(21, kernel_initializer=initializer())(embeddings)
    outputs = layers.Activation('linear', dtype='float32')(outputs)
    
    model = tf.keras.Model(inputs, outputs, name='model')
    
    return model


def load_model_from_checkpoint(checkpoint_file):
    model = load_model(
        checkpoint_file,
        custom_objects={
            'Transformer': Transformer,
            'Attention': Attention,
            'DenseNoMask': DenseNoMask,
            'gelu': gelu,
            'WarmUp': WarmUp,
            'masked_sparse_categorical_crossentropy': masked_sparse_categorical_crossentropy,
            'ECE': ECE})    
    
    return model

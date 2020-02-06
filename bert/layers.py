import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers

from bert.attention_utils import relative_attention_inner, initializer

def gelu(x):
    """
    Gelu activation from arXiv:1606.08415.
    """
    cdf = 0.5 * (
        1.0 + tf.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
    return x * cdf


class PositionEmbedding(layers.Embedding):
    """ Return masked embeddings according to the position index of each input
    """
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        idx = tf.tile(tf.expand_dims(tf.range(seq_len), 0), [batch_size, 1]) + 1
        return super(PositionEmbedding, self).call(idx)


class TokenEmbedding(layers.Embedding):
    """ Slight modification of original keras embedding layer in order to
    permit re-use of weights when converting final embeddings to logit
    predictions. """
    
    def call(self, inputs, transpose=False):
        if not transpose:
            return super(TokenEmbedding, self).call(inputs)
        else:
            return tf.matmul(inputs, self.embeddings, transpose_b=True)


class Attention(layers.Layer):
    """ Implements the multi-head attention transformer model.
    Includes the preceeding dense layers on the value, key, and query matrices
    """

    def __init__(self, units, num_heads, dropout=0.0, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units          # H
        self.num_heads = num_heads  # N
        self.dropout = dropout
        
    def build(self, input_shape):
        """ B, S, N, H - batch, fseq_len, num_heads, size_per_head """
        
        dense_units = self.units * self.num_heads  # N*H
        
        self.qk_layer = layers.Dense(
            dense_units, kernel_initializer=initializer(), name='qk')
        self.value_layer = layers.Dense(
            dense_units, kernel_initializer=initializer(), name='value')
        
        self.dropout_layer = layers.Dropout(self.dropout)
        
    def create_attention_mask(self, input_shape, input_mask):
        mask = tf.cast(tf.expand_dims(input_mask, axis=1), tf.float32)                   # [B, 1, S]
        ones = tf.expand_dims(tf.ones(shape=input_shape[:2], dtype=tf.float32), axis=-1)  # [B, S, 1]
        mask = ones * mask  # broadcast along two dimensions
        # Don't allow nodes to attend to themselves
        mask = mask - tf.eye(tf.shape(input_mask)[-1]) 
        mask = tf.clip_by_value(mask, 0, 1)
        return tf.expand_dims(mask, axis=1)  # [B,1,S,S]
    
    def transpose_scores(self, input_tensor):
        input_shape  = tf.shape(input_tensor)
        output_shape = [input_shape[0], input_shape[1], self.num_heads, self.units]
        output_tensor = tf.reshape(input_tensor, output_shape)
        return tf.transpose(a=output_tensor, perm=[0, 2, 1, 3])  # [B,N,S,H]

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0], input_shape[1], self.num_heads * self.units]
        return output_shape  # [B, S, N*H]

    def calculate_attention(self, qk, input_shape):
        return (tf.matmul(qk, qk, transpose_b=True) 
                / tf.sqrt(float(self.units)))        
        
    def call(self, inputs, mask=None, training=None):

        input_shape = tf.shape(inputs) # [B, S, N*H]
        
        # query and key can be the same vector
        qk = self.transpose_scores(self.qk_layer(inputs))  # [B,N,S,H]
        value = self.transpose_scores(self.value_layer(inputs))  # [B,N,S,H]

        # Equation 1 of "Attention is all you need"
        attention_scores = self.calculate_attention(qk, input_shape)  # [B,N,S,S]

        # zero out masked values
        attention_mask = self.create_attention_mask(input_shape, mask)
        attention_scores = attention_scores + (1. - attention_mask) * -10000.0
        
        attention_probs = tf.nn.softmax(attention_scores)  # [B,N,S,S]
        attention_probs = self.dropout_layer(attention_probs, training=training)
        context_layer = tf.matmul(attention_probs, value)  # [B,N,S,S]
        context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1, 3])            
        
        output_shape = [input_shape[0], input_shape[1], self.num_heads*self.units]
        context_layer = tf.reshape(context_layer, output_shape)

        return context_layer, attention_probs

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({'units': self.units, 
                       'num_heads': self.num_heads,
                       'dropout': self.dropout})
        return config


class RelativeAttention(Attention):
    def __init__(self, units, num_heads, max_relative_position, **kwargs):
        self.max_relative_position = max_relative_position
        super(RelativeAttention, self).__init__(units, num_heads, **kwargs)

    def build(self, input_shape):
        super(RelativeAttention, self).build(input_shape)
        
        self.relations_keys_embedding = layers.Embedding(
            self.max_relative_position * 2 + 1, self.units,
            embeddings_initializer=initializer(),
            name='relative_positions_keys')
                
    def _generate_relative_positions_matrix(self, length):
        """Generates matrix of relative positions between inputs.
        From https://github.com/tensorflow/tensor2tensor/ """
        range_vec = tf.range(length)
        range_mat = tf.reshape(tf.tile(range_vec, [length]), [length, length])
        distance_mat = range_mat - tf.transpose(range_mat)
        distance_mat_clipped = tf.clip_by_value(
            distance_mat, -self.max_relative_position,
            self.max_relative_position)
        
        # Shift values to be >= 0. Each integer still uniquely identifies a relative
        # position difference.
        final_mat = distance_mat_clipped + self.max_relative_position
        return final_mat

    def calculate_attention(self, qk, input_shape):
        """ Eq. 4 of arXiv:1803.02155 """
        relative_positions = self._generate_relative_positions_matrix(input_shape[1])
        relations_keys = self.relations_keys_embedding(relative_positions)
        attention_scores = relative_attention_inner(qk, qk, relations_keys, True) 
        return attention_scores

    def get_config(self):
        config = super(RelativeAttention, self).get_config()
        config.update({'max_relative_position': self.max_relative_position})
        return config
    
    
class Projection(layers.Layer):
    """ Performs a dense layer, dropout, layer norm and residual update """
    def __init__(self, units, dropout=0.0, use_residual=True, use_layernorm=True, **kwargs):
        super(Projection, self).__init__(**kwargs)        
        self.units = units
        self.dropout = dropout        
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm

    def build(self, input_shape):
        self.dense_layer = layers.Dense(self.units,
                                        kernel_initializer=initializer(),
                                        activation=gelu)
        
        self.dropout_layer = layers.Dropout(self.dropout)
        self.layer_norm = layers.LayerNormalization()

    def call(self, inputs, training=None):
        
        output, residual = inputs if self.use_residual else (inputs, None)
        
        output = self.dense_layer(output)
        output = self.dropout_layer(output, training=training)
        
        if self.use_residual:
            output = output + residual
        if self.use_layernorm:
            output = self.layer_norm(output)
            
        return output

    def get_config(self):
        config = super(Projection, self).get_config()
        config.update({'units': self.units,
                       'dropout': self.dropout,
                       'use_residual': self.use_residual,
                       'use_layernorm': self.use_layernorm})
        return config


class Transformer(layers.Layer):
    def __init__(self, num_heads, 
                 intermediate_units, 
                 dropout=0.0,
                 attention_type='attention',
                 max_relative_position=10,
                 use_layernorm=True,
                 **kwargs):
        """Performs the multi-headed attention and normalization of a single
        transformer block.
    
        Arguments:
            num_heads - number of attention heads. Typically model_dimension // 64
            intermediate_units - dimension of the dense layers inside transformer. 
                Typically 4x the model dimension.
            dropout - dropout rate
            attention_type - 'attention' or 'relative'. Wether to use 
                relative positional encodings from arXiv:1803.02155.
            max_relative_position - for relative positions
            use_layernorm - whether to use layernorm in the final layer
        """
        
        super(Transformer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.dropout = dropout
        self.intermediate_units = intermediate_units
        self.attention_type = attention_type
        self.max_relative_position = max_relative_position
        self.use_layernorm = use_layernorm
        
    def build(self, input_shape):

        # Split the model dimension equally amoung attention heads
        d_model = input_shape[-1]
        assert d_model % self.num_heads == 0, \
            f"input dimension {d_model} not divisible by {self.num_heads} "\
            "attention heads"
        
        self.units = d_model // self.num_heads
        
        if self.attention_type == 'attention':
            self.attention_layer = Attention(self.units, self.num_heads, self.dropout)
            
        elif self.attention_type == 'relative':
            self.attention_layer = RelativeAttention(
                self.units, self.num_heads, self.max_relative_position,
                dropout=self.dropout)
            
        self.intermediate_layer = layers.Dense(self.intermediate_units,
                                               kernel_initializer=initializer(),
                                               activation=gelu)
        
        self.attention_projection = Projection(d_model, self.dropout,
                                               name='attention_projection')
        
        self.output_projection = Projection(d_model, self.dropout,
                                            name='output_projection',
                                            use_layernorm=self.use_layernorm)


    def call(self, inputs, mask=None, training=None):
        
        # Multi-head attention block
        attention_output, attention_scores = self.attention_layer(inputs, mask=mask)
        attention_output = self.attention_projection([attention_output, inputs])
        
        intermediate_values = self.intermediate_layer(attention_output)
        output = self.output_projection([intermediate_values, attention_output])
        return output, attention_scores
    
    def compute_mask(self, inputs, mask=None):
        return mask 

    def get_config(self):
        config = super(Transformer, self).get_config()
        config.update({'num_heads': self.num_heads,
                       'intermediate_units': self.intermediate_units,
                       'dropout': self.dropout,
                       'attention_type': self.attention_type,
                       'max_relative_position': self.max_relative_position,
                       'use_layernorm': self.use_layernorm})
        return config


class Bias(layers.Layer):
    """ Final bias layer added to logits prior to softmax scoring. This layer
    also applys the input mask from the input to mask non-randomized prediction
    targets """

    def build(self, input_shape):
        self.bias = self.add_weight(name='classifier_bias',
                                    dtype=K.floatx(),
                                    shape=[input_shape[-1]],
                                    initializer=initializer())
        
    def call(self, inputs):
        logits = tf.nn.bias_add(inputs, self.bias)
        return logits
    
    
class DenseNoMask(layers.Dense):
    def compute_mask(self, inputs, mask=None):
        return None
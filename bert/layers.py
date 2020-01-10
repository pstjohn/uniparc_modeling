import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback

initializer = lambda: tf.keras.initializers.TruncatedNormal(stddev=0.02)

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

    def __init__(self, units, num_heads, dropout=0.1, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units          # H
        self.num_heads = num_heads  # N
        self.dropout = dropout
        
    def build(self, input_shape):
        """ B, S, N, H - batch, fseq_len, num_heads, size_per_head """
        
        dense_units = self.units * self.num_heads  # N*H
        
        self.query_layer = layers.Dense(
            dense_units, kernel_initializer=initializer(), name='query')
        self.key_layer = layers.Dense(
            dense_units, kernel_initializer=initializer(), name='key')
        self.value_layer = layers.Dense(
            dense_units, kernel_initializer=initializer(), name='value')
        
        self.dropout_layer = layers.Dropout(self.dropout)
        
    def create_attention_mask(self, input_mask):
        mask = tf.cast(input_mask, tf.float32)
        attention_mask = tf.linalg.einsum('aj,ak->ajk', mask, mask)
        return tf.expand_dims(attention_mask, axis=1)  # [B,1,S,S]
    
    def transpose_scores(self, input_tensor):
        input_shape  = tf.shape(input_tensor)
        output_shape = [input_shape[0], input_shape[1], self.num_heads, self.units]
        output_tensor = tf.reshape(input_tensor, output_shape)
        return tf.transpose(a=output_tensor, perm=[0, 2, 1, 3])  # [B,N,S,H]

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0], input_shape[1], self.num_heads * self.units]
        return output_shape  # [B, S, N*H]
        
    def call(self, inputs, mask=None, training=None):
        query = self.transpose_scores(self.query_layer(inputs))  # [B,N,S,H]
        key   = self.transpose_scores(self.key_layer(inputs))    # [B,N,S,H]
        value = self.transpose_scores(self.value_layer(inputs))  # [B,N,S,H]

        # Equation 1 of "Attention is all you need"
        attention_scores = (tf.matmul(query, key, transpose_b=True) 
                            / tf.sqrt(float(self.units)))  # [B,N,S,S]

        # zero out masked values
        attention_mask = self.create_attention_mask(mask)
        attention_scores = attention_scores + (1. - attention_mask) * -10000.0
        
        attention_probs = tf.nn.softmax(attention_scores)  # [B,N,S,S]
        attention_probs = self.dropout_layer(attention_probs, training=training)
        context_layer = tf.matmul(attention_probs, value)  # [B,N,S,S]
        
        input_shape  = tf.shape(inputs)
        output_shape = [input_shape[0], input_shape[1], self.num_heads*self.units]
        context_layer = tf.reshape(context_layer, output_shape)

        return context_layer

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({'units': self.units, 
                       'num_heads': self.num_heads,
                       'dropout': self.dropout})
        return config
    

class Projection(layers.Layer):
    """ Performs a dense layer, dropout, layer norm and residual update """
    def __init__(self, units, dropout=0.1, use_residual=True, **kwargs):
        super(Projection, self).__init__(**kwargs)        
        self.units = units
        self.dropout = dropout        
        self.use_residual = use_residual

    def build(self, input_shape):
        self.dense_layer = layers.Dense(self.units,
                                        kernel_initializer=initializer(),
                                        activation=gelu)
        self.dropout_layer = layers.Dropout(self.dropout)
        self.layer_norm = layers.LayerNormalization()

    def call(self, inputs, training=None, ):
        
        output, residual = inputs if self.use_residual else (inputs, None)
        
        output = self.dense_layer(output)
        output = self.dropout_layer(output, training=training)
        
        if self.use_residual:
            return self.layer_norm(output + residual)
        else:
            return self.layer_norm(output)

    def get_config(self):
        config = super(Projection, self).get_config()
        config.update({'units': self.units,
                       'dropout': self.dropout,
                       'use_residual': self.use_residual})
        return config


class Transformer(layers.Layer):
    """ Performs the multi-headed attention and normalization of a single
    transformer block """

    def __init__(self, num_heads, intermediate_units, dropout=0.1, **kwargs):
        super(Transformer, self).__init__(**kwargs)        
        self.num_heads = num_heads
        self.dropout = dropout
        self.intermediate_units = intermediate_units
        
        
    def build(self, input_shape):

        # Split the model dimension equally amoung attention heads
        d_model = input_shape[-1]
        assert d_model % self.num_heads == 0, \
            f"input dimension {d_model} not divisible by {self.num_heads} "\
            "attention heads"
        
        self.units = d_model // self.num_heads
        
        self.attention_layer = Attention(self.units, self.num_heads, self.dropout)
        
        self.intermediate_layer = layers.Dense(self.intermediate_units,
                                               kernel_initializer=initializer(),
                                               activation=gelu)
        
        self.attention_projection = Projection(d_model, self.dropout,
                                               name='attention_projection')
        self.output_projection = Projection(d_model, self.dropout,
                                            name='output_projection')


    def call(self, inputs, mask=None, training=None):
        
        # Multi-head attention block
        attention_output = self.attention_layer(inputs, mask=mask)
        attention_output = self.attention_projection([attention_output, inputs])
        
        intermediate_values = self.intermediate_layer(attention_output)
        output = self.output_projection([intermediate_values, attention_output])
        return output
    
    def compute_mask(self, inputs, mask=None):
        return mask 

    def get_config(self):
        config = super(Transformer, self).get_config()
        config.update({'num_heads': self.num_heads,
                       'intermediate_units': self.intermediate_units,
                       'dropout': self.dropout})
        return config

    
class Bias(layers.Layer):
    """ Final bias layer added to logits prior to softmax scoring. This layer
    also applys the input mask from the input to mask non-randomized prediction
    targets """

    def build(self, input_shape):
        self.bias = self.add_weight(name='classifier_bias',
                                    dtype=K.floatx(),
                                    shape=[input_shape[-1]],
                                    initializer=tf.zeros_initializer())
        
    def call(self, inputs):
        logits = tf.nn.bias_add(inputs, self.bias)
        return logits


def masked_sparse_categorical_crossentropy(y_true, y_pred):
    """ Computes the mean categorical cross_entropy loss across each batch
    example, where masked or randomized tokens are specified by nonzero entries
    in y_true """

    masked_entries = tf.not_equal(y_true, 0)
    y_true_mask = tf.boolean_mask(y_true, masked_entries)
    y_pred_mask = tf.boolean_mask(y_pred, masked_entries)

    return tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(
        y_true_mask, y_pred_mask, from_logits=True))


def ECE(y_true, y_pred):
    """ Exponentiated cross entropy metric """
    return tf.exp(masked_sparse_categorical_crossentropy(y_true, y_pred))


# class InverseSqrtWarmupSchedule(LearningRateSchedule):
#     """A LearningRateSchedule that uses a linear warmup followed by 
#     inverse square root decay."""

#     def __init__(self, initial_learning_rate=1E-4, warmup_updates=16000,
#                  name=None):
#         super(InverseSqrtWarmupSchedule, self).__init__()
#         self.initial_learning_rate = initial_learning_rate
#         self.warmup_updates = warmup_updates
#         self.name = name

#     def __call__(self, step):
#         with ops.name_scope_v2(self.name or "InverseSqrtWarmupSchedule") as name:
#             initial_learning_rate = ops.convert_to_tensor(
#               self.initial_learning_rate, name="initial_learning_rate")
#             dtype = initial_learning_rate.dtype
#             warmup_updates = math_ops.cast(self.warmup_updates, dtype)
#             global_step = math_ops.cast(step, dtype)
            
    
#             return tf.cond(global_step <= warmup_updates,
#                            lambda: (global_step / warmup_updates) * initial_learning_rate,  # in warmup
#                            lambda: initial_learning_rate * tf.math.sqrt(warmup_updates) / tf.math.sqrt(global_step))

#     def get_config(self):
#         return {
#             "initial_learning_rate": self.initial_learning_rate,
#             "warmup_updates": self.warmup_updates,
#             "name": self.name
#         }

class InverseSquareRootSchedule(Callback):
    def __init__(self, 
                 learning_rate=1E-4,
                 warmup_updates=16000):
        """ Implements the linear learning rate warmup and learning
        rate decay used by google in BERT pretraining """
        
        self.learning_rate = learning_rate
        self.warmup_updates = warmup_updates
        self.decay_factor = learning_rate * warmup_updates**0.5
        
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        
    def on_train_batch_begin(self, batch, logs=None):
        
        global_step = (
            batch + self.current_epoch * self.params['steps'])
        
        # Still in warmup
        if global_step <= self.warmup_updates:
            scheduled_lr = self.learning_rate * (
                global_step / self.warmup_updates)
        
        # Linear decay
        else:
            scheduled_lr = self.decay_factor * global_step**(-0.5)
                        
        K.set_value(self.model.optimizer.lr, scheduled_lr)

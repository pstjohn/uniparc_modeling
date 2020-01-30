import tensorflow as tf

initializer = lambda: tf.keras.initializers.TruncatedNormal(stddev=0.02)

def create_attention_mask(input_shape, input_mask):
    """
    Creates 3D attention. From github.com/kpe/bert-for-tf2
    :param from_shape:  [batch_size, seq_len, ...]
    :param input_mask:  [batch_size, seq_len]
    :return: [batch_size, from_seq_len, seq_len]
    """
    
    mask = tf.cast(tf.expand_dims(input_mask, axis=1), tf.float32)                   # [B, 1, S]
    ones = tf.expand_dims(tf.ones(shape=input_shape[:2], dtype=tf.float32), axis=-1)  # [B, S, 1]
    mask = ones * mask  # broadcast along two dimensions
    
    return tf.expand_dims(mask, axis=1)  # [B, 1, S, S]


def relative_attention_inner(x, y, z, transpose):
    """Relative position-aware dot-product attention inner calculation.
    This batches matrix multiply calculations to avoid unnecessary broadcasting.
    Args:
        x: Tensor with shape [batch_size, heads, length or 1, length or depth].
        y: Tensor with shape [batch_size, heads, length or 1, depth].
        z: Tensor with shape [length or 1, length, depth]
        transpose: Whether to transpose inner matrices of y and z. Should be true if
            last dimension of x is depth, not length.
            
    Returns:
        A Tensor with shape [batch_size, heads, length, length or depth].
            
    From https://github.com/tensorflow/tensor2tensor/
    """

    batch_size = tf.shape(x)[0]
    heads = x.get_shape().as_list()[1]
    length = tf.shape(x)[2]

    # xy_matmul is [batch_size, heads, length or 1, length or depth]
    xy_matmul = tf.matmul(x, y, transpose_b=transpose)
    # x_t is [length or 1, batch_size, heads, length or depth]
    x_t = tf.transpose(x, [2, 0, 1, 3])
    # x_t_r is [length or 1, batch_size * heads, length or depth]
    x_t_r = tf.reshape(x_t, [length, heads * batch_size, -1])
    # x_tz_matmul is [length or 1, batch_size * heads, length or depth]
    x_tz_matmul = tf.matmul(x_t_r, z, transpose_b=transpose)
    # x_tz_matmul_r is [length or 1, batch_size, heads, length or depth]
    x_tz_matmul_r = tf.reshape(x_tz_matmul, [length, batch_size, heads, -1])
    # x_tz_matmul_r_t is [batch_size, heads, length or 1, length or depth]
    x_tz_matmul_r_t = tf.transpose(x_tz_matmul_r, [1, 2, 0, 3])
    return xy_matmul + x_tz_matmul_r_t


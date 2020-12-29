import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.callbacks import Callback

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import ops

import itertools

def masked_sparse_categorical_crossentropy(target, output):
    """ Computes the mean categorical cross_entropy loss across each batch
    example, where masked or randomized tokens are specified by nonzero entries
    in y_true """
    
    axis=-1
        
    output = ops.convert_to_tensor_v2(output)

    if isinstance(output.shape, (tuple, list)):
        output_rank = len(output.shape)
    else:
        output_rank = output.shape.ndims
    if output_rank is not None:
        axis %= output_rank
        if axis != output_rank - 1:
            permutation = list(
                itertools.chain(range(axis), range(axis + 1, output_rank), [axis]))
            output = array_ops.transpose(output, perm=permutation)
    elif axis != -1:
        raise ValueError(
                'Cannot compute sparse categorical crossentropy with `axis={}` on an '
                'output tensor with unknown rank'.format(axis))

    target = math_ops.cast(target, 'int64')

    # Try to adjust the shape so that rank of labels = rank of logits - 1.
    output_shape = array_ops.shape_v2(output)
    target_rank = target.shape.ndims

    update_shape = (
        target_rank is not None and output_rank is not None and
        target_rank != output_rank - 1)
    
    if update_shape:
        target = array_ops.reshape(target, [-1])
        output = array_ops.reshape(output, [-1, output_shape[-1]])

    res = nn.sparse_softmax_cross_entropy_with_logits_v2(
        labels=target, logits=output)

    if update_shape and output_rank >= 3:
        # If our output includes timesteps or spatial dimensions we need to reshape
        res = array_ops.reshape(res, output_shape[:-1])

    
    masked_entries = tf.not_equal(target, 0)
    return tf.reduce_mean(tf.boolean_mask(res, masked_entries))


def ECE(y_true, y_pred):
    """ Exponentiated cross entropy metric """
    return tf.exp(masked_sparse_categorical_crossentropy(y_true, y_pred))


def masked_sparse_categorical_accuracy(y_true, y_pred):
    """Calculates how often predictions matches integer labels.
    You can provide logits of classes as `y_pred`, since argmax of
    logits and probabilities are same.
    Args:
        y_true: Integer ground truth values.
        y_pred: The prediction values.
    Returns:
        Sparse categorical accuracy values.
    """
    y_pred_rank = ops.convert_to_tensor_v2(y_pred).shape.ndims
    y_true_rank = ops.convert_to_tensor_v2(y_true).shape.ndims
    # If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
    if (y_true_rank is not None) and (y_pred_rank is not None) and (len(
            K.int_shape(y_true)) == len(K.int_shape(y_pred))):
        y_true = array_ops.squeeze(y_true, [-1])
    y_pred = math_ops.argmax(y_pred, axis=-1)

    # If the predicted output and actual output types don't match, force cast them
    # to match.
    if K.dtype(y_pred) != K.dtype(y_true):
        y_pred = math_ops.cast(y_pred, K.dtype(y_true))
        
    masked_entries = tf.not_equal(y_true, 0)
    acc = math_ops.cast(math_ops.equal(y_true, y_pred), K.floatx())
    
    return tf.reduce_mean(tf.boolean_mask(acc, masked_entries))

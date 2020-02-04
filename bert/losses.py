import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.callbacks import Callback


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

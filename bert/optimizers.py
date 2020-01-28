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
        

class BertLinearSchedule(Callback):
    def __init__(self, 
                 learning_rate=1E-4,
                 warmup_steps=10000,
                 total_steps=1000000,
                 write_summary=False,
                 update_freq=50,
                ):
        """ Implements the linear learning rate warmup and linear learning rate
        decay used by google in BERT pretraining """
        
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps - warmup_steps    
        self.write_summary = write_summary
        self.update_freq = update_freq
        
        
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        
    def on_train_batch_begin(self, batch, logs=None):
        
        global_step = (
            batch + self.current_epoch * self.params['steps'])
        
        if global_step % self.update_freq is not 0:
            # Only log / update every update_freq steps
            return
            
        # Still in warmup
        if global_step < self.warmup_steps:
            scheduled_lr = self.learning_rate * (
                global_step / self.warmup_steps)
        
        # Linear decay to zero at total_steps
        else:
            scheduled_lr = self.learning_rate * (
                1 - ((global_step - self.warmup_steps) / self.total_steps))
            
        if self.write_summary:
            tf.summary.scalar('learning rate', data=scheduled_lr,
                              step=global_step)
            
        K.set_value(self.model.optimizer.lr, scheduled_lr)

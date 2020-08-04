import tensorflow as tf
from tensorflow.python.module.module import camel_to_snake
from tensorflow.python import math_ops

class TreeNorm(tf.keras.layers.Layer):
    """ Multiply each GO score by the scores of its ancestor nodes to normalize
    tree-valued predictions to be monotonically decreasing with depth.
    
    For some reason, I have to use `unsorted_segment_prod` here:
    https://github.com/tensorflow/tensorflow/issues/41090
    
    """
    def __init__(self, segments, ids, **kwargs):
        super(TreeNorm, self).__init__(**kwargs)        
        self.segments = segments
        self.ids = ids
    
    def call(self, inputs):
        return tf.transpose(tf.math.segment_min(
            tf.gather(tf.transpose(inputs), self.ids), self.segments))
    
    def compute_output_shape(self, input_shape):
        return input_shape
    

class LogitMixin:
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.nn.sigmoid(y_pred)
        super().update_state(y_true, y_pred, sample_weight)

        
class OntSplitMixin:
    def __init__(self, ont, root_index, *args, **kwargs):
        root_node_index = ont.get_head_node_indices()[root_index]
        self.term_name = ont.G.nodes[ont.term_index[root_node_index]]['name']
        self.ontology_index = tf.constant(ont.terms_to_indices(
            ont.get_descendants(ont.term_index[root_node_index])))
        
        name = f'{camel_to_snake(type(self).__name__)}_{self.term_name}'
        kwargs['name'] = name
        
        super().__init__(*args, **kwargs)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.gather(y_true, self.ontology_index, axis=-1)
        y_pred = tf.gather(y_pred, self.ontology_index, axis=-1)
        if sample_weight is not None:
            sample_weight = tf.gather(sample_weight, self.ontology_index)
        super().update_state(y_true, y_pred, sample_weight)

        
class Fmax(tf.keras.metrics.AUC):
    def result(self):
        precision = math_ops.div_no_nan(
            self.true_positives, self.true_positives + self.false_positives)
        recall = math_ops.div_no_nan(
            self.true_positives, self.true_positives + self.false_negatives)
        return tf.reduce_max(2 * precision * recall / (precision + recall))    

    
class LogitPrecision(LogitMixin, tf.keras.metrics.Precision):
    pass

class LogitRecall(LogitMixin, tf.keras.metrics.Recall):
    pass

class LogitAUC(LogitMixin, tf.keras.metrics.AUC):
    pass

class LogitFmax(LogitMixin, Fmax):
    pass

class LogitSplitFmax(OntSplitMixin, LogitMixin, Fmax):
    pass

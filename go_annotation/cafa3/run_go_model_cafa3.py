import os
import argparse
import shutil
import numpy as np
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()

from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

import sys
sys.path.append('../..')

from bert.dataset import encode
from bert.model import create_model
from bert.go import TreeNorm, Ontology
from bert.go.layers import LogitSplitFmax

parser = argparse.ArgumentParser(description='GO model training')
parser.add_argument('--modelName', default='go-model',
                    help='model name for directory saving')
parser.add_argument('--batchSize', type=int, default=24, 
                    help='batch size')
parser.add_argument('--warmup', type=int, default=10000, 
                    help='warmup steps')
parser.add_argument('--epochs', type=int, default=10, 
                    help='total steps')
parser.add_argument('--lr', type=float, default=1E-4, 
                    help='initial learning rate')
parser.add_argument('--checkpointDir', default=None, 
                    help='directory for pretrained model')
parser.add_argument('--scratchDir', default=None, 
                    help='Directory for tensorboard logs and checkpoints')
parser.add_argument('--sequenceLength', type=int, default=1024, 
                    help='Protein AA sequence length')
parser.add_argument('--stepsPerEpoch', type=int, default=500, 
                    help='steps per epoch')
parser.add_argument('--validationSteps', type=int, default=25, 
                    help='validation steps')

arguments = parser.parse_args()
print(arguments)

cafa_code_dir = '/ccs/home/pstjohn/uniparc_modeling/go_annotation/cafa3'
ont = Ontology(obo_file=os.path.join(cafa_code_dir, 'go_cafa3.obo.gz'))
print(ont.total_nodes)

## Create the dataset iterators
def parse_example(example):
    parsed = tf.io.parse_single_example(example, features={
        'sequence': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'annotation': tf.io.FixedLenFeature([], tf.string, default_value=''),
    })
    
    sequence = encode(parsed['sequence'], arguments.sequenceLength)
    annotation = tf.io.parse_tensor(parsed['annotation'], out_type=tf.int64)
    
    return sequence, annotation

cafa3_dir = '/gpfs/alpine/bie108/proj-shared/cafa3/'
train_dataset = tf.data.TFRecordDataset(
    os.path.join(cafa3_dir, 'tfrecords', 'go_train.tfrecord.gz'),
    compression_type='GZIP', num_parallel_reads=tf.data.experimental.AUTOTUNE)\
    .map(parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    .repeat().shuffle(buffer_size=5000)\
    .padded_batch(batch_size=arguments.batchSize,
                  padded_shapes=(([arguments.sequenceLength], [ont.total_nodes])))\
    .prefetch(tf.data.experimental.AUTOTUNE)

valid_dataset = tf.data.TFRecordDataset(
    os.path.join(cafa3_dir, 'tfrecords', 'go_valid.tfrecord.gz'),
    compression_type='GZIP', num_parallel_reads=tf.data.experimental.AUTOTUNE)\
    .map(parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    .repeat().shuffle(buffer_size=5000)\
    .padded_batch(batch_size=arguments.batchSize,
                  padded_shapes=(([arguments.sequenceLength], [ont.total_nodes])))\
    .prefetch(tf.data.experimental.AUTOTUNE)

initial_bias = np.load(os.path.join(cafa3_dir, 'tfrecords', 'bias.npy'))

## Load the original model
# checkpoint_dir = '/ccs/home/pstjohn/member_work/uniparc_checkpoints/12_layer_relative_adam_20200625.186949'
# tf.train.latest_checkpoint(arguments.checkpointDir)

with strategy.scope():   

    dimension = 768
    model = create_model(model_dimension=dimension,
                         transformer_dimension=dimension * 4,
                         num_attention_heads=dimension // 64,
                         num_transformer_layers=12,
                         vocab_size=24,
                         dropout_rate=0.0,
                         max_relative_position=64,
                         attention_type='relative')

    model.load_weights(tf.train.latest_checkpoint(arguments.checkpointDir)).expect_partial()

    ## Append the GO annotations
    final_embedding = model.layers[-2].input
    raw_residue_predictions = tf.keras.layers.Dense(
        ont.total_nodes,
        bias_initializer=tf.keras.initializers.Constant(initial_bias)
    )(final_embedding)
    
    protein_predictions = tf.keras.layers.GlobalMaxPooling1D()(raw_residue_predictions)
    
    segments, ids = zip(*ont.iter_ancestor_array())
    treenorm = TreeNorm(segments, ids)
    normed = treenorm(protein_predictions)
    
    go_model = tf.keras.Model(model.inputs, normed)

    go_model.summary()

    optimizer = tf.keras.optimizers.Adam(arguments.lr)

    metrics = [
        LogitSplitFmax(ont, 0),
        LogitSplitFmax(ont, 1),
        LogitSplitFmax(ont, 2),
    ]

    go_model.compile(
       loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
       metrics=metrics,
       optimizer=optimizer)


checkpoint_dir = os.path.join(arguments.scratchDir, arguments.modelName)
logdir = os.path.join(arguments.scratchDir, 'tblogs', arguments.modelName)

# Make the checkpoint directory
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Make sure this script is available later
shutil.copy(__file__, checkpoint_dir)

# file_writer = tf.summary.create_file_writer(logdir + "/metrics")
# file_writer.set_as_default()


callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "weights.{epoch:03d}-{val_loss:.3f}"),
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        monitor='val_loss'),
    
    tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=0,
        write_graph=False,
        update_freq='epoch',
        profile_batch=0,
        embeddings_freq=0)
]

if __name__ == "__main__":

    go_model.fit(
        train_dataset,
        validation_data=valid_dataset,
        verbose=1,
        epochs=arguments.epochs,
        steps_per_epoch=arguments.stepsPerEpoch,
        validation_steps=arguments.validationSteps,
        callbacks=callbacks)


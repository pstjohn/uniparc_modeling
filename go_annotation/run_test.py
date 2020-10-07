import os
from functools import partial
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

from bert.dataset import encode
from bert.model import create_model
from bert.go import TreeNorm
from bert.go import Ontology

ont = Ontology()
swissprot_dir = '/gpfs/alpine/bie108/proj-shared/swissprot/'

swissprot = pd.read_parquet(os.path.join(swissprot_dir, 'parsed_swissprot_uniref_clusters.parquet'))
go_terms = pd.read_parquet(os.path.join(swissprot_dir, 'swissprot_quickgo.parquet'))
swissprot_annotated = swissprot[swissprot.accession.isin(go_terms['GENE PRODUCT ID'].unique())]
swissprot_annotated = swissprot_annotated[swissprot_annotated.length < 10000]

test = np.load('uniref50_split.npz', allow_pickle=True)['test']
swissprot_test = swissprot_annotated[swissprot_annotated['UniRef50 ID'].isin(test)]

test_data = tf.data.Dataset.from_tensor_slices(swissprot_test.sequence.values)\
    .map(partial(encode, max_sequence_length=1024), num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    .padded_batch(batch_size=16,
                  padded_shapes=([1024]))\
    .prefetch(tf.data.experimental.AUTOTUNE)


dimension = 768
model = create_model(model_dimension=dimension,
                     transformer_dimension=dimension * 4,
                     num_attention_heads=dimension // 64,
                     num_transformer_layers=12,
                     vocab_size=24,
                     dropout_rate=0.0,
                     max_relative_position=64,
                     attention_type='relative')

## Append the GO annotations
final_embedding = model.layers[-2].input
raw_residue_predictions = tf.keras.layers.Dense(ont.total_nodes)(final_embedding)

protein_predictions = tf.keras.layers.GlobalMaxPooling1D()(raw_residue_predictions)

segments, ids = zip(*ont.iter_ancestor_array())
treenorm = TreeNorm(segments, ids)
normed = treenorm(protein_predictions)

go_model_sigmoid = tf.keras.Model(model.inputs, tf.nn.sigmoid(normed))

checkpoint = tf.train.latest_checkpoint(
    '/ccs/home/pstjohn/member_work/uniparc_checkpoints/go_finetuning_new_split_1024_ont1.258061')
go_model_sigmoid.load_weights(checkpoint).expect_partial()

if __name__ == '__main__':

    y_true = []
    y_pred = []

    for batch in tqdm(valid_dataset):
        y_true += [batch[1].numpy()]
        y_pred += [go_model_sigmoid.predict_on_batch(batch)]

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    np.savez_compressed('/ccs/home/pstjohn/member_work/valid.258061.npz', 
                        y_true=y_true,
                        y_pred=y_pred)

    head_nodes = {ont.G.nodes[ont.term_index[index]]['name']: index for index in ont.get_head_node_indices()}
    from sklearn.metrics import precision_recall_curve

    y_true_bp = y_true[:, ont.terms_to_indices(ont.get_descendants(ont.term_index[head_nodes['biological_process']]))]
    y_pred_bp = y_pred[:, ont.terms_to_indices(ont.get_descendants(ont.term_index[head_nodes['biological_process']]))]

    y_true_mf = y_true[:, ont.terms_to_indices(ont.get_descendants(ont.term_index[head_nodes['molecular_function']]))]
    y_pred_mf = y_pred[:, ont.terms_to_indices(ont.get_descendants(ont.term_index[head_nodes['molecular_function']]))]

    y_true_cc = y_true[:, ont.terms_to_indices(ont.get_descendants(ont.term_index[head_nodes['cellular_component']]))]
    y_pred_cc = y_pred[:, ont.terms_to_indices(ont.get_descendants(ont.term_index[head_nodes['cellular_component']]))]

    precision_bp, recall_bp, thresholds_bp = precision_recall_curve(y_true_bp.flatten(), y_pred_bp.flatten())
    precision_mf, recall_mf, thresholds_mf = precision_recall_curve(y_true_mf.flatten(), y_pred_mf.flatten())
    precision_cc, recall_cc, thresholds_cc = precision_recall_curve(y_true_cc.flatten(), y_pred_cc.flatten())

    np.savez_compressed('/ccs/home/pstjohn/member_work/valid_pr.258061.npz', 
                        precision_bp=precision_bp,
                        recall_bp=recall_bp,
                        thresholds_bp=thresholds_bp,
                        precision_mf=precision_mf,
                        recall_mf=recall_mf,
                        thresholds_mf=thresholds_mf,
                        precision_cc=precision_cc,
                        recall_cc=recall_cc,
                        thresholds_cc=thresholds_cc)
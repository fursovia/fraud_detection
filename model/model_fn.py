"""
Here we define architecture, loss, metrics and so on
"""

import tensorflow as tf
from tensorflow.nn.rnn_cell import GRUCell
import numpy as np
import gc
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def load_word2vec(filename, vocab_path):
    print('Starting the word2vec initialization...')

    tokens = []
    with open(vocab_path, 'r') as file:
        for line in file:
            tokens.append(line.replace('\n', ''))

    embeddings_index = dict()
    with open(filename, 'r') as file:

        file.readline()  # drop first line

        for line in file:
            values = line.split()
            token = values[0]
            if token in tokens:
                coefs = np.array(values[1:], dtype=np.float64)
                embeddings_index[token] = coefs

    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    vocab_size = len(tokens) + 1
    embedding_size = list(embeddings_index.values())[1].shape[0]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (vocab_size, embedding_size))

    count = 0
    for i, token in enumerate(tokens):
        embedding_vector = embeddings_index.get(token)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            count += 1

    percent_initialized = 100 * (count + 1) / vocab_size
    print(f'Percenct initialized: {percent_initialized}')

    del embeddings_index, all_embs
    gc.collect()

    return embedding_matrix, embedding_size


def get_architecture(params, embeddings, meta_features=None):

    name = params['arch_name']

    if name == 'swem_max':
        out = tf.reduce_max(embeddings, axis=1)

        out = tf.layers.dense(out, params['units'][name])
        out = tf.nn.relu(out)

    elif name == 'swem_aver':
        out = tf.reduce_mean(embeddings, axis=1)

        out = tf.layers.dense(out, params['units'][name])
        out = tf.nn.relu(out)

    elif name == 'swem_max_features':

        with tf.name_scope('features'):
            meta_features = tf.layers.dense(meta_features, 5)

        with tf.name_scope('treatments'):
            out = tf.reduce_max(embeddings, axis=1)

            out = tf.layers.dense(out, params['units'][name][0])
            out = tf.nn.relu(out)

            out = tf.layers.dense(out, params['units'][name][1])
            out = tf.nn.relu(out)

        with tf.name_scope('concat'):
            out = tf.concat([out, meta_features], axis=-1)

    elif name == 'gru':
        all_states, last_state = tf.nn.dynamic_rnn(
            cell=GRUCell(params['units'][name][0]),
            inputs=embeddings,
            dtype=tf.float64
        )

        out = tf.reduce_mean(all_states, axis=1)

        out = tf.layers.dense(out, params['units'][name][1])
        out = tf.nn.relu(out)

    elif name == 'gru_feats':

        with tf.name_scope('features'):
            meta_features = tf.layers.dense(meta_features, 5)

        with tf.name_scope('treatments'):
            all_states, last_state = tf.nn.dynamic_rnn(
                cell=GRUCell(params['units'][name][0]),
                inputs=embeddings,
                dtype=tf.float64)

            out = tf.reduce_mean(all_states, axis=1)

            out = tf.layers.dense(out, params['units'][name][1])
            out = tf.nn.relu(out)

        with tf.name_scope('concat'):
            out = tf.concat([out, meta_features], axis=-1)

    else:
        raise NotImplemented(f'{name} is not implemented')

    out = tf.layers.dense(out, 2)

    return out


def build_model(emb_matrix, features, params):

    embeddings = tf.nn.embedding_lookup(emb_matrix, features['treatments'])

    meta_features = features['meta_features'] if 'meta_features' in features else None

    out = get_architecture(params, embeddings, meta_features)

    return out


def model_fn(features, labels, mode, params):
    # is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    with tf.variable_scope('embeddings', reuse=tf.AUTO_REUSE):

        # TODO: fix for word2vec initialization
        emb_dim = params['emb_dim']

        emb_matrix = tf.get_variable('treatments_embeddings',
                                     shape=[params['num_treatments'], emb_dim],
                                     dtype=tf.float64)

    def init_fn(scaffold, sess):

        if params['use_pretrained']:
            initial_value, _ = load_word2vec(filename=params['word2vec_filename'],
                                             vocab_path=params['treatments_vocab_path'])
        else:
            initial_value = np.random.normal(0, 0.001, (params['num_treatments'], emb_dim)).astype(np.float64)

        sess.run(emb_matrix.initializer, {emb_matrix.initial_value: initial_value})

    scaffold = tf.train.Scaffold(init_fn=init_fn)

    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        logits = build_model(emb_matrix, features, params)

    preds = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'logits': logits, 'preds': preds}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    weights = tf.multiply(tf.add(tf.to_float(labels), 1), 0.7)  # 0.7 and 1.4 for 0 and 1

    onehot_labels = tf.one_hot(labels, depth=2)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits, weights=weights)

    accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(logits, axis=1), labels)))

    if mode == tf.estimator.ModeKeys.EVAL:
        with tf.variable_scope('metrics'):

            roc_auc = tf.metrics.auc(labels=onehot_labels, predictions=preds)

            eval_metric_ops = {'accuracy': tf.metrics.mean(accuracy),
                               'roc_auc': roc_auc}

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    optimizer_fn = tf.train.AdamOptimizer(params['learning_rate'])

    global_step = tf.train.get_global_step()

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=params['learning_rate'],
        optimizer=optimizer_fn,
        name='optimizer')

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, scaffold=scaffold)

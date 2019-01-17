"""
Here we define architecture, loss, metrics and so on
"""

import tensorflow as tf
import numpy as np
import gc


def load_word2vec(filename, vocab_path):
    print('Starting the word2vec initialization...')

    tokens = []
    with open(vocab_path, 'r') as file:
        for line in file:
            tokens.append(line.replace('\n', ''))

    embeddings_index = dict()
    with open(filename, 'r') as file:
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


def get_architecture(name, embeddings, meta_features=None):

    if name == 'swem_max':
        out = tf.reduce_max(embeddings, axis=1)

        out = tf.layers.dense(out, 128)
        out = tf.nn.relu(out)

        out = tf.layers.dense(out, 2)

    elif name == 'swem_aver':
        out = tf.reduce_mean(embeddings, axis=1)

        out = tf.layers.dense(out, 128)
        out = tf.nn.relu(out)

        out = tf.layers.dense(out, 2)

    elif name == 'swem_max_features':
        out = tf.reduce_max(embeddings, axis=1)

        out = tf.layers.dense(out, 128)
        out = tf.nn.relu(out)

        out = tf.layers.dense(out, 64)
        out = tf.concat([out, meta_features], axis=-1)

        out = tf.layers.dense(out, 2)

    else:
        raise NotImplemented(f'{name} is not implemented')

    return out


def build_model(features, params):

    if params['use_pretrained']:
        loaded_weights, emb_dim = load_word2vec(filename=params['word2vec_filename'],
                                                vocab_path=params['treatments_vocab_path'])
        initializer = tf.constant_initializer(loaded_weights)
    else:
        initializer = tf.initializers.truncated_normal(stddev=0.001)
        emb_dim = params['emb_dim']

    emb_matrix = tf.get_variable('treatments_embeddings',
                                 initializer=initializer,
                                 shape=[params['num_treatments'], emb_dim],
                                 dtype=tf.float64)

    embeddings = tf.nn.embedding_lookup(emb_matrix, features['treatments'])

    meta_features = features['meta_features'] if 'meta_features' in features else None

    out = get_architecture(params['arch_name'], embeddings, meta_features)

    return out


def model_fn(features, labels, mode, params):
    # is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    with tf.variable_scope('model'):
        logits = build_model(features, params)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'logits': logits}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    weights = tf.multiply(tf.add(tf.to_float(labels), 1), 0.7)  # 0.7 and 1.4 for 0 and 1

    onehot_labels = tf.one_hot(labels, depth=2)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits, weights=weights)

    accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(logits, axis=1), labels)))

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

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

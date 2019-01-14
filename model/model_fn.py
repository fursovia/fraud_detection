"""
Here we define architecture, loss, metrics and so on
"""

import tensorflow as tf


def get_architecture(embeddings, name):

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

    else:
        raise NotImplemented(f'{name} is not implemented')

    return out


def build_model(features, params):

    emb_matrix = tf.get_variable('treatments_embeddings',
                                 shape=[params['num_treatments'], 256],
                                 dtype=tf.float64)

    embeddings = tf.nn.embedding_lookup(emb_matrix, features['treatments'])

    out = get_architecture(embeddings, params['arch_name'])

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

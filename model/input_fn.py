"""
Here we define inputs to the model
"""

import tensorflow as tf


def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'treatments': tf.FixedLenFeature([], tf.string),
            'treatment_type': tf.FixedLenFeature([], tf.string),
            'num_treatments': tf.FixedLenFeature([], tf.int64)
        })

    label = features.pop('label')

    return features, label


def build_vocab(file_name):
    # there is a difference between <UNK> and not in vocab
    tokens = tf.contrib.lookup.index_table_from_file(
        file_name,
        num_oov_buckets=1,
        delimiter='\n',
        name='vocab'
    )

    return tokens


def vectorize(string, vocab):
    splitted = tf.string_split([string]).values
    vectorized = vocab.lookup(splitted)
    vectorized = vectorized[:20]
    return vectorized


def input_fn(data_path, params, train_time=True):
    dataset = tf.data.TFRecordDataset(data_path)
    vocab = build_vocab(params['treatments_vocab_path'])

    treat_pad_word = vocab.lookup(tf.constant('<PAD>'))
    fake_padding = tf.constant(9999, dtype=tf.int64)

    if train_time:
        dataset = dataset.shuffle(100000)
        dataset = dataset.repeat(params['num_epochs'])

    dataset = dataset.map(decode)
    dataset = dataset.map(lambda feats, labs: (vectorize(feats['treatments'], vocab), labs))

    padded_shapes = (tf.TensorShape([params['seq_len']]), tf.TensorShape([]))
    padding_values = (treat_pad_word, fake_padding)

    dataset = dataset.padded_batch(
        batch_size=params['batch_size'], padded_shapes=padded_shapes, padding_values=padding_values
    )

    dataset = dataset.map(lambda treats, y: ({'treatments': treats}, y))
    dataset = dataset.prefetch(buffer_size=None)

    return dataset
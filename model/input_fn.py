"""
Here we define inputs to the model
"""

import sys
sys.path.append('..')
import tensorflow as tf
import pandas as pd
from data_prep import META_FEATURES


def build_vocab(file_name, vocab_size=None):
    # there is a difference between <UNK> and not in vocab
    tokens = tf.contrib.lookup.index_table_from_file(
        file_name,
        vocab_size=vocab_size,
        num_oov_buckets=1,
        delimiter='\n',
        name='vocab'
    )

    return tokens


def vectorize(string, vocab, seq_len):
    splitted = tf.string_split([string]).values
    vectorized = vocab.lookup(splitted)
    vectorized = vectorized[:seq_len]
    return vectorized


def input_fn(data_path, params, train_time=True):
    data = pd.read_csv(data_path)
    num_features = len(META_FEATURES)

    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'treatments': data['treatments'].values,
            'meta_features': data[META_FEATURES].values,
        },
        data['target'].values
    ))

    vocab = build_vocab(params['treatments_vocab_path'], params['num_treatments'] - 1)

    treat_pad_word = vocab.lookup(tf.constant('<PAD>'))
    fake_padding1 = tf.constant(9999, dtype=tf.float64)
    fake_padding2 = tf.constant(9999, dtype=tf.int64)

    if train_time:
        dataset = dataset.shuffle(params['train_size'])
        dataset = dataset.repeat(params['num_epochs'])

    dataset = dataset.map(lambda feats, labs: (vectorize(feats['treatments'], vocab, params['seq_len']),
                                               tf.cast(feats['meta_features'], dtype=tf.float64),
                                               labs))

    # 5 is the number of features
    padded_shapes = (tf.TensorShape([params['seq_len']]), tf.TensorShape([num_features]), tf.TensorShape([]))
    padding_values = (treat_pad_word, fake_padding1, fake_padding2)

    dataset = dataset.padded_batch(
        batch_size=params['batch_size'], padded_shapes=padded_shapes, padding_values=padding_values
    )

    dataset = dataset.map(lambda treats, feats,  y: ({'treatments': treats, 'meta_features': feats}, y))
    dataset = dataset.prefetch(buffer_size=None)

    return dataset

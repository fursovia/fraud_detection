"""
New features and data cleaning
"""

import tensorflow as tf
import pandas as pd
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import argparse

from model.utils import save_vocab_to_txt_file

parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--data_dir', default='data')
parser.add_argument('-s', '--sample', action='store_true')
parser.add_argument('-tmf', '--treat_min_freq', type=int, default=5)

parser.set_defaults(sample=False)

META_FEATURES = ['amount', 'age', 'sex', 'ins_type', 'speciality']


cols_mapping = {
    'ID': 'id',
    'KORREKTUR': 'adj',
    'RECHNUNGSBETRAG': 'amount',
    'ALTER': 'age',
    'GESCHLECHT': 'sex',
    'VERSICHERUNG': 'ins_type',
    'FACHRICHTUNG': 'speciality',  # why only 0/1 ?
    'NUMMER': 'treatment',
    'NUMMER_KAT': 'treatment_type',
    'TYP': 'billing_type',
    'ANZAHL': 'num_treatments',
    'FAKTOR': 'factor',
    'BETRAG': 'cost',
    'ART': 'cost_type',
    'LEISTUNG': 'ben_type'
}


def update_vocabulary(tokens, counter):
    counter.update(tokens)


def create_vocab(pd_series, min_count=0):
    vocabulary = Counter()
    _ = pd_series.apply(lambda x: update_vocabulary(x.split(), vocabulary))
    vocabulary = [tok for tok, count in vocabulary.most_common() if count >= min_count]
    vocabulary.insert(0, '<PAD>')

    if '<UNK>' not in vocabulary:
        vocabulary.insert(1, '<UNK>')

    vocab_length = len(vocabulary)
    return vocabulary, vocab_length


def get_treatment_seq(df):
    # TODO: use num treatments to duplicate treatments
    print('df shape =', df.shape)
    unique_ids = df['id'].unique()
    unique_targets = df[['id', 'target']].drop_duplicates()

    targets = {row['id']: row['target'] for _, row in unique_targets.iterrows()}

    treatments_seq = {id_: '' for id_ in unique_ids}
    treatments_types_seq = {id_: '' for id_ in unique_ids}

    for idx, row in tqdm(df.iterrows()):
        curr_id = row['id']
        curr_treat = row['treatment']
        curr_treat_type = row['treatment_type']

        treatments_seq[curr_id] = treatments_seq[curr_id] + ' ' + curr_treat
        treatments_types_seq[curr_id] = treatments_types_seq[curr_id] + ' ' + curr_treat_type

    data_dict = {id_: [] for id_ in unique_ids}

    for id_ in unique_ids:
        data_dict[id_].extend([id_, treatments_seq[id_], treatments_types_seq[id_], targets[id_]])

    sequences = pd.DataFrame.from_dict(data_dict, orient='index', columns=['id', 'treatments', 'types', 'target'])
    sequences = sequences.reset_index(drop=True)

    return sequences


def get_meta_features(df):

    meta_features = META_FEATURES + ['id']

    features = df.loc[~df['id'].duplicated(), meta_features]

    return features


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_tfrecords_features(dfrow):
    target = dfrow['target']
    treatments, treatment_type = dfrow['treatments'], dfrow['types']
    meta_features = dfrow[META_FEATURES].tolist()
    num_treatments = len(treatments.split())

    features = {'label': _int64_feature(target),
                'treatments': _bytes_feature(tf.compat.as_bytes(treatments)),
                'treatment_type': _bytes_feature(tf.compat.as_bytes(treatment_type)),
                'num_treatments': _int64_feature(num_treatments),
                'meta_features': _float_feature(meta_features)}

    return features


def convert_to_records(dframe, name, save_to):

    filename = os.path.join(save_to, name + '.tfrecords')
    print('Writing {} ...'.format(filename))

    with tf.python_io.TFRecordWriter(filename) as writer:
        for idx, row in tqdm(dframe.iterrows()):
            input_features = get_tfrecords_features(row)
            features = tf.train.Features(feature=input_features)
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())


if __name__ == '__main__':

    args = parser.parse_args()

    data1 = pd.read_csv(os.path.join(args.data_dir, 'arzta_daten_anonym1.csv'), sep=';')
    data2 = pd.read_csv(os.path.join(args.data_dir, 'arzta_daten_anonym2.csv'), sep=';')
    data3 = pd.read_csv(os.path.join(args.data_dir, 'arzta_daten_anonym3.csv'), sep=';')
    data4 = pd.read_csv(os.path.join(args.data_dir, 'arzta_daten_anonym4.csv'), sep=';')

    data = pd.concat([data1, data2, data3, data4])

    columns_comma = ['RECHNUNGSBETRAG', 'FAKTOR', 'BETRAG', 'ALTER', 'KORREKTUR']

    data[columns_comma] = data[columns_comma].apply(lambda x: x.str.replace(',', '.'))

    for column in columns_comma:
        data[column] = pd.to_numeric(data[column], downcast='float')

    data = data.rename(columns=cols_mapping)

    data['target'] = data['adj'].astype(bool).astype(int)
    data = data.drop(columns=['adj'])

    data['treatment'] = data['treatment'].fillna(value='<UNK>')
    data['treatment_type'] = data['treatment_type'].fillna(value='<UNK>')

    if args.sample:
        nrows = 10000
    else:
        nrows = None

    data.to_csv(os.path.join(args.data_dir, 'data.csv'), index=False)
    data = data[:nrows]

    # initial data is always at subfolder
    # data = pd.read_csv(os.path.join(args.data_dir.split(sep='/')[0], 'data.csv'), nrows=nrows)

    sequences = get_treatment_seq(data)
    print('Full data size =', sequences.shape)

    features = get_meta_features(data)

    df = pd.merge(features, sequences, on='id')

    # TODO: drop examples with no treatments (based on min treatment frequency)

    train, valid = train_test_split(df, stratify=df['target'], test_size=0.1, random_state=24)

    df.to_csv(os.path.join(args.data_dir, 'full.csv'), index=False)
    train.to_csv(os.path.join(args.data_dir, 'train.csv'), index=False)
    valid.to_csv(os.path.join(args.data_dir, 'eval.csv'), index=False)

    convert_to_records(df, 'full', args.data_dir)
    convert_to_records(train, 'train', args.data_dir)
    convert_to_records(valid, 'eval', args.data_dir)

    treatments_vocabulary, treatments_count = create_vocab(df['treatments'], min_count=args.treat_min_freq)
    print(f'Number of unique treatments left = {treatments_count}')

    save_vocab_to_txt_file(treatments_vocabulary, os.path.join(args.data_dir, 'treatments.txt'))

"""
New features and data cleaning
"""

import pandas as pd
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import argparse
from model.utils import save_vocab_to_txt_file

parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--data_dir', default='data')
parser.add_argument('-tmf', '--treat_min_freq', type=int, default=1)
parser.add_argument('--random_split', action='store_true')
parser.add_argument('-s', '--sample', action='store_true')

parser.set_defaults(sample=False)
parser.set_defaults(random_split=True)

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


def save_string(text, path):
    with open(path, 'w') as file:
        file.write(text)


if __name__ == '__main__':

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    data1 = pd.read_csv(os.path.join('data', 'arzta_daten_anonym1.csv'), sep=';')
    data2 = pd.read_csv(os.path.join('data', 'arzta_daten_anonym2.csv'), sep=';')
    data3 = pd.read_csv(os.path.join('data', 'arzta_daten_anonym3.csv'), sep=';')
    data4 = pd.read_csv(os.path.join('data', 'arzta_daten_anonym4.csv'), sep=';')

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

    sequences = get_treatment_seq(data)
    print('Full data size =', sequences.shape)

    features = get_meta_features(data)

    df = pd.merge(features, sequences, on='id')

    if args.random_split:
        train, valid = train_test_split(df, stratify=df['target'], test_size=0.3, random_state=24)
    else:
        train = df[:int(0.7 * len(df))]
        valid = df[int(0.7 * len(df)):]

    all_treatments = ' '.join(df['treatments'].tolist()).strip()
    save_string(all_treatments, os.path.join(args.data_dir, 'all_treatments.txt'))

    df.to_csv(os.path.join(args.data_dir, 'full.csv'), index=False)
    train.to_csv(os.path.join(args.data_dir, 'train.csv'), index=False)
    valid.to_csv(os.path.join(args.data_dir, 'eval.csv'), index=False)

    treatments_vocabulary, treatments_count = create_vocab(df['treatments'], min_count=args.treat_min_freq)
    print(f'Number of unique treatments left = {treatments_count}')

    save_vocab_to_txt_file(treatments_vocabulary, os.path.join(args.data_dir, 'treatments.txt'))

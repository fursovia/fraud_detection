"""
xgboost
"""

import os
from xgboost import XGBClassifier
import argparse
import pandas as pd
import numpy as np
from model.utils import save_dict_to_yaml, get_yaml_config
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from data_prep import META_FEATURES
from scipy import sparse


parser = argparse.ArgumentParser()

parser.add_argument('-dd', '--data_dir', default='data')
parser.add_argument('-md', '--model_dir', default='experiments')
parser.add_argument('-v', '--vect', choices=['bow', 'tfidf'])
parser.add_argument('-f', '--features', action='store_true')

parser.set_defaults(vect='bow')
parser.set_defaults(features=False)


if __name__ == '__main__':

    args = parser.parse_args()

    train = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
    val = pd.read_csv(os.path.join(args.data_dir, 'eval.csv'))

    with open(os.path.join(args.data_dir, 'treatments.txt'), 'r') as f:
        vocab = {}

        c = 0
        for line in f:
            vocab[line.rstrip('\n')] = c
            c += 1

    if args.vect == 'bow':
        cv = CountVectorizer(vocabulary=vocab, lowercase=False)
    elif args.vect == 'tfidf':
        cv = TfidfVectorizer(vocabulary=vocab, lowercase=False)
    else:
        raise NotImplemented('No such vectorizer')

    cv.fit(train['treatments'])

    x_train = cv.transform(train['treatments'])
    y_train = train['target'].values

    x_eval = cv.transform(val['treatments'])
    y_eval = val['target'].values

    print('Data shapes:', x_train.shape, x_eval.shape)

    if args.features:
        print('Adding Features...')
        x_train = x_train.todense()
        tr_feats = train[META_FEATURES].values
        x_train = np.hstack([x_train, tr_feats])
        x_train = sparse.csr_matrix(x_train)

        x_eval = x_eval.todense()
        ev_feats = val[META_FEATURES].values
        x_eval = np.hstack([x_eval, ev_feats])
        x_eval = sparse.csr_matrix(x_eval)

        print('Data shapes:', x_train.shape, x_eval.shape)

    print('Training...')
    xgb = XGBClassifier()
    xgb.fit(x_train, y_train)

    probs = xgb.predict_proba(x_eval)

    np.save(os.path.join(args.model_dir, 'eval_probs.npy'), probs)

    roc_auc = roc_auc_score(y_eval, probs[:, 1])
    aver_pr = average_precision_score(y_eval, probs[:, 1])

    print('========== ROC AUC =', roc_auc)
    print('========== Aver PR =', aver_pr)

    params = dict()

    params['vect'] = str(args.vect)
    params['roc_auc'] = str(roc_auc)
    params['aver_pr'] = str(aver_pr)

    save_dict_to_yaml(params, os.path.join(args.model_dir, 'results.yaml'))

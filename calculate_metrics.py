"""
Metrics calculation
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

parser = argparse.ArgumentParser()

parser.add_argument('-dd', '--data_dir', default='data/treatments_features')
parser.add_argument('-md', '--model_dir', default='experiments/treatments_max_features')


if __name__ == '__main__':
    args = parser.parse_args()

    probs = np.load(os.path.join(args.model_dir, 'eval_probs.npy'))
    y_pred = (probs[:, 1] > 0.3).astype(int)
    labels = pd.read_csv(os.path.join(args.data_dir, 'eval.csv'))['target'].values

    roc_auc = roc_auc_score(labels, probs[:, 1])
    aver_pr = average_precision_score(labels, probs[:, 1])
    tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=y_pred).ravel()

    print('========== ROC AUC =', roc_auc)
    print('========== Aver PR =', aver_pr)
    print('========== TN, FP, FN, TP =', tn, fp, fn, tp)

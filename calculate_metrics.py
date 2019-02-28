"""
Metrics calculation
"""

import os
import argparse
import numpy as np
import pandas as pd
from model.utils import calculate_metrics

parser = argparse.ArgumentParser()

parser.add_argument('-dd', '--data_dir', default='data')
parser.add_argument('-md', '--model_dir', default='experiments')


if __name__ == '__main__':
    args = parser.parse_args()

    probs = np.load(os.path.join(args.model_dir, 'eval_probs.npy'))
    labels = pd.read_csv(os.path.join(args.data_dir, 'eval.csv'))['target'].values

    metrics = calculate_metrics(probs, labels, thres=0.3)

    for key, val in metrics.items():
        print(f'{key} = {val}')

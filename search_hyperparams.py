"""Peform hyperparameter search"""

import argparse
import os
import numpy as np
import glob
import re
from subprocess import check_call
import sys
from model.utils import get_yaml_config, save_dict_to_yaml


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('-pd', '--parent_dir', default='experiments')
parser.add_argument('-dd', '--data_dir', default='data')


def launch_training_job(parent_dir, data_dir, job_name, params):

    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    save_dict_to_yaml(params, os.path.join(model_dir, 'config.yaml'))

    cmd = "{python} train.py --model_dir {model_dir} --data_dir {data_dir}"
    cmd = cmd.format(python=PYTHON, model_dir=model_dir, data_dir=data_dir)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    args = parser.parse_args()

    params = get_yaml_config(os.path.join(args.parent_dir, 'config.yaml'))
    data_dir = args.data_dir

    for use_features in [True, False]:
        for encoder in ['no_encoder', 'GRU', 'biGRU', 'LSTM', 'biLSTM']:
            for agg_strategy in ['mean', 'max', 'concat']:

                params['features'] = use_features
                params['encoder'] = encoder
                params['aggregation'] = agg_strategy

                job_name = f'swem_{agg_strategy}_encoder={encoder}_features={str(use_features)}'
                launch_training_job(args.parent_dir, data_dir, job_name, params)

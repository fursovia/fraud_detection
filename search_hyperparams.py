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

    data_dir = args.data_dir
    params = get_yaml_config(os.path.join(args.parent_dir, 'config.yaml'))

    for use_features in [True, False]:
        for encoder in ['no_encoder', 'GRU', 'biGRU', 'LSTM', 'biLSTM']:
            for agg_strategy in ['mean', 'max', 'concat']:

                params['features'] = use_features
                params['encoder'] = encoder
                params['aggregation'] = agg_strategy

                job_name = f'swem_{agg_strategy}_encoder={encoder}_features={str(use_features)}'
                launch_training_job(args.parent_dir, data_dir, job_name, params)

    params = get_yaml_config(os.path.join(args.parent_dir, 'config.yaml'))
    seq_lens = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for seq_len in seq_lens:

        params['seq_len'] = seq_len
        job_name = f'swem_seq_len={seq_len}'
        launch_training_job(args.parent_dir, data_dir, job_name, params)

    params = get_yaml_config(os.path.join(args.parent_dir, 'config.yaml'))
    unit_sizes = [(64, 32, 16), (128, 64, 32), (256, 128, 64), (512, 256, 128)]
    for emb_dim, num_units1, num_units2 in unit_sizes:
        params['emb_dim'] = emb_dim
        params['num_units1'] = num_units1
        params['num_units2'] = num_units2

        job_name = f'swem_emb_size={emb_dim},num_units1={num_units1},num_units2={num_units2}'
        launch_training_job(args.parent_dir, data_dir, job_name, params)

    params = get_yaml_config(os.path.join(args.parent_dir, 'config.yaml'))
    for use_pretrained in [True, False]:
        for emb_dim in [50, 100, 300, 500, 700, 1000]:
            params['emb_dim'] = emb_dim
            params['use_pretrained'] = use_pretrained
            params['word2vec_filename'] = 'data/word2vec_treatments_{emb_dim}.txt'
            job_name = f'swem_emb_size={emb_dim},pretrained={str(use_pretrained)}'
            launch_training_job(args.parent_dir, data_dir, job_name, params)

    params = get_yaml_config(os.path.join(args.parent_dir, 'config.yaml'))
    for num_epochs in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        params['num_epochs'] = num_epochs

        job_name = f'swem_num_epochs={num_epochs}'
        launch_training_job(args.parent_dir, data_dir, job_name, params)

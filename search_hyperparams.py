"""Peform hyperparameter search"""

import argparse
import os
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

    emb_dims = [50, 100, 300]
    use_pretrained_options = [True]  # False

    units = [[32, 16], [64, 32], [128, 64]]

    for pretr in use_pretrained_options:
        for i, emb_dim in enumerate(emb_dims):

            print('*' * 200)
            print('Running training process')

            params['units']['swem_max_features'] = units[i]

            pretr_option = 'pretr' if pretr else 'not_pretr'
            params['use_pretrained'] = pretr
            params['emb_dim'] = emb_dim
            params['word2vec_filename'] = f'data/word2vec_treatments_{emb_dim}.txt'

            job_name = f'emb_dims_{emb_dim}_{pretr_option}'
            launch_training_job(args.parent_dir, args.data_dir, job_name, params)

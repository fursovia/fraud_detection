"""Peform hyperparameter search"""

import argparse
import os
import numpy as np
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

    # emb_dims = [50, 100, 300]
    # use_pretrained_options = [True, False]
    # units = [[32, 16], [64, 32], [128, 64]]

    seq_lens = np.arange(5, 101, 5)

    for seq_len in seq_lens:
        print(f'Running training process for seq_len = {seq_len}')

        params['use_pretrained'] = True
        params['seq_len'] = int(seq_len)

        job_name = f'swem_max_seq_len={seq_len}'
        launch_training_job(args.parent_dir, args.data_dir, job_name, params)

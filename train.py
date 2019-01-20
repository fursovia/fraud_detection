"""
Starts the training process
"""

import os
import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from model.model_fn import model_fn
from model.input_fn import input_fn
from model.utils import save_dict_to_yaml, get_yaml_config
from sklearn.metrics import roc_auc_score, average_precision_score

parser = argparse.ArgumentParser()

parser.add_argument('-dd', '--data_dir', default='data/treatments_features')
parser.add_argument('-md', '--model_dir', default='experiments/treatments_max_features')
parser.add_argument('-a', '--architecture', choices=['swem_aver', 'swem_max', 'swem_max_features'])
parser.add_argument('-pre', '--use_pretrained', action='store_true')

parser.set_defaults(architecture='swem_max_features')
parser.set_defaults(use_pretrained=False)


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parser.parse_args()

    params = get_yaml_config(os.path.join(args.model_dir, 'config.yaml'))
    params['model_dir'] = args.model_dir
    params['data_dir'] = args.data_dir
    params['treatments_vocab_path'] = os.path.join(args.data_dir, 'treatments.txt')
    params['num_treatments'] = sum(1 for _ in open(params['treatments_vocab_path'], 'r')) + 1
    params['train_size'] = pd.read_csv(os.path.join(args.data_dir, 'train.csv')).shape[0]
    params['arch_name'] = args.architecture if 'architecture' not in params else params['architecture']
    params['use_pretrained'] = args.use_pretrained if 'use_pretrained' not in params else params['use_pretrained']

    config = tf.estimator.RunConfig(tf_random_seed=24,
                                    save_checkpoints_steps=int(params['train_size'] / params['batch_size']),
                                    keep_checkpoint_max=None,
                                    model_dir=args.model_dir)

    estimator = tf.estimator.Estimator(model_fn,
                                       params=params,
                                       config=config)

    tf.estimator.train_and_evaluate(
        estimator,
        train_spec=tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(os.path.join(args.data_dir, 'train.tfrecords'), params, True),
            max_steps=int((params['train_size'] / params['batch_size']) * params['num_epochs']),
        ),

        eval_spec=tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(os.path.join(args.data_dir, 'eval.tfrecords'), params, False),
            steps=None,
            start_delay_secs=30,
            throttle_secs=60
        )
    )

    # SAVE PREDICTIONS

    eval_preds = estimator.predict(lambda: input_fn(os.path.join(args.data_dir, 'eval.tfrecords'), params, False))

    eval_logits = []
    probs = []
    for p in eval_preds:
        eval_logits.append(p['logits'])
        probs.append(p['preds'])

    eval_logits = np.array(eval_logits, np.float64).reshape(-1, 2)
    probs = np.array(probs, np.float64).reshape(-1, 2)

    np.save(os.path.join(args.model_dir, 'eval_logits.npy'), eval_logits)
    np.save(os.path.join(args.model_dir, 'eval_probs.npy'), probs)

    # CALCULATE METRICS

    labels = pd.read_csv(os.path.join(args.data_dir, 'eval.csv'))['target'].values

    roc_auc = roc_auc_score(labels, probs[:, 1])
    aver_pr = average_precision_score(labels, probs[:, 1])

    print('========== ROC AUC =', roc_auc)
    print('========== Aver PR =', aver_pr)

    params['roc_auc'] = roc_auc
    params['aver_pr'] = aver_pr

    save_dict_to_yaml(params, os.path.join(args.model_dir, 'config.yaml'))

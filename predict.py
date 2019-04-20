import os
import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from model.model_fn import model_fn
from model.input_fn import input_fn
from model.utils import get_yaml_config, calculate_metrics, save_dict_to_yaml

parser = argparse.ArgumentParser()

parser.add_argument('-dd', '--data_dir', default='data')
parser.add_argument('-md', '--model_dir', default='experiments')


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parser.parse_args()

    params = get_yaml_config(os.path.join(args.model_dir, 'config.yaml'))
    seq_lens = np.arange(5, 101, 5)

    for seq_len in seq_lens:
        tf.reset_default_graph()
        params['seq_len'] = seq_len

        config = tf.estimator.RunConfig(tf_random_seed=24,
                                        save_checkpoints_steps=int(params['train_size'] / params['batch_size']),
                                        keep_checkpoint_max=None,
                                        model_dir=args.model_dir,
                                        save_summary_steps=20)

        estimator = tf.estimator.Estimator(model_fn,
                                           params=params,
                                           config=config)

        eval_preds = estimator.predict(lambda: input_fn(os.path.join(args.data_dir, 'eval.csv'), params, False))

        eval_logits = []
        probs = []
        for p in eval_preds:
            eval_logits.append(p['logits'])
            probs.append(p['preds'])

        eval_logits = np.array(eval_logits, np.float64).reshape(-1, 2)
        probs = np.array(probs, np.float64).reshape(-1, 2)

        labels = pd.read_csv(os.path.join(args.data_dir, 'eval.csv'))['target'].values
        metrics = calculate_metrics(probs, labels, thres=0.3)

        for key, val in metrics.items():
            params[key] = str(val)
            print(f'{key} = {val}')

        save_dict_to_yaml(params, os.path.join(args.model_dir, f'config_seq_len={seq_len}.yaml'))

"""
Starts the training process
"""

import os
import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from model.model_fn import model_fn
from model.input_fn import input_fn, input_fn_in_memory
from model.utils import save_dict_to_yaml, get_yaml_config, calculate_metrics
from shutil import copyfile
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--data_dir', default='data')
parser.add_argument('-md', '--model_dir', default='experiments')
parser.add_argument('-a', '--aggregation', choices=['mean', 'max', 'concat'])
parser.add_argument('-e', '--encoder', choices=['GRU', 'biGRU', 'LSTM', 'biLSTM'])

parser.add_argument('-f', '--features', action='store_true', help='Add feature tower?')
parser.add_argument('-pre', '--use_pretrained', action='store_true')
parser.add_argument('-ime', '--in_memory_embeddings', action='store_true')  # TODO: not working
parser.add_argument('--seq_len', type=int, default=None)
parser.add_argument('--num_epochs', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--learning_rate', type=float, default=None)
parser.add_argument('--vocab_frac', type=float, default=None)

parser.set_defaults(encoder=None)
parser.set_defaults(features=None)
parser.set_defaults(aggregation=None)
parser.set_defaults(use_pretrained=None)
parser.set_defaults(in_memory_embeddings=None)


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if not os.path.exists(os.path.join(args.model_dir, 'config.yaml')):
        copyfile(os.path.join('experiments', 'config.yaml'), os.path.join(args.model_dir, 'config.yaml'))

    params = get_yaml_config(os.path.join(args.model_dir, 'config.yaml'))
    params['model_dir'] = args.model_dir
    params['data_dir'] = args.data_dir
    params['treatments_vocab_path'] = os.path.join(args.data_dir, 'treatments.txt')
    vocab_fraq = args.vocab_frac if args.vocab_frac is not None else params['vocab_frac']
    params['num_treatments'] = int(sum(1 for _ in open(params['treatments_vocab_path'], 'r')) * vocab_fraq) + 1
    params['train_size'] = pd.read_csv(os.path.join(args.data_dir, 'train.csv')).shape[0]

    for key, val in args.__dict__.items():
        params[key] = val if val is not None else params[key]

    for key, val in params.items():
        print(f'{key} >>>>>>>>>> {val}')

    config = tf.estimator.RunConfig(tf_random_seed=24,
                                    # uncomment to calculate metrics on validation every K steps
                                    # save_checkpoints_steps=int(params['train_size'] / params['batch_size']),
                                    keep_checkpoint_max=None,
                                    model_dir=args.model_dir,
                                    save_summary_steps=20)

    estimator = tf.estimator.Estimator(model_fn,
                                       params=params,
                                       config=config)

    train_path = os.path.join(args.data_dir, 'train.csv')
    eval_path = os.path.join(args.data_dir, 'eval.csv')

    if args.in_memory_embeddings:
        train_pickle_path = os.path.join(args.data_dir, 'train.pkl')
        eval_pickle_path = os.path.join(args.data_dir, 'eval.pkl')

        train_input = lambda: input_fn_in_memory(train_path, train_pickle_path, params, True)
        eval_input = lambda: input_fn_in_memory(eval_path, eval_pickle_path, params, True)
    else:
        train_input = lambda: input_fn(train_path, params, True)
        eval_input = lambda: input_fn(eval_path, params, False)

    # TRAINING
    tf.estimator.train_and_evaluate(
        estimator,
        train_spec=tf.estimator.TrainSpec(
            input_fn=train_input,
            max_steps=int((params['train_size'] / params['batch_size']) * params['num_epochs']),
        ),

        eval_spec=tf.estimator.EvalSpec(
            input_fn=eval_input,
            steps=None,
            start_delay_secs=0,
            throttle_secs=60
        )
    )

    # SAVE PREDICTIONS
    if args.in_memory_embeddings:
        csv_path = os.path.join(args.data_dir, 'eval.csv')
        pkl_path = os.path.join(args.data_dir, 'eval.pkl')
        eval_preds = estimator.predict(
            lambda: input_fn_in_memory(csv_path, pkl_path, params, False)
        )
    else:
        eval_preds = estimator.predict(
            lambda: input_fn(os.path.join(args.data_dir, 'eval.csv'), params, False)
        )

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
    metrics = calculate_metrics(probs, labels, thres=0.3)

    for key, val in metrics.items():
        params[key] = str(val)
        print(f'{key} = {val}')

    save_dict_to_yaml(params, os.path.join(args.model_dir, 'config.yaml'))

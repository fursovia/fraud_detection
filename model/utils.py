"""
utility functions
"""

import json
import yaml
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score


def get_yaml_config(config_path):

    with open(config_path, 'r', encoding='utf-8') as f:
        params = yaml.load(f)

    return params


def save_dict_to_yaml(d, yaml_path):
    with open(yaml_path, 'w') as file:
        yaml.dump(d, file, default_flow_style=False)


def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        json.dump(d, f, indent=4)


def save_vocab_to_txt_file(vocab, txt_path):
    with open(txt_path, "w") as f:
        f.write("\n".join(token for token in vocab))


def calculate_metrics(probs, labels, thres=0.3):

    # y_pred = (probs[:, 1] > thres).astype(int)

    metrics = dict()
    metrics['roc_auc'] = roc_auc_score(labels, probs[:, 1])
    metrics['aver_pr'] = average_precision_score(labels, probs[:, 1])
    metrics['f1'] = max(
        [f1_score(y_true=labels, y_pred=(probs[:, 1] > threshold).astype(int))
            for threshold in np.linspace(0.001, 0.99)]
    )
    # tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=y_pred).ravel()

    return metrics

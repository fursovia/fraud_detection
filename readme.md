
You will need TensorFlow `1.12.0` to run experiments


## Data preparation

1. Put `arzta_daten_anonym1.csv`, `arzta_daten_anonym2.csv`, `arzta_daten_anonym3.csv` and `arzta_daten_anonym4.csv` to `data_path` folder.
2. Run `python data_prep.py --data_dir data_path`

## Pre-train your embeddings

1. Be sure to run data preparation script first
2. Run `python pretrain_embeddings.py --data_path data_path/full.csv` (`full.csv` is the file created by `data_prep.py` script)

## Training

1. Choose the model `swem_aver`, `swem_max`, `swem_max_features`, `gru` or `gru_feats` (see `model/model_fn.py`)
for detailed information)
2. Create experiment folder `exp_path`
3. Put `experiments/config.yaml` to `exp_path` and specify model params inside the yaml file.
4. Run `python train.py --model_dir exp_path --data_dir data_path --architecture swem_max --use_pretrained`

This command will initialize embeddings from `word2vec_filename` (specified in `exp_path/config.yaml`) and train the model (`swem_max`).
After the training `exp_path/config.yaml` will be updated with `ROC AUC` and other metrics.

## Other scripts

* `xgb.py` will train `XGBClassifier`
* `search_hyperparams.py` will iterate over hyperparams (hard-coded inside the script) and run `train.py` multiple times
* `calculate_metrics.py` will calculate metrics
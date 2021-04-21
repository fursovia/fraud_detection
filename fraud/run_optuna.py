from functools import partial

import typer
import optuna


def set_trial(trial: optuna.Trial):
    trial.suggest_categorical("rnn_type", ['gru', 'lstm'])
    trial.suggest_float("emb_dropout", 0.0, 0.9)
    trial.suggest_int("embedding_dim", 8, 2048, log=True)
    trial.suggest_categorical("bidirectional", ['true', 'false'])
    trial.suggest_float("lstm_dropout", 0.0, 0.9)
    trial.suggest_int("lstm_dim", 8, 2048, log=True)
    # trial.suggest_int("num_examples_per_class", 8, 1024, log=True)
    trial.suggest_int("num_examples_per_class", 8, 32, log=True)
    trial.suggest_categorical("num_layers", [1, 2, 3, 4, 5])
    trial.suggest_categorical("num_highway_layers", [1, 2, 3, 4, 5])
    trial.suggest_categorical("hidden_dims_id", [str(x) for x in list(range(1, 11))])
    trial.suggest_float("features_dropout", 0.0, 0.9)
    trial.suggest_float("lr", 0.00001, 0.1, log=True)
    trial.suggest_categorical(
        "features_act",
        [
            'linear',
            'mish',
            'swish',
            'relu',
            'relu6',
            'elu',
            'gelu',
            'prelu',
            'leaky_relu',
            'hardtanh',
            'sigmoid',
            'tanh',
            'log_sigmoid',
            'softplus',
            'softshrink',
            'softsign',
            'tanhshrink',
            'selu'
        ]
    )


def fraud_objective(
        trial: optuna.Trial,
        config_path: str,
        serialization_dir: str
) -> float:
    set_trial(trial)

    executor = optuna.integration.allennlp.AllenNLPExecutor(
        trial=trial,
        config_file=config_path,
        serialization_dir=f"{serialization_dir}/{trial.number}",
        metrics="best_validation_roc_auc",
        include_package="fraud",
    )
    return executor.run()


def main(
        config_path: str,
        serialization_dir: str,
        num_trials: int = 500,
        n_jobs: int = 1,
        timeout: int = 60 * 60 * 24,
        study_name: str = "optuna_fraud"
):
    study = optuna.create_study(
        storage="sqlite:///result/final_classifier.db",
        sampler=optuna.samplers.TPESampler(seed=245),
        study_name=study_name,
        pruner=optuna.pruners.HyperbandPruner(),
        direction="maximize",
        load_if_exists=True,
    )

    objective = partial(fraud_objective, config_path=config_path, serialization_dir=serialization_dir)
    study.optimize(
        objective,
        n_jobs=n_jobs,
        n_trials=num_trials,
        timeout=timeout,
    )


if __name__ == "__main__":
    typer.run(main)

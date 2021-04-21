local rnn_type = std.extVar('rnn_type');
local emb_dropout = std.parseJson(std.extVar('emb_dropout'));
local embedding_dim = std.parseInt(std.extVar('embedding_dim'));
local bidirectional = std.extVar('bidirectional');
local lstm_dropout = std.parseJson(std.extVar('lstm_dropout'));
local lstm_dim = std.parseInt(std.extVar('lstm_dim'));
local num_layers = std.parseInt(std.extVar('num_layers'));
local features_dropout = std.parseJson(std.extVar('features_dropout'));
local features_act = std.extVar('features_act');
local num_highway_layers = std.parseInt(std.extVar('num_highway_layers'));
local hidden_dims_id = std.extVar('hidden_dims_id');
local lr = std.parseJson(std.extVar('lr'));
local num_examples_per_class = std.parseInt(std.extVar('num_examples_per_class'));


local hidden_dims = {
  "1": [4, 8],
  "2": [16, 32],
  "3": [32, 64],
  "4": [16, 16],
  "5": [64, 128],
  "6": [4, 4],
  "7": [2, 4],
  "8": [16, 16],
  "9": [32, 16],
  "10": [16, 4],
};


{
    "dataset_reader": "fraud_reader",
    "model": {
        "type": "fraud_classifier",
        "dropout": emb_dropout,
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": embedding_dim
                }
            }
        },
        "encoder": {
            "type": rnn_type,
            "bidirectional": if bidirectional == 'true' then true else false,
            "dropout": lstm_dropout,
            "hidden_size": lstm_dim,
            "input_size": embedding_dim,
            "num_layers": num_layers
        },
        "features_encoder": {
            "activations": features_act,
            "dropout": features_dropout,
            "hidden_dims": hidden_dims[hidden_dims_id],
            "input_dim": 5,
            "num_layers": std.length(hidden_dims[hidden_dims_id])
        },
        "num_highway_layers": num_highway_layers
    },
    "train_data_path": "./data_full/train.jsonl",
    "validation_data_path": "./data_full/valid.jsonl",
    "trainer": {
//        "callbacks": [
//            {"type": "optuna_pruner"},
//        ],
        "cuda_device": 0,
//        "num_epochs": 1,
        "num_epochs": 100,
        "optimizer": {
            "type": "adam",
            "lr": lr
        },
//        "patience": 1,
        "patience": 3,
        "validation_metric": "+roc_auc"
    },
//    "distributed": {
//        "cuda_devices": [0, 1]
//    },
    "data_loader": {
        "batch_sampler": {
            "type": "balanced",
            "num_classes_per_batch": 2,
            "num_examples_per_class": num_examples_per_class
        },
        "batches_per_epoch": std.ceil(5000 / $["data_loader"]["batch_sampler"]["num_examples_per_class"])
    },
    "validation_data_loader": {
        "batch_size": 2048
    }
}
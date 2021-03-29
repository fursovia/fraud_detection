local VOCAB = import 'common/vocab.jsonnet';
local LOADER = import 'common/loader.jsonnet';

{
  "dataset_reader": "fraud_reader",
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("VALID_DATA_PATH"),
  "model": {
    "type": "fraud_classifier",
    "embedder": "",
    "encoder": {
      "type": "gru",
      "input_size": 192,
      "hidden_size": 128,
      "num_layers": 2,
      "dropout": 0.4,
      "bidirectional": true
    },
    "features_encoder": null,
  },
  "data_loader": {
    "batch_size": 64,
    "shuffle": true,
  },
  "trainer": {
    "num_epochs": 10,
    "patience": 3,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "cuda_device": -1
  }
}
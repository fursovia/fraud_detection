local TRAINER = import 'common/trainer.jsonnet';
local LOADER = import 'common/loader.jsonnet';

{
  "dataset_reader": "fraud_reader",
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("VALID_DATA_PATH"),
  "model": {
    "type": "fraud_classifier",
    "embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 64,
        }
      }
    },
    "seq_encoder": {
      "type": "pytorch_transformer",
      "input_dim": 64,
      "num_layers": 6,
      "positional_encoding": "embedding"
    },
    "encoder": {
      "type": "boe",
      "embedding_dim": 64,
      "averaged": true,
    },
    "features_encoder": {
      "input_dim": 5,
      "num_layers": 2,
      "hidden_dims": [16, 32],
      "activations": "relu",
      "dropout": 0.1
    },
  },
  "data_loader": LOADER['data_loader'],
  "trainer": TRAINER['trainer']
}
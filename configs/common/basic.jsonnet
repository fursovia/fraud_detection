local TRAINER = import 'trainer.jsonnet';
local LOADER = import 'loader.jsonnet';

{
  "dataset_reader": "fraud_reader",
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("VALID_DATA_PATH"),
  "model": {
    "type": "fraud_classifier",
    "num_highway_layers": 2,
    "dropout": 0.5,
    "features_encoder": {
      "input_dim": 5,
      "num_layers": 2,
      "hidden_dims": [16, 8],
      "activations": "tanh",
      "dropout": 0.3
    },
    "embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 256,
        }
      }
    },
  },
  "data_loader": LOADER['data_loader'],
  "validation_data_loader": LOADER['validation_data_loader'],
  "trainer": TRAINER['trainer']
}
local TRAINER = import 'trainer.jsonnet';
local LOADER = import 'loader.jsonnet';

{
  "dataset_reader": "fraud_reader",
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("VALID_DATA_PATH"),
  "model": {
    "type": "fraud_classifier",
    "features_encoder": {
      "input_dim": 5,
      "num_layers": 2,
      "hidden_dims": [16, 32],
      "activations": "sigmoid",
      "dropout": 0.1
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
  "trainer": TRAINER['trainer']
}
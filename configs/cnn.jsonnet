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
    "encoder": {
      "type": "cnn",
      "embedding_dim": 64,
      "num_filters": 32,
      "conv_layer_activation": "relu",
    },
    "features_encoder": null,
  },
  "data_loader": LOADER['data_loader'],
  "trainer": TRAINER['trainer']
}
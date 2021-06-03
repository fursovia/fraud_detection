local TRAINER = import 'trainer.jsonnet';
local LOADER = import 'loader.jsonnet';

{
  "dataset_reader": "fraud_reader",
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("VALID_DATA_PATH"),
  "model": {
    "type": "fraud_classifier",
    "features_encoder": {
      "type": "bag_of_embeddings",
      "embedding_dim": 256,
      "averaged": true,
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
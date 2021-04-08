local config = import 'common/basic.jsonnet';

config + {
    "model"+: {
        "encoder": {
          "type": "bag_of_embeddings",
          "embedding_dim": 64,
          "averaged": true,
        },
        "features_encoder": null
    },
}
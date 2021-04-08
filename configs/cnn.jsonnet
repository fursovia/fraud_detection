local config = import 'common/basic.jsonnet';

config + {
    "model"+: {
        "encoder": {
          "type": "cnn",
          "embedding_dim": 256,
          "num_filters": 64,
          "conv_layer_activation": "relu",
        },
        "features_encoder": null
    },
}
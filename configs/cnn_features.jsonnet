local config = import 'common/basic.jsonnet';

config + {
    "model"+: {
        "encoder": {
          "type": "cnn",
          "embedding_dim": 128,
          "num_filters": 64,
          "conv_layer_activation": "relu",
        },
    },
}
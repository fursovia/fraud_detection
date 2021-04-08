local config = import 'common/basic.jsonnet';

config + {
    "model"+: {
        "type": "fraud_classifier",
        "encoder": {
          "type": "cnn",
          "embedding_dim": 256,
          "num_filters": 64,
          "conv_layer_activation": "relu",
        },
    },
}
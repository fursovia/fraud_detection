local config = import 'common/basic.jsonnet';

config + {
    "model"+: {
        "seq_encoder": {
          "type": "pytorch_transformer",
          "input_dim": 256,
          "num_layers": 6,
          "positional_encoding": null
        },
        "encoder": {
          "type": "boe",
          "embedding_dim": 256,
          "averaged": true,
        },
        "features_encoder": null
    },
}
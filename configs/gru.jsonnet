local config = import 'common/basic.jsonnet';

config + {
    "model"+: {
        "encoder": {
          "type": "gru",
          "input_size": 256,
          "hidden_size": 512,
          "num_layers": 2,
          "dropout": 0.4,
          "bidirectional": true
        },
        "features_encoder": null
    },
}
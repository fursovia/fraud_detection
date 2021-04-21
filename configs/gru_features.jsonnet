local config = import 'common/basic.jsonnet';

config + {
    "model"+: {
        "encoder": {
          "type": "gru",
          "input_size": 128,
          "hidden_size": 256,
          "num_layers": 1,
          "dropout": 0.4,
          "bidirectional": true
        },
    },
}
local config = import 'common/basic.jsonnet';

config + {
    "model"+: {
        "encoder": {
          "type": "lstm",
          "input_size": 128,
          "hidden_size": 256,
          "num_layers": 2,
          "dropout": 0.4,
          "bidirectional": true
        },
    },
}
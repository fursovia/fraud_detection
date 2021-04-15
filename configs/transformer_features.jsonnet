local config = import 'common/basic.jsonnet';

config + {
    "model"+: {
        "seq_encoder": {
          "type": "pytorch_transformer",
          "input_dim": 128,
          "num_layers": 6,
          "positional_encoding": "embedding"
        },
        "encoder": {
          "type": "boe",
          "embedding_dim": 128,
          "averaged": true,
        },
    },
    "data_loader"+: {"batch_size": 512}
}
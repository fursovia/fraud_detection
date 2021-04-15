{
  "trainer": {
    "num_epochs": 100,
    "patience": 5,
    "validation_metric": "+roc_auc",
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "cuda_device": 0,
    "callbacks": ["tensorboard"],
  }
}
{
  "data_loader": {
//    "batch_size": 2048,
    "batch_sampler": {
      "type": "balanced",
      "num_classes_per_batch": 2,
      "num_examples_per_class": 128
    },
    "batches_per_epoch": 32,
    "shuffle": true,
  },
  "validation_data_loader": {
    "batch_size": 2048,
  },
}
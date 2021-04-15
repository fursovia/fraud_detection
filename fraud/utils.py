import math
import random
from typing import Iterable, Iterator, List, TypeVar, Sequence

import numpy as np
from allennlp.data.samplers import BatchSampler
from allennlp.data.instance import Instance

Example = TypeVar('Example')
LABEL_FIELD_NAME = 'target'


def iterate_random_batches(data: Iterable[Example], batch_size: int) -> Iterator[List[Example]]:
    """
    Uniformly sample random batches of the same size from the data indefinitely (without replacement)

    Args:
        data: Iterable with data examples
        batch_size: Batch size to use for all batches

    Returns:
        Iterator over batches
    """
    population = list(data)

    if len(population) < batch_size:
        raise ValueError(f'Population size {len(population)} must be greater than batch size {batch_size}')

    seen: List[Example] = []
    while True:
        random.shuffle(population)

        num_full, num_trailing = divmod(len(population), batch_size)

        for start in range(0, num_full * batch_size, batch_size):
            batch = population[start : start + batch_size]
            seen.extend(batch)
            yield batch

        if num_trailing > 0:
            trailing = population[-num_trailing:]
            random.shuffle(seen)
            num_missing = batch_size - num_trailing
            seen, population = seen[:num_missing], seen[num_missing:] + trailing
            yield trailing + seen
        else:
            population = seen
            seen = []


@BatchSampler.register('balanced')
class BalancedBatchSampler(BatchSampler):
    """BalancedBatchSampler"""

    def __init__(
        self, num_classes_per_batch: int = 8, num_examples_per_class: int = 32
    ) -> None:
        self._num_classes_per_batch = num_classes_per_batch
        self._num_examples_per_class = num_examples_per_class

    def get_batch_size(self) -> int:
        """
        Returns batch size
        """
        return self._num_classes_per_batch * self._num_examples_per_class

    def get_num_batches(self, instances: Sequence[Instance]) -> int:
        batch_count_float = len(instances) / self.get_batch_size()
        return math.ceil(batch_count_float)

    def get_batch_indices(self, instances: Sequence[Instance]) -> Iterator[List[int]]:
        labels = np.array([instance.fields[LABEL_FIELD_NAME].label for instance in instances])
        unique_labels, counts = np.unique(labels, return_counts=True)
        unique_labels = list(unique_labels)
        class_examples_generators = {
            label: iterate_random_batches(np.flatnonzero(labels == label), self._num_examples_per_class)
            for label in unique_labels
        }
        batch_classes_generator = iterate_random_batches(unique_labels, self._num_classes_per_batch)

        for _ in range(self.get_num_batches(instances)):
            batch = []
            chosen_labels = next(batch_classes_generator)
            for label in chosen_labels:
                batch.extend(
                    next(class_examples_generators[label])
                )
            yield batch

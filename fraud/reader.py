from typing import List, Optional

import numpy as np
import jsonlines
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField, TensorField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer


@DatasetReader.register("fraud_reader")
class FraudReader(DatasetReader):
    def __init__(self, ) -> None:
        super().__init__(max_instances=None)
        self._tokenizer = WhitespaceTokenizer()

    def text_to_instance(
            self,
            amount: float,
            age: float,
            sex: float,
            ins_type: float,
            speciality: float,
            treatments: List[str],
            types: List[str],
            target: Optional[int] = None,
            **kwargs
    ) -> Instance:

        treatments = self._tokenizer.tokenize(" ".join(treatments))
        types = self._tokenizer.tokenize(" ".join(types))
        # we can use embeddings for [sex, ins_type, speciality]
        features = np.array([amount, age, sex, ins_type, speciality])

        fields = {
            "treatments": TextField(treatments, {"tokens": SingleIdTokenIndexer()}),
            "types": TextField(types, {"tokens": SingleIdTokenIndexer()}),
            "features": TensorField(features),
        }

        if target is not None:
            fields["target"] = LabelField(label=target, skip_indexing=True)
        return Instance(fields)

    def _read(self, file_path: str):

        with jsonlines.open(cached_path(file_path), "r") as reader:
            for items in reader:
                instance = self.text_to_instance(**items)
                yield instance

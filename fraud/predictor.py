from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor


@Predictor.register('fraud')
class FraudPredictor(Predictor):
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        instance = self._dataset_reader.text_to_instance(**json_dict)
        return instance

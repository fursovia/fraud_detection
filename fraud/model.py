from typing import Dict, Optional

import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, FeedForward, Seq2VecEncoder, Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics.auc import Auc


@Model.register("fraud_classifier")
class FraudClassifier(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            embedder: TextFieldEmbedder,
            encoder: Seq2VecEncoder,
            seq_encoder: Optional[Seq2SeqEncoder] = None,
            features_encoder: Optional[FeedForward] = None,
    ) -> None:
        super().__init__(vocab)
        self._embedder = embedder
        self._encoder = encoder
        self._seq_encoder = seq_encoder
        self._features_encoder = features_encoder

        output_dim = self._encoder.get_output_dim()
        if self._features_encoder is not None:
            output_dim += self._features_encoder.get_output_dim()

        self._linear = torch.nn.Linear(output_dim, 2)
        self._loss = torch.nn.CrossEntropyLoss()
        self._auc = Auc()

    def forward(
            self,
            treatments: TextFieldTensors,
            features: torch.Tensor,
            target: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(treatments)

        embeddings = self._embedder(treatments)
        if self._seq_encoder is not None:
            embeddings = self._seq_encoder(embeddings, mask)

        context_embeddings = self._encoder(embeddings, mask)

        if self._features_encoder is not None:
            feature_embeddings = self._features_encoder(features.float())
            context_embeddings = torch.cat((context_embeddings, feature_embeddings), dim=-1)

        logits = self._linear(context_embeddings)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = dict(logits=logits, probs=probs)
        if target is not None:
            loss = self._loss(logits, target.long().view(-1))
            output_dict["loss"] = loss
            self._auc(predictions=probs[:, 1], gold_labels=target)
        return output_dict

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_idx = str(label_idx)
            # label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(
            #     label_idx, str(label_idx)
            # )
            classes.append(label_idx)
        output_dict["label"] = classes
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"roc_auc": self._auc.get_metric(reset)}
        return metrics

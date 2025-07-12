from torch import Tensor
from torch.nn import Dropout, Linear
from transformers import BertModel, BertPreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.tokenization_utils_base import BatchEncoding


class BertForPairwiseCLS(BertPreTrainedModel):
    """同义句分类模型."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.bert: BertModel = BertModel(config, add_pooling_layer=False)
        self.dropout: Dropout = Dropout(config.hidden_dropout_prob)
        self.classifier: Linear = Linear(768, 2)
        self.post_init()

    def forward(self, x: BatchEncoding) -> Tensor:
        bert_outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.bert(**x)
        assert bert_outputs.last_hidden_state
        cls_vectors: Tensor = bert_outputs.last_hidden_state[:, 0, :]
        cls_vectors: Tensor = self.dropout(cls_vectors)
        logits: Tensor = self.classifier(cls_vectors)
        return logits

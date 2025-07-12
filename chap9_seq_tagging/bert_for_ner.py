from torch import nn, Tensor
from transformers import BertModel, BertPreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.tokenization_utils_base import BatchEncoding


class BertForNER(BertPreTrainedModel):
    """BERT模型在NER任务上的微调模型."""

    def __init__(self, config: PretrainedConfig, label_num: int) -> None:
        super().__init__(config)
        self.bert: BertModel = BertModel(config, add_pooling_layer=False)
        self.dropout: nn.Dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier: nn.Linear = nn.Linear(768, label_num)
        self.post_init()

    def forward(self, x: BatchEncoding) -> Tensor:
        bert_output: BaseModelOutputWithPoolingAndCrossAttentions = self.bert(**x)
        assert bert_output.last_hidden_state is not None
        sequence_output: Tensor = bert_output.last_hidden_state
        sequence_output: Tensor = self.dropout(sequence_output)
        return self.classifier(sequence_output)

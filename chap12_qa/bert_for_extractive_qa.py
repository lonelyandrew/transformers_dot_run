from typing import Any

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from transformers.configuration_utils import PretrainedConfig


class BertForExtractiveQA(BertPreTrainedModel):
    """BERT模型用于抽取式问答任务."""

    def __init__(self, config: PretrainedConfig) -> None:
        """初始化BERT模型.

        Args:
            config: 配置.
        """
        super().__init__(config)
        self.num_labels: int = config.num_labels
        self.bert: BertModel = BertModel(config, add_pooling_layer=False)
        self.dropout: nn.Dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier: nn.Linear = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, x: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播.

        Args:
            x: 输入.

        Returns:
            开始和结束位置的logits.
        """
        bert_output: Any = self.bert(**x)
        sequence_output: torch.Tensor = bert_output.last_hidden_state
        sequence_output: torch.Tensor = self.dropout(sequence_output)
        logits: torch.Tensor = self.classifier(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits: torch.Tensor = start_logits.squeeze(-1).contiguous()
        end_logits: torch.Tensor = end_logits.squeeze(-1).contiguous()

        return start_logits, end_logits

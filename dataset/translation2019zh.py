from typing import Any, Iterable, Optional

import jsonlines
import torch
from transformers import AutoModelForSeq2SeqLM
from dataset.dataset_base import DatasetBase


class Translation2019ZH(DatasetBase):
    """翻译2019中文数据集."""

    def __init__(self, data_file_path: str, checkpoint: str, nrows: Optional[int] = None) -> None:
        """初始化数据集.

        Args:
            data_file_path: 数据集文件路径.
            checkpoint: 模型checkpoint名称.
            nrows: 加载数据集行数.
        """
        self.nrows: Optional[int] = nrows
        super().__init__(data_file_path, checkpoint)

    def load_data(self, data_file_path: str) -> dict[int, dict[str, Any]]:
        """加载数据集.

        Args:
            data_file_path: 数据集文件路径.

        Returns:
            返回一个字典，key为样本索引，value为样本数据字典.
        """
        result_dict: dict[int, dict[str, Any]] = {}
        with jsonlines.open(data_file_path) as reader:
            for idx, data in enumerate(reader):
                if self.nrows and idx >= self.nrows:
                    break
                result_dict[idx] = data
        return result_dict

    def collate_fn(
        self,
        batch_samples: Iterable[dict[str, Any]],
        model: AutoModelForSeq2SeqLM,
        max_length: int = 128,
    ) -> Any:
        """批量处理函数.

        Args:
            batch_samples: 批量样本列表
            max_length: 最大长度
            model: 模型
        Returns:
            处理后的批量数据，通常是模型输入和标签的元组
        """
        batch_inputs, batch_targets = [], []
        for sample in batch_samples:
            batch_inputs.append(sample["chinese"])
            batch_targets.append(sample["english"])
        batch_data = self.tokenizer(
            batch_inputs,
            text_target=batch_targets,
            padding=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        batch_data["decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(batch_data["labels"])
        end_token_index = torch.where(batch_data["labels"] == self.tokenizer.eos_token_id)[1]
        for idx, end_idx in enumerate(end_token_index):
            batch_data["labels"][idx][end_idx + 1 :] = -100
        return batch_data

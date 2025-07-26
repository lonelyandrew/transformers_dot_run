from typing import Iterable, override

import jsonlines
import torch
from torch import Tensor
from transformers.tokenization_utils_base import BatchEncoding
from loguru import logger

from dataset.dataset_base import DatasetBase


class AFQMC(DatasetBase):
    """AFQMC数据集.

    AFQMC (Ant Financial Question Matching Corpus) ：蚂蚁金融语义相似度数据集，该数据集由蚂蚁金服提供。
    """

    def __init__(self, data_file: str, checkpoint: str) -> None:
        """初始化数据集.

        Args:
            data_file: 数据集文件路径.
            checkpoint: 模型checkpoint名称.
        """
        super().__init__(data_file, checkpoint)
        logger.info("加载AFQMC数据集, 样本量{}条", len(self.data))

    @override
    def load_data(
        self,
        data_file: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        shuffle: bool = True,
    ) -> dict[int, dict[str, str]]:
        """加载数据集.

        Args:
            data_file: 数据集文件路径.

        Returns:
            返回一个样本字典, key为索引, value为样本数据.
        """
        data: dict[int, dict[str, str]] = {}

        with jsonlines.open(data_file) as reader:
            for idx, sample in enumerate(reader):
                data[idx] = dict(sample)
        return data

    @override
    def collate_fn(self, batch_samples: Iterable[dict[str, str]]) -> tuple[BatchEncoding, Tensor]:
        """Batch处理函数.

        Args:
            batch_samples: Batch样本列表.

        Returns:
            返回一个特征Tensor和一个标签Tensor.
        """
        batch_sentence_1: list[str] = []
        batch_sentence_2: list[str] = []
        batch_label: list[int] = []

        for sample in batch_samples:
            batch_sentence_1.append(sample["sentence1"])
            batch_sentence_2.append(sample["sentence2"])
            batch_label.append(int(sample["label"]))
        x: BatchEncoding = self.tokenizer(
            batch_sentence_1, batch_sentence_2, padding=True, truncation=True, return_tensors="pt"
        )
        y: Tensor = torch.tensor(batch_label)
        return x, y

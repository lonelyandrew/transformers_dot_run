from typing import Iterable
import jsonlines
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from loguru import logger

from chap7_fine_tune import checkpoint


class AFQMC(Dataset):
    """AFQMC数据集.

    AFQMC (Ant Financial Question Matching Corpus) ：蚂蚁金融语义相似度数据集，该数据集由蚂蚁金服提供。
    """

    def __init__(self, data_file: str) -> None:
        """初始化数据集.

        Args:
            data_file: 数据集文件路径.
        """
        self.data: dict[int, dict[str, str]] = self.load_data(data_file)
        self.tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(checkpoint)
        logger.info("加载AFQMC数据集, 样本量{}条", len(self.data))

    def load_data(self, data_file: str) -> dict[int, dict[str, str]]:
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> dict[str, str]:
        return self.data[idx]

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

    def as_dataloader(self, batch_size: int, shuffle: bool = False) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)

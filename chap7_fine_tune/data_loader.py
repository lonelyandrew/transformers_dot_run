from typing import Iterable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BatchEncoding

from chap7_fine_tune import checkpoint

tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def collate_fn(batch_samples: Iterable[dict[str, str]]) -> tuple[BatchEncoding, Tensor]:
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
    x: BatchEncoding = tokenizer(batch_sentence_1, batch_sentence_2, padding=True, truncation=True, return_tensors="pt")
    y: Tensor = torch.tensor(batch_label)
    return x, y


def get_data_loader(dataset: Dataset, shuffle: bool = False) -> DataLoader:
    """获取数据集的加载器.

    Args:
        dataset: 数据集.
        shuffle: 是否随机打乱数据.

    Returns:
        返回一个DataLoader对象.
    """
    return DataLoader(dataset, batch_size=4, shuffle=shuffle, collate_fn=collate_fn)

from abc import ABC, abstractmethod
from typing import Any, Optional

from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer


class DatasetBase(ABC):
    """数据集基类"""

    def __init__(self) -> None:
        """初始化数据集."""
        raise OSError("DatasetBase类不能实例化")

    @classmethod
    @abstractmethod
    def load(
        cls,
        data_file_path: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dataset:
        """加载数据集.

        Args:
            data_file_path: 数据集文件路径.
            limit: 限制加载数据集行数.
            offset: 偏移量.

        Returns:
            返回一个数据集实例.
        """
        raise NotImplementedError("子类必须实现load方法")

    @abstractmethod
    def tokenize(self, dataset: Dataset, tokenizer: PreTrainedTokenizer, *args, **kwargs) -> Dataset:
        """预处理数据集.

        Args:
            dataset: 数据集.
            tokenizer: 分词器.
        """
        raise NotImplementedError("子类必须实现tokenize方法")

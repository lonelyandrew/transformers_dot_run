from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertTokenizer


class DatasetBase(Dataset, ABC):
    """数据集基类.

    提供统一的数据集接口，包括数据加载、批量处理等功能。
    子类需要实现load_data和collate_fn方法。
    """

    def __init__(self, data_file_path: str, checkpoint: str) -> None:
        """初始化数据集.

        Args:
            data_file_path: 数据集文件路径.
            checkpoint: 模型checkpoint名称.
        """
        self.data: dict[int, dict[str, Any]] = self.load_data(data_file_path)
        self.checkpoint: str = checkpoint

    @property
    def tokenizer(self) -> BertTokenizer:
        """获取tokenizer，如果未初始化则自动初始化."""
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        return self._tokenizer

    @abstractmethod
    def load_data(self, data_file_path: str) -> dict[int, dict[str, Any]]:
        """加载数据集.

        Args:
            data_file_path: 数据集文件路径

        Returns:
            返回一个字典，key为样本索引，value为样本数据字典
        """
        raise NotImplementedError("子类必须实现load_data方法")

    def __len__(self) -> int:
        """获取数据集长度."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """获取数据集中的一个样本."""
        return self.data[idx]

    @abstractmethod
    def collate_fn(self, batch_samples: Iterable[dict[str, Any]]) -> Any:
        """批量处理函数.

        Args:
            batch_samples: 批量样本列表

        Returns:
            处理后的批量数据，通常是模型输入和标签的元组
        """
        raise NotImplementedError("子类必须实现collate_fn方法")

    def as_dataloader(
        self,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        collate_fn_args: Optional[dict[str, Any]] = None,
    ) -> DataLoader:
        """将数据集转换为DataLoader.

        Args:
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 数据加载的工作进程数
            pin_memory: 是否将数据加载到CUDA固定内存中

        Returns:
            PyTorch DataLoader实例
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda x: self.collate_fn(x, **(collate_fn_args or {})),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

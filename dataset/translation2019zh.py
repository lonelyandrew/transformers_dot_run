from typing import Any, Optional, override

import jsonlines
from dataset.dataset_base import DatasetBase
from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding


class Translation2019ZH(DatasetBase):
    """翻译2019中文数据集."""

    @override
    @classmethod
    def load(
        cls,
        data_file_path: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dataset:
        """加载数据集.

        Args:
            data_file_path: 数据集文件路径.
            limit: 加载数据集行数.
            offset: 偏移量.

        Returns:
            返回一个数据集实例.
        """
        if offset is None:
            offset = 0
        data_list: list[dict[str, Any]] = []
        with jsonlines.open(data_file_path) as reader:
            for idx, data in enumerate(reader):
                if limit and len(data_list) >= limit:
                    break
                if offset and idx < offset:
                    continue
                data_list.append(data)

        return Dataset.from_list(data_list)

    @classmethod
    def encode(cls, examples: dict[str, Any], tokenizer: PreTrainedTokenizer, max_length: int) -> BatchEncoding:
        model_inputs: BatchEncoding = tokenizer(
            examples["chinese"], max_length=max_length, truncation=True, padding=False
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["english"], max_length=max_length, truncation=True, padding=False)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    @override
    @classmethod
    def tokenize(cls, dataset: Dataset, tokenizer: PreTrainedTokenizer, max_length: int) -> Dataset:
        """分词数据集.

        Args:
            dataset: 数据集.
            tokenizer: 分词器.
            max_length: 最大长度.

        Returns:
            返回一个数据集实例.
        """
        return dataset.map(cls.encode, batched=True, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length})


if __name__ == "__main__":
    from transformers import AutoTokenizer

    model_checkpoint = "Helsinki-NLP/opus-mt-zh-en"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    dataset: Dataset = Translation2019ZH.load("data/translation2019zh/train.jsonl", limit=10)
    dataset = Translation2019ZH.tokenize(dataset, tokenizer, max_length=128)
    print(dataset[0])

import json
from typing import Any, Iterable, override

import torch
from transformers.tokenization_utils_base import BatchEncoding

from dataset.dataset_base import DatasetBase


class CMRC2018(DatasetBase):
    """The Second Evaluation Workshop on Chinese Machine Reading Comprehension (CMRC 2018)数据集."""

    def __init__(
        self, data_file_path: str, checkpoint: str, max_length: int = 512, stride: int = 128, mode: str = "train"
    ) -> None:
        """初始化数据集.

        Args:
            data_file_path: 数据集文件路径.
            checkpoint: 模型checkpoint名称.
            max_length: 最大长度.
            stride: 步长.
            mode: 数据集模式.
        """
        super().__init__(data_file_path, checkpoint)
        self.max_length: int = max_length
        self.stride: int = stride
        self.mode: str = mode

    @override
    def load_data(self, data_file_path: str) -> dict[int, dict[str, Any]]:
        """加载数据集.

        Args:
            data_file_path: 数据集文件路径.
        """
        data: dict[int, dict[str, Any]] = {}

        with open(data_file_path, encoding="utf-8") as f:
            json_data: dict[str, Any] = json.load(f)
            example_idx: int = 0
            for article in json_data["data"]:
                title: str = article["title"]
                context: str = article["paragraphs"][0]["context"]
                for question_dict in article["paragraphs"][0]["qas"]:
                    question_id: str = question_dict["id"]
                    question: str = question_dict["question"]
                    answers_text: list[str] = [ans["text"] for ans in question_dict["answers"]]
                    answers_start: list[int] = [ans["answer_start"] for ans in question_dict["answers"]]
                    data[example_idx] = {
                        "id": question_id,
                        "title": title,
                        "context": context,
                        "question": question,
                        "answers": {"text": answers_text, "answer_start": answers_start},
                    }
                    example_idx += 1
        return data

    @override
    def collate_fn(self, batch_samples: Iterable[dict[str, Any]]) -> Any:
        """批量处理函数.

        Args:
            batch_samples: 批量样本列表.
        """
        if self.mode == "train":
            return self.train_collate_fn(batch_samples)
        elif self.mode == "test":
            return self.test_collate_fn(batch_samples)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def train_collate_fn(self, batch_samples: Iterable[dict[str, Any]]) -> Any:
        """批量处理函数.

        Args:
            batch_samples: 批量样本列表.

        Returns:
            处理后的批量数据.
        """
        batch_question, batch_context, batch_answers = [], [], []
        for sample in batch_samples:
            batch_question.append(sample["question"])
            batch_context.append(sample["context"])
            batch_answers.append(sample["answers"])

        batch_data: BatchEncoding = self.tokenizer(
            batch_question,
            batch_context,
            max_length=self.max_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )

        # offset_mapping: 每个token在原始文本中的位置
        offset_mapping: list[tuple[int, int]] = batch_data.pop("offset_mapping")
        # sample_mapping: 每个样本对应原始样本的索引
        sample_mapping: list[int] = batch_data.pop("overflow_to_sample_mapping")

        answer_start_positions: list[int] = []  # 答案开始位置
        answer_end_positions: list[int] = []  # 答案结束位置

        for i, offset in enumerate(offset_mapping):
            sample_idx: int = sample_mapping[i]  # 原始样本索引
            answer: dict[str, Any] = batch_answers[sample_idx]
            start_char: int = answer["answer_start"][0]  # 答案开始位置
            end_char: int = answer["answer_start"][0] + len(answer["text"][0])  # 答案结束位置
            sequence_ids: list[int | None] = batch_data.sequence_ids(i)  # type: ignore

            # 找到context的开始和结束位置
            idx: int = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start: int = idx

            while sequence_ids[idx] == 1:
                idx += 1
            context_end: int = idx - 1

            # 如果答案不在context中，则label为(0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:  # type: ignore
                answer_start_positions.append(0)
                answer_end_positions.append(0)
            else:
                # 否则是开始和结束token的位置
                idx: int = context_start
                while idx <= context_end and offset[idx][0] <= start_char:  # type: ignore
                    idx += 1
                answer_start_positions.append(idx - 1)

                idx: int = context_end
                while idx >= context_start and offset[idx][1] >= end_char:  # type: ignore
                    idx -= 1
                answer_end_positions.append(idx + 1)
        return batch_data, torch.tensor(answer_start_positions), torch.tensor(answer_end_positions)

    def test_collate_fn(self, batch_samples: Iterable[dict[str, Any]]) -> Any:
        """测试集批量处理函数.

        Args:
            batch_samples: 批量样本列表.

        Returns:
            处理后的批量数据.
        """
        batch_id, batch_question, batch_context = [], [], []
        for sample in batch_samples:
            batch_id.append(sample["id"])
            batch_question.append(sample["question"])
            batch_context.append(sample["context"])
        batch_data = self.tokenizer(
            batch_question,
            batch_context,
            max_length=self.max_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )

        offset_mapping: list[tuple[int, int]] = batch_data.pop("offset_mapping").numpy().tolist()
        sample_mapping: list[int] = batch_data.pop("overflow_to_sample_mapping")
        example_ids: list[str] = []

        for i in range(len(batch_data["input_ids"])):  # type: ignore
            sample_idx: int = sample_mapping[i]
            example_ids.append(batch_id[sample_idx])

            sequence_ids: list[int | None] = batch_data.sequence_ids(i)  # type: ignore
            offset: list[tuple[int, int]] = offset_mapping[i]  # type: ignore
            offset_mapping[i] = [o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)]  # type: ignore
        return batch_data, offset_mapping, example_ids

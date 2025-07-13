import json
from typing import Any, Iterable, override

from dataset.dataset_base import DatasetBase


class CMRC2018(DatasetBase):
    """The Second Evaluation Workshop on Chinese Machine Reading Comprehension (CMRC 2018)数据集."""

    def __init__(self, data_file_path: str, checkpoint: str) -> None:
        """初始化数据集.

        Args:
            data_file_path: 数据集文件路径
        """
        super().__init__(data_file_path, checkpoint)

    @override
    def load_data(self, data_file_path: str) -> dict[int, dict[str, Any]]:
        """加载数据集.

        Args:
            data_file_path: 数据集文件路径
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
            batch_samples: 批量样本列表

        Returns:
            处理后的批量数据
        """
        # 这里可以根据具体需求实现批量处理逻辑
        # 目前返回原始批量数据
        return list(batch_samples)

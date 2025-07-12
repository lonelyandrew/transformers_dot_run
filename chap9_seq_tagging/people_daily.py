from typing import Any

from transformers.tokenization_utils_base import BatchEncoding

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertTokenizer
from loguru import logger

from chap9_seq_tagging import checkpoint


class PeopleDaily(Dataset):
    """1998 年人民日报语料库."""

    def __init__(self, data_file: str) -> None:
        """初始化数据集.

        Args:
            data_file: 数据文件路径.
        """
        self.tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.categories: set[str] = set()
        self.data: dict[int, dict[str, Any]] = self.load_data(data_file)
        self.id2label = {0: "O"}
        for category in list(sorted(self.categories)):
            self.id2label[len(self.id2label)] = f"B-{category}"
            self.id2label[len(self.id2label)] = f"I-{category}"
        self.label2id = {v: k for k, v in self.id2label.items()}
        logger.info("加载人民日报语料库, 样本量{}条, 标签: {}", len(self.data), self.categories)
        logger.info("标签到id的映射: {}", self.id2label)
        logger.info("id到标签的映射: {}", self.label2id)

    def load_data(self, data_file) -> dict[int, dict[str, Any]]:
        data: dict[int, dict[str, Any]] = {}

        with open(data_file, encoding="utf-8") as f:
            for sentence_idx, line in enumerate(f.read().split("\n\n")):
                if not line:
                    break
                sentence: str = ""
                labels: list[Any] = []

                for char_idx, item in enumerate(line.split("\n")):
                    char: str
                    tag: str
                    char, tag = item.split(" ")
                    sentence += char
                    if tag.startswith("B"):
                        # start_idx, end_idx, word, tag
                        labels.append([char_idx, char_idx, char, tag[2:]])
                        self.categories.add(tag[2:])
                    elif tag.startswith("I"):
                        labels[-1][1] = char_idx
                        labels[-1][2] += char
                data[sentence_idx] = {"sentence": sentence, "labels": labels}
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collote_fn(self, batch_samples) -> tuple[BatchEncoding, Tensor]:
        batch_sentence: list[str] = []
        batch_tags: list[list[Any]] = []

        for sample in batch_samples:
            batch_sentence.append(sample["sentence"])
            batch_tags.append(sample["labels"])

        # 句子进行Tokenization
        batch_inputs: BatchEncoding = self.tokenizer(batch_sentence, padding=True, truncation=True, return_tensors="pt")

        # 创建标签矩阵
        batch_label: np.ndarray = np.zeros_like(batch_inputs["input_ids"], dtype=int)

        # 遍历每个句子
        for s_idx, sentence in enumerate(batch_sentence):
            # encoding: BatchEncoding = self.tokenizer(sentence, truncation=True)
            batch_label[s_idx][0] = -100  # 第一个token是[CLS]，不需要标签
            batch_label[s_idx][-1] = -100  # 最后一个token是[SEP]，不需要标签

            # 遍历每个字符
            for char_start, char_end, _, tag in batch_tags[s_idx]:
                token_start: int | None = batch_inputs.char_to_token(s_idx, char_start)
                token_end: int | None = batch_inputs.char_to_token(s_idx, char_end)
                if token_start is None or token_end is None:
                    continue
                batch_label[s_idx][token_start] = self.label2id[f"B-{tag}"]
                batch_label[s_idx][token_start + 1 : token_end + 1] = self.label2id[f"I-{tag}"]
        return batch_inputs, torch.tensor(batch_label)

    def as_dataloader(self, batch_size: int, shuffle: bool = False) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collote_fn)

from torch.utils.data import DataLoader, Subset, random_split
from transformers import AutoModelForSeq2SeqLM
from loguru import logger

from dataset.translation2019zh import Translation2019ZH
from chap10_translation import checkpoint


def main() -> None:
    train_set_size: int = 200000
    valid_set_size: int = 2000
    model: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    dataset: Translation2019ZH = Translation2019ZH(
        data_file_path="data/translation2019zh/train.jsonl",
        checkpoint=checkpoint,
        nrows=train_set_size + valid_set_size,
    )
    train_set: Subset[Translation2019ZH]
    valid_set: Subset[Translation2019ZH]
    train_set, valid_set = random_split(
        dataset,
        [train_set_size, valid_set_size],
    )
    logger.info(type(dataset))
    logger.info(f"train_set_size: {len(train_set)}")
    logger.info(f"valid_set_size: {len(valid_set)}")
    test_set: Translation2019ZH = Translation2019ZH(
        data_file_path="data/translation2019zh/valid.jsonl",
        checkpoint="bert-base-chinese",
    )
    logger.info(f"test_set_size: {len(test_set)}")
    logger.info(next(iter(train_set)))

    train_dataloader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda x: dataset.collate_fn(x, model, max_length=128),
    )
    batch = next(iter(train_dataloader))
    logger.info(list(batch.keys()))
    logger.info("batch shape: {}", {k: v.shape for k, v in batch.items()})


if __name__ == "__main__":
    main()

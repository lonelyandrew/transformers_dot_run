from transformers import AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from dataset.translation2019zh import Translation2019ZH
from chap10_translation import checkpoint
from chap10_translation.epoch import test_loop


def test() -> None:
    dataset: Translation2019ZH = Translation2019ZH(
        data_file_path="data/translation2019zh/valid.jsonl",
        checkpoint=checkpoint,
    )
    model: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda x: dataset.collate_fn(x, model, max_length=128),
    )
    test_loop(dataloader, model, dataset.tokenizer, max_length=128, mode="Test")


if __name__ == "__main__":
    test()

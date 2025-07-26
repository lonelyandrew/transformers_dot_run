import torch
from transformers import AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers.optimization import get_scheduler
from chap10_translation import checkpoint
from chap10_translation.epoch import test_loop, train_loop
from dataset.translation2019zh import Translation2019ZH


def train() -> None:
    learning_rate: float = 2e-5
    epoch_num: int = 3
    train_set_size: int = 200000
    valid_set_size: int = 2000

    dataset: Translation2019ZH = Translation2019ZH(
        data_file_path="data/translation2019zh/train.jsonl",
        checkpoint=checkpoint,
        nrows=train_set_size + valid_set_size,
    )
    train_set, valid_set = random_split(
        dataset,
        [train_set_size, valid_set_size],
    )
    train_dataloader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda x: dataset.collate_fn(x, model, max_length=128),
    )
    valid_dataloader = DataLoader(
        valid_set,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda x: dataset.collate_fn(x, model, max_length=128),
    )

    model: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=epoch_num * len(train_dataloader),
    )

    total_loss = 0.0
    best_bleu = 0.0
    for t in range(epoch_num):
        print(f"Epoch {t + 1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t + 1, total_loss)
    valid_bleu = test_loop(valid_dataloader, model, dataset.tokenizer, max_length=128, mode="Valid")
    print(f"BLEU: {valid_bleu:>0.2f}\n")
    if valid_bleu > best_bleu:
        best_bleu = valid_bleu
        print("saving new weights...\n")
        torch.save(model.state_dict(), f"epoch_{t + 1}_valid_bleu_{valid_bleu:0.2f}_model_weights.bin")
    print("Done!")


if __name__ == "__main__":
    train()

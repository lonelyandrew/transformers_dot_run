import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.optimization import get_scheduler

from chap9_seq_tagging.bert_for_ner import BertForNER
from chap9_seq_tagging import checkpoint, device
from chap9_seq_tagging.epoch import train_loop, test_loop
from dataset.people_daily import PeopleDaily
from loguru import logger
from utils.random import seed_everything


seed_everything(7)


def main() -> None:
    learning_rate: float = 1e-5
    epoch_num: int = 3

    train_data: PeopleDaily = PeopleDaily("data/china-people-daily-ner-corpus/example.train")
    train_dataloader: DataLoader = train_data.as_dataloader(batch_size=4, shuffle=True)

    valid_data: PeopleDaily = PeopleDaily("data/china-people-daily-ner-corpus/example.dev")
    valid_dataloader: DataLoader = valid_data.as_dataloader(batch_size=4)

    config: PretrainedConfig = AutoConfig.from_pretrained(checkpoint)
    model: BertForNER = BertForNER(config, label_num=len(train_data.id2label))
    model.to(device)  # type: ignore

    loss_fn: CrossEntropyLoss = CrossEntropyLoss()
    optimizer: AdamW = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler: LRScheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=epoch_num * len(train_dataloader),
    )

    total_loss: float = 0.0
    best_f1: float = 0.0
    for epoch_idx in range(epoch_num):
        logger.info(f"Epoch {epoch_idx + 1}/{epoch_num}")
        total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, epoch_idx + 1, total_loss)
        metrics: dict[str, dict[str, float]] = test_loop(valid_dataloader, model, train_data.id2label)
        valid_macro_f1, valid_micro_f1 = metrics["macro avg"]["f1-score"], metrics["micro avg"]["f1-score"]
        valid_f1 = metrics["weighted avg"]["f1-score"]
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            filename: str = f"epoch_{epoch_idx + 1}_valid_macrof1_{(100 * valid_macro_f1):0.3f}_microf1_{(100 * valid_micro_f1):0.3f}_weights.bin"
            torch.save(model.state_dict(), filename)
    logger.info("训练完成.")


if __name__ == "__main__":
    main()

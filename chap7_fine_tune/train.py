import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, get_scheduler, BertConfig
from loguru import logger

from chap7_fine_tune import device, checkpoint
from chap7_fine_tune.afqmc import AFQMC
from chap7_fine_tune.bert_for_pairwise_cls import BertForPairwiseCLS
from chap7_fine_tune.data_loader import get_data_loader
from chap7_fine_tune.epoch import train_loop, test_loop
from utils.random import seed_everything

seed_everything(42)

def main() -> None:
    # 数据准备
    train_dataset: AFQMC = AFQMC("data/AFQMC/train.jsonl")
    train_dataloader: DataLoader = get_data_loader(train_dataset)
    valid_dataset: AFQMC = AFQMC("data/AFQMC/dev.jsonl")
    valid_dataloader: DataLoader = get_data_loader(valid_dataset)

    # 创建模型
    config: BertConfig = AutoConfig.from_pretrained(checkpoint)
    model: Module = BertForPairwiseCLS.from_pretrained(checkpoint, config=config).to(device)

    # 配置超参
    learning_rate: float = 1e-5
    epoch_num: int = 3

    # 配置训练策略
    loss_fn: CrossEntropyLoss = CrossEntropyLoss()
    optimizer: Optimizer = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler: LRScheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=epoch_num * len(train_dataloader),
    )

    total_loss: float = 0.0
    best_acc: float = 0.0
    test_loop(valid_dataloader, model, mode="Valid")
    writer: SummaryWriter = SummaryWriter()


    for epoch_idx in range(epoch_num):
        logger.info(f"Epoch {epoch_idx + 1}/{epoch_num}")
        total_loss: float = train_loop(
            train_dataloader, model, loss_fn, optimizer, lr_scheduler, epoch_idx + 1, total_loss, writer
        )
        valid_acc: float = test_loop(valid_dataloader, model, mode="Valid")
        if valid_acc > best_acc:
            best_acc = valid_acc
            filename: str = f"epoch_{epoch_num + 1}_valid_acc_{(100 * valid_acc):0.1f}_model_weights.bin"
            logger.info(f"保存模型权重: {filename}, 当前最优ACC: {best_acc}")
            torch.save(model.state_dict(), filename)
    logger.info("训练完成.")
    writer.close()


if __name__ == "__main__":
    main()
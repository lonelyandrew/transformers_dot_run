import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from loguru import logger

from chap7_fine_tune import device


def test_loop(dataloader, model, mode="Test") -> float:
    assert mode in ["Valid", "Test"]
    size: int = len(dataloader.dataset)
    correct_cnt: float = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            prediction: Tensor = model(x)
            correct_cnt += (prediction.argmax(1) == y).type(torch.float).sum().item()

    acc: float = correct_cnt / size
    logger.info(f"{mode} Accuracy: {(100 * acc):>0.1f}%\n")
    return acc


def train_loop(
    dataloader: DataLoader,
    model: Module,
    loss_fn: Module,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    epoch_idx: int,
    total_loss: float,
    summary_writer: SummaryWriter,
) -> float:
    progress_bar: tqdm = tqdm(range(len(dataloader)))
    progress_bar.set_description(f"loss: {0:>7f}")
    finish_step_num: int = (epoch_idx - 1) * len(dataloader)

    model.train()
    x: Tensor
    y: Tensor
    for step_idx, (x, y) in enumerate(dataloader, start=1):
        x, y = x.to(device), y.to(device)
        prediction: Tensor = model(x)
        loss: Tensor = loss_fn(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        global_step_idx: int = finish_step_num + step_idx
        summary_writer.add_scalar("Loss/train", loss.item(), global_step_idx)
        avg_loss: float = total_loss / global_step_idx
        summary_writer.add_scalar("Loss/train_avg", avg_loss, global_step_idx)
        progress_bar.set_description(f"Loss: {avg_loss:>7f}")
        progress_bar.update(1)
    return total_loss

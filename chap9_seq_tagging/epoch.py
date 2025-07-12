from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from torch import nn
import torch
from seqeval.metrics import classification_report
from loguru import logger

from chap9_seq_tagging import device
from chap9_seq_tagging.bert_for_ner import BertForNER


def train_loop(
    dataloader: DataLoader,
    model: BertForNER,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    epoch_idx: int,
    total_loss: float,
) -> float:
    progress_bar: tqdm = tqdm(range(len(dataloader)))
    progress_bar.set_description(f"Loss: {0:>7f}")
    finish_batch_num: int = (epoch_idx - 1) * len(dataloader)

    model.train()
    for batch, (x, y) in enumerate(dataloader, start=1):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred.permute(0, 2, 1), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f"Loss: {total_loss / (finish_batch_num + batch):>7f}")
        progress_bar.update(1)
    return total_loss


def test_loop(dataloader: DataLoader, model: BertForNER, id2label: dict[int, str]) -> dict[str, dict[str, float]]:
    y_true, y_pred = [], []

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            predictions = pred.argmax(dim=-1).cpu().numpy().tolist()
            labels = y.cpu().numpy().tolist()
            y_true += [[id2label[int(i)] for i in label if i != -100] for label in labels]
            y_pred += [
                [id2label[int(p)] for (p, i) in zip(prediction, label) if i != -100]
                for prediction, label in zip(predictions, labels)
            ]
    metrics = classification_report(y_true, y_pred, mode="strict", scheme="IOB2", output_dict=True)
    logger.info(metrics)
    return metrics

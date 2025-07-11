import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from transformers import BertConfig, AutoConfig

from chap7_fine_tune import checkpoint, device
from chap7_fine_tune.afqmc import AFQMC
from chap7_fine_tune.bert_for_pairwise_cls import BertForPairwiseCLS
from chap7_fine_tune.data_loader import get_data_loader
from chap7_fine_tune.epoch import test_loop


def main() -> None:
    config: BertConfig = AutoConfig.from_pretrained(checkpoint)
    model: Module = BertForPairwiseCLS.from_pretrained(checkpoint, config=config).to(device)

    valid_dataset: AFQMC = AFQMC("data/AFQMC/dev.jsonl")
    valid_dataloader: DataLoader = get_data_loader(valid_dataset)

    model.load_state_dict(torch.load("epoch_3_valid_acc_74.1_model_weights.bin"))
    test_loop(valid_dataloader, model, mode="Test")


if __name__ == "__main__":
    main()

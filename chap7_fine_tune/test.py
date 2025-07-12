import torch
from torch.utils.data import DataLoader
from transformers import BertConfig
from transformers.configuration_utils import PretrainedConfig

from chap7_fine_tune import checkpoint, device
from dataset.afqmc import AFQMC
from chap7_fine_tune.bert_for_pairwise_cls import BertForPairwiseCLS
from chap7_fine_tune.epoch import test_loop


def main() -> None:
    config: PretrainedConfig = BertConfig.from_pretrained(checkpoint)
    model: BertForPairwiseCLS = BertForPairwiseCLS.from_pretrained(checkpoint, config=config)
    model.to(device)  # type: ignore

    valid_dataset: AFQMC = AFQMC("data/AFQMC/dev.jsonl")
    valid_dataloader: DataLoader = valid_dataset.as_dataloader(batch_size=4)

    model.load_state_dict(torch.load("epoch_3_valid_acc_74.1_model_weights.bin"))
    test_loop(valid_dataloader, model, mode="Test")


if __name__ == "__main__":
    main()

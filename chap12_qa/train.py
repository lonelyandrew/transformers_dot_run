import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.optimization import get_scheduler

from chap12_qa import device, checkpoint
from chap12_qa.bert_for_extractive_qa import BertForExtractiveQA
from chap12_qa.epoch import train_loop, test_loop
from dataset.cmrc2018 import CMRC2018


train_data: CMRC2018 = CMRC2018("data/cmrc2018/cmrc2018_train.json", checkpoint)
valid_data: CMRC2018 = CMRC2018("data/cmrc2018/cmrc2018_dev.json", checkpoint)
test_data: CMRC2018 = CMRC2018("data/cmrc2018/cmrc2018_trial.json", checkpoint)

train_dataloader: DataLoader = train_data.as_dataloader(batch_size=16, shuffle=True)
valid_dataloader: DataLoader = valid_data.as_dataloader(batch_size=16, shuffle=False)
test_dataloader: DataLoader = test_data.as_dataloader(batch_size=16, shuffle=False)

learning_rate = 1e-5
epoch_num = 3
config: PretrainedConfig = AutoConfig.from_pretrained(checkpoint)
config.num_labels = 2
model = BertForExtractiveQA.from_pretrained(checkpoint, config=config).to(device)  # type: ignore

loss_fn = nn.CrossEntropyLoss()
optimizer: AdamW = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num * len(train_dataloader),
)

total_loss = 0.0
best_avg_score = 0.0
for t in range(epoch_num):
    print(f"Epoch {t + 1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t + 1, total_loss)
    valid_scores = test_loop(valid_dataloader, valid_data, model, mode="Valid")
    avg_score = valid_scores["avg"]
    if avg_score > best_avg_score:
        best_avg_score = avg_score
        print("saving new weights...\n")
        torch.save(model.state_dict(), f"epoch_{t + 1}_valid_avg_{avg_score:0.4f}_model_weights.bin")
print("Done!")

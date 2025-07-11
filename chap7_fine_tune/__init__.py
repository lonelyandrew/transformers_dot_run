import torch
from torch.types import Device

checkpoint: str = "bert-base-chinese"
device: Device = 'cuda' if torch.cuda.is_available() else 'cpu'

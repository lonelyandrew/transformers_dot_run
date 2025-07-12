import torch

checkpoint: str = "bert-base-chinese"
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

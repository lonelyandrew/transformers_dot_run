import torch
from loguru import logger

checkpoint: str = "bert-base-chinese"
device: str = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {device}")

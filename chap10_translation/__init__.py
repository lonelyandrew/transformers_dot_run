import torch
from loguru import logger

checkpoint: str = "Helsinki-NLP/opus-mt-zh-en"
device: str = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {device}")

from loguru import logger

from dataset.people_daily import PeopleDaily
from utils.random import seed_everything


seed_everything(7)

train_data = PeopleDaily("data/china-people-daily-ner-corpus/example.train")
logger.info(train_data[0])
train_dataloader = train_data.as_dataloader(batch_size=4, shuffle=True)
batch_X, batch_y = next(iter(train_dataloader))
logger.info("batch_X shape: {}", {k: v.shape for k, v in batch_X.items()})
logger.info("batch_y shape: {}", batch_y.shape)
logger.info("batch_X: {}", batch_X)
logger.info("batch_y: {}", batch_y)

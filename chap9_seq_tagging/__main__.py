from loguru import logger

from chap9_seq_tagging.people_daily import PeopleDaily

train_data = PeopleDaily("data/china-people-daily-ner-corpus/example.train")
logger.info(train_data[0])

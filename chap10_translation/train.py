from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel

from dataset.translation2019zh import Translation2019ZH
from utils.random import seed_everything
from eval import compute_metrics

seed_everything(42)


def train() -> None:
    """训练主函数."""

    # 定义超参数
    train_set_size: int = 200000
    valid_set_size: int = 2000
    max_length: int = 128
    batch_size: int = 32

    # 加载数据集
    model_checkpoint: str = "Helsinki-NLP/opus-mt-zh-en"
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train_dataset: Dataset = Translation2019ZH.load("data/translation2019zh/train.jsonl", limit=train_set_size)
    valid_dataset: Dataset = Translation2019ZH.load(
        "data/translation2019zh/valid.jsonl",
        limit=valid_set_size,
        offset=train_set_size,
    )
    train_dataset = Translation2019ZH.tokenize(train_dataset, tokenizer, max_length=max_length)
    valid_dataset = Translation2019ZH.tokenize(valid_dataset, tokenizer, max_length=max_length)

    # 加载模型
    model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    # 定义数据整理器
    data_collator: DataCollatorForSeq2Seq = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100
    )

    # 定义训练参数
    training_args: TrainingArguments = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="best",
        learning_rate=1e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        logging_steps=1000,
    )

    # 定义训练器
    trainer: Trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
    )

    # 训练模型
    trainer.train()


if __name__ == "__main__":
    train()

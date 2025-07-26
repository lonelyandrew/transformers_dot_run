from transformers.trainer_utils import EvalPrediction
import numpy as np
from transformers.tokenization_utils import PreTrainedTokenizer

from sacrebleu.metrics.bleu import BLEU


def compute_metrics(eval_preds: EvalPrediction, tokenizer: PreTrainedTokenizer) -> dict[str, float]:
    """计算评估指标."""

    predictions, labels = eval_preds

    # 解码
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 文本清理
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # 使用sacrebleu计算BLEU
    bleu: BLEU = BLEU()
    return bleu.corpus_score(decoded_preds, decoded_labels).score

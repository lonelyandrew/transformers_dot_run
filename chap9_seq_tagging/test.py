import json

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from seqeval.scheme import IOB2
from seqeval.metrics import classification_report

from chap9_seq_tagging.bert_for_ner import BertForNER
from chap9_seq_tagging import device, checkpoint
from dataset.people_daily import PeopleDaily


def test() -> None:
    test_data: PeopleDaily = PeopleDaily("data/china-people-daily-ner-corpus/example.test")
    test_dataloader: DataLoader = test_data.as_dataloader(batch_size=4)

    config: PretrainedConfig = AutoConfig.from_pretrained(checkpoint)
    model: BertForNER = BertForNER(config, label_num=len(test_data.id2label))
    model.to(device)  # type: ignore
    model.load_state_dict(
        torch.load("epoch_3_valid_macrof1_95.878_microf1_96.049_weights.bin", map_location=torch.device("cpu"))
    )

    model.eval()
    with torch.no_grad():
        print("evaluating on test set...")
        true_labels, true_predictions = [], []
        for X, y in tqdm(test_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            predictions = pred.argmax(dim=-1).cpu().numpy().tolist()
            labels = y.cpu().numpy().tolist()
            true_labels += [[test_data.id2label[int(i)] for i in label if i != -100] for label in labels]
            true_predictions += [
                [test_data.id2label[int(p)] for (p, i) in zip(prediction, label) if i != -100]
                for prediction, label in zip(predictions, labels)
            ]
        print(classification_report(true_labels, true_predictions, mode="strict", scheme=IOB2))
        results = []
        print("predicting labels...")
        for s_idx in tqdm(range(len(test_data))):
            example = test_data[s_idx]
            inputs = test_data.tokenizer(example["sentence"], truncation=True, return_tensors="pt")
            inputs = inputs.to(device)
            pred = model(inputs)
            probabilities = torch.nn.functional.softmax(pred, dim=-1)[0].cpu().numpy().tolist()
            predictions = pred.argmax(dim=-1)[0].cpu().numpy().tolist()

            pred_label = []
            inputs_with_offsets = test_data.tokenizer(example["sentence"], return_offsets_mapping=True)
            offsets = inputs_with_offsets["offset_mapping"]

            idx = 0
            while idx < len(predictions):
                pred = predictions[idx]
                label = test_data.id2label[pred]
                if label != "O":
                    label = label[2:]  # Remove the B- or I-
                    start, end = offsets[idx]
                    all_scores: list[float] = [probabilities[idx][pred]]
                    # Grab all the tokens labeled with I-label
                    while idx + 1 < len(predictions) and test_data.id2label[predictions[idx + 1]] == f"I-{label}":
                        all_scores.append(probabilities[idx + 1][predictions[idx + 1]])
                        _, end = offsets[idx + 1]
                        idx += 1

                    score = np.mean(all_scores).item()
                    word = example["sentence"][start:end]
                    pred_label.append(
                        {
                            "entity_group": label,
                            "score": score,
                            "word": word,
                            "start": start,
                            "end": end,
                        }
                    )
                idx += 1
            results.append({"sentence": example["sentence"], "pred_label": pred_label, "true_label": example["labels"]})
        with open("test_data_pred.json", "wt", encoding="utf-8") as f:
            for exapmle_result in results:
                f.write(json.dumps(exapmle_result, ensure_ascii=False) + "\n")

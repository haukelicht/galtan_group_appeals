import numpy as np
import torch
import gc
import re

from transformers.trainer_utils import PredictionOutput
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, classification_report

from typing import List

# helper function to clean memory and reduce risk of out-of-memory error
def clear_memory():
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
  gc.collect()


def clean_text(text: str) -> str:
    text = re.sub(r'"+', '"', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def tokenize_nli_format(examples, tokenizer, **kwargs):
  return tokenizer(examples["text_prepared"], examples["hypothesis"], truncation=True, **kwargs)


def compute_metrics_binary(eval_pred: PredictionOutput, label_classes: List[str]):
    predictions, labels = eval_pred

    label_text_alphabetical = sorted(label_classes)

    ### reformat model output to enable calculation of standard metrics
    # split in chunks with predictions for each hypothesis for one unique premise
    def chunks(x: List, n):  # Yield successive n-sized chunks from lst. https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
        for i in range(0, len(x), n):
            yield x[i:i + n]

    # for each chunk/premise, select the most likely hypothesis
    prediction_chunks_lst = list(chunks(predictions, len(set(label_text_alphabetical)) ))
    hypo_position_highest_prob = []
    for i, chunk in enumerate(prediction_chunks_lst):
        hypo_position_highest_prob.append(np.argmax(np.array(chunk)[:, 0]))  # only accesses the first column of the array, i.e. the entailment/true prediction logit of all hypos and takes the highest one

    label_chunks_lst = list(chunks(labels, len(set(label_text_alphabetical)) ))
    label_position_gold = []
    for chunk in label_chunks_lst:
        label_position_gold.append(np.argmin(chunk))  # argmin to detect the position of the 0 among the 1s

    ### calculate standard metrics
    f1_macro = f1_score(label_position_gold, hypo_position_highest_prob, average='macro', zero_division=0.0)
    f1_micro = f1_score(label_position_gold, hypo_position_highest_prob, average='micro', zero_division=0.0)
    acc_balanced = balanced_accuracy_score(label_position_gold, hypo_position_highest_prob)
    acc_not_balanced = accuracy_score(label_position_gold, hypo_position_highest_prob)
    metrics = {'f1_macro': f1_macro,
               'f1_micro': f1_micro,
               'accuracy': acc_not_balanced,
               'balanced_accuracy': acc_balanced,
               }
    tmp = classification_report(label_position_gold, hypo_position_highest_prob, labels=np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]), target_names=label_text_alphabetical, output_dict=True, zero_division=0.0)

    tmp = {
        str(f'f1_{re.sub(",? ", "_", l)}'): v
        for l in label_classes
        for m, v in tmp[l].items() if 'f1' in m
      }
    metrics.update(tmp)

    metrics['worst_class_f1'] = min([metrics[f"f1_{l}"] for l in label_classes])


    return metrics


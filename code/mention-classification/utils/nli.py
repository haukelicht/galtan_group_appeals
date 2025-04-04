

import numpy as np
import pandas as pd
import re

from transformers.trainer_utils import PredictionOutput
import torch
from torch.nn import Softmax
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support, accuracy_score, classification_report
from transformers.pipelines.base import ArgumentHandler # , ChunkPipeline, build_pipeline_init_args
import gc

from typing import List, Dict, Union, Tuple, Union

STANCE_LABEL_CLASSES = ['positive', 'neutral', 'negative']
STANCE_HYPOTHESIS_TEMPLATE = "The author of the quote takes a {label} stance towards \"{entity}\"."

# helper function to clean memory and reduce risk of out-of-memory error
def clean_memory():
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
  gc.collect()


def clean_text(text: str) -> str:
    text = re.sub(r'"+', '"', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def format_nli_trainset(
        df: pd.DataFrame, 
        hypo_label_dict: Dict, 
        label_text_col: str='label_text',
        text_col: str='text_prepared',
        entity_col: str='mention',
        keep_label_text_col: bool=False,
        random_seed: int=42, 
        verbose: bool=False
    ) -> pd.DataFrame:
    """
    Formats the training data for NLI task.

    Args:
    df: pd.DataFrame
        The training data.
    hypo_label_dict: Dict
        A dictionary with keys as label_text and values as hypothesis template.
        Note that the hypothesis template should have a placeholder which can be replace with the string values in column `entity_col`.
    text_col: str
        The column name which contains the texts.
    entity_col: str
        The column name which contains the string values to be replaced in the hypothesis template.
    random_seed: int
        Random seed for reproducibility.
    verbose: bool
        Whether to print the logs.

    Returns:
    pd.DataFrame
        The formatted training data.
    """

    assert len(df) > 0, "The training data is empty."
    assert len(hypo_label_dict) > 0, "The hypo_label_dict is empty."
    required_cols = [label_text_col, entity_col, text_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    assert len(missing_cols) == 0, f"Missing columns: {missing_cols}"

    if verbose: print(f"Length of df before formatting step: {len(df)}.")
    length_original_data_train = len(df)

    dfs = []
    for label_text, hypothesis in hypo_label_dict.items():
        ## entailment
        df_step = df[df.label_text == label_text].copy(deep=True)
        df_step["hypothesis"] = df_step[entity_col].apply(lambda m: hypothesis % m)
        df_step["label"] = [0] * len(df_step)
        ## not entailment
        df_step_not_entail = df[df.label_text != label_text].copy(deep=True)
        # down-sample not-entailment examples (if needed)
        df_step_not_entail = df_step_not_entail.sample(n=min(len(df_step), len(df_step_not_entail)), random_state=random_seed)
        df_step_not_entail["hypothesis"] = df_step_not_entail[entity_col].apply(lambda m: hypothesis % m)
        df_step_not_entail["label"] = [1] * len(df_step_not_entail)
        # append
        dfs.append(pd.concat([df_step, df_step_not_entail]))
    df = pd.concat(dfs)

    # encode label
    df["label"] = df.label.apply(int)

    # shuffle
    df = df.sample(frac=1, random_state=random_seed).reset_index()

    if verbose: 
        print(f"After adding not_entailment training examples, the training data was augmented to {len(df)} texts.")
        print(f"Max augmentation could be: len(df) * 2 = {length_original_data_train*2}. It can also be lower, if there are more entail examples than not-entail for a majority class.")
    cols = ['label', 'hypothesis', text_col]
    if keep_label_text_col:
        cols.append(label_text_col)
    return df[cols].copy(deep=True)

def format_nli_testset(
        df: pd.DataFrame, 
        hypo_label_dict: Dict, 
        label_text_col: str='label_text',
        text_col: str='text_prepared',
        entity_col: str='mention',
        verbose: bool=False
    ):
    """
    Formats the test data for NLI task.

    Args:
    df: pd.DataFrame
        The training data.
    hypo_label_dict: Dict
        A dictionary with keys as label_text and values as hypothesis template.
        Note that the hypothesis template should have a placeholder which can be replace with the string values in column `entity_col`.
    text_col: str
        The column name which contains the texts.
    entity_col: str
        The column name which contains the string values to be replaced in the hypothesis template.
    verbose: bool
        Whether to print the logs.

    Returns:
    pd.DataFrame
        The formatted training data.
    """

    assert len(df) > 0, "The training data is empty."
    assert len(hypo_label_dict) > 0, "The hypo_label_dict is empty."
    required_cols = [label_text_col, entity_col, text_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    assert len(missing_cols) == 0, f"Missing columns: {missing_cols}"

    ## explode test dataset for N hypotheses
    hypothesis_list = list(hypo_label_dict.values())
    if verbose: print("Number of hypothesis_list/classes: ", len(hypothesis_list))

    # label lists with 0 at alphabetical position of their true hypo, 1 for not-true hypos
    label_text_label_dict_explode = {}
    for key, value in hypo_label_dict.items():
        labels = [0 if value == hypo else 1 for hypo in hypothesis_list]
        label_text_label_dict_explode[key] = labels

    df["label"] = df.label_text.map(label_text_label_dict_explode)
    df["hypothesis"] = df[entity_col].apply(lambda m: [hypo % m for hypo in hypo_label_dict.values()]).values
    if verbose: print(f"Original test set size: {len(df)}")

    # explode dataset to have K-1 additional rows with not_entail label and K-1 other hypothesis_list
    # ! after exploding, cannot sample anymore, because distorts the order to true label values, which needs to be preserved for evaluation code
    df = df.explode(["hypothesis", "label"])    # multi-column explode requires pd.__version__ >= '1.3.0'
    if verbose: print(f"Test set size for NLI classification: {len(df)}\n")

    # df["label_nli_explicit"] = ["True" if label == 0 else "Not-True" for label in df["label"]]    # adding this just to simplify readibility

    cols = ['label', 'hypothesis', text_col]
    return df[cols].copy(deep=True)

def tokenize_nli_format(examples, tokenizer):
  return tokenizer(examples["text_prepared"], examples["hypothesis"], truncation=True, max_length=512)

def compute_metrics_binary(eval_pred: PredictionOutput, label_classes: List[str]):
    predictions, labels = eval_pred

    label_text_alphabetical = sorted(label_classes)

    ### reformat model output to enable calculation of standard metrics
    # split in chunks with predictions for each hypothesis for one unique premise
    def chunks(x: List, n):  # Yield successive n-sized chunks from lst. https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
        for i in range(0, len(x), n):
            yield x[i:i + n]

    # for each chunk/premise, select the most likely hypothesis
    softmax = Softmax(dim=1)
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



# NOTE: the only thing we need to modify when using the zero-shot pipeline for NLI is that the hypothesis template must allow including the mentioned entity
class ZeroShotMentionClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for zero-shot for text classification by turning each possible label into an NLI
    premise/hypothesis pair.

    based on ZeroShotClassificationArgumentHandler from transformers.pipelines.zero_shot_classification
     (see https://github.com/huggingface/transformers/blob/fc689d75a04e846f63f8d7a4a420da0cf796f86b/src/transformers/pipelines/zero_shot_classification.py#L14)
    """

    def _parse_labels(self, labels):
        if isinstance(labels, str):
            labels = [label.strip() for label in labels.split(",") if label.strip()]
        return labels

    def __call__(self, sequences: Union[Tuple[str, str], List[Tuple[str, str]]], labels: List[str], hypothesis_template: str = "{entity} is {label}."):
        if isinstance(sequences, tuple):
            sequences = [sequences]

        if len(labels) == 0 or len(sequences) == 0:
            raise ValueError("You must include at least one label and at least one sequence.")
        if any(len(sequence) != 2 for sequence in sequences):
            raise ValueError("the sequence inputs must be a list of tuples with two elements: the text and the mentioned entity.")
        entities = [entity for _, entity in sequences]
        sequences = [sequence for sequence, _ in sequences]

        if hypothesis_template.format(entity=entities[0], label=labels[0]) == hypothesis_template:
            raise ValueError(
                # TODO: change the error message
                (
                    'The provided hypothesis_template "{}" was not able to be formatted with the target labels. '
                    "Make sure the passed template includes formatting syntax such as {{}} where the label should go."
                ).format(hypothesis_template)
            )

        sequence_pairs = []
        for sequence, entity in zip(sequences, entities):
            sequence_pairs.extend([[sequence, hypothesis_template.format(label=label, entity=entity)] for label in labels])

        return sequence_pairs, sequences

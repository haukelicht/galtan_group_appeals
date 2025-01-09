from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

import pandas as pd

from typing import List, Dict, Union

def tokenize_and_align_sequence_labels(examples, tokenizer, **kwargs):
    # source: simplied from  https://github.com/huggingface/transformers/blob/730a440734e1fb47c903c17e3231dac18e3e5fd6/examples/pytorch/token-classification/run_ner.py#L442
    tokenized_inputs = tokenizer(examples['tokens'], is_split_into_words=True, **kwargs)

    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs

def create_token_classification_dataset(
    corpus: List[Dict], 
    tokens_field: str='tokens',
    labels_field: Union[None, str]='labels'
):
    dataset = Dataset.from_list(corpus)
    if tokens_field != 'tokens':
        dataset = dataset.rename_column(tokens_field, 'tokens')
    if labels_field is not None and labels_field != 'labels':
        dataset = dataset.rename_column(labels_field, 'labels')
    required = ['tokens'] if labels_field is None else ['tokens', 'labels']
    rm = [c for c in dataset.column_names if c not in required]
    if len(rm) > 0:
        dataset = dataset.remove_columns(rm)
    return dataset

def split_dataset(dataset, test_size: Union[None, float]=0.2, dev_size: Union[None, float]=0.2, seed=42) -> 'DatasetDict':
    if test_size: assert 0 < test_size < 1, "test_size must be in (0, 1)"
    if dev_size: assert 0 < dev_size < 1, "dev_size must be in (0, 1)"
    if dev_size and test_size: assert (dev_size + test_size) < 1, "dev_size + test_size must be less than 1"

    n = len(dataset)
    n_test = int(test_size * n) if test_size else 0
    n_dev = int(dev_size * n) if dev_size else 0
    assert n_test + n_dev < n, "test_size + dev_size must be less than the number of examples"

    idxs = list(range(n))
    tmp, test_idxs = train_test_split(idxs, test_size=n_test, random_state=seed) if n_test > 0 else (idxs, [])
    train_idxs, dev_idxs = train_test_split(tmp, test_size=n_dev, random_state=seed) if n_dev > 0 else (idxs, [])

    dataset_dict = {'train': dataset.select(train_idxs)}
    if dev_size:
        dataset_dict['dev'] = dataset.select(dev_idxs)
    if test_size:
        dataset_dict['test'] = dataset.select(test_idxs)
    
    return DatasetDict(dataset_dict)


def format_nli_trainset(
        df: pd.DataFrame, 
        hypo_label_dict: Dict, 
        text_col: str='text_prepared',
        entity_col: str='mention',
        random_seed: int=0, 
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
    required_cols = ['label_text', entity_col, text_col]
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

    return df[['label', 'hypothesis', text_col]].copy(deep=True)

def format_nli_testset(
        df: pd.DataFrame, 
        hypo_label_dict: Dict, 
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
    required_cols = ['label_text', entity_col, text_col]
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

    return df[['label', 'hypothesis', text_col]].copy(deep=True)
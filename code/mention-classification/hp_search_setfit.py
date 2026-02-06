
# determine if current environment is a python script
is_python_script = '__file__' in globals()

# evaluate below if run as a python script
if is_python_script:
        
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_splits_path', type=str, required=True, help='Path to data splits directory. Should contain files "train.pkl", "val.pkl", and "test.pkl"')
    parser.add_argument('--combine_train_val', action='store_true', help='Whether to combine train and validation splits')
    
    parser.add_argument('--label_cols', type=str, required=True, help='Comma-separated list of label column names')
    parser.add_argument('--exclude_label_cols', type=str, nargs='+', default=None, help='Comma-separated list of label column names to exclude from --label_cols')
    parser.add_argument('--id_col', type=str, default='mention_id', help='Column name for unique mention IDs')
    parser.add_argument('--text_col', type=str , default='text', help='Column name for mention context text')
    parser.add_argument('--mention_col', type=str , default='mention', help='Column name for mention text')
    parser.add_argument('--span_col', type=str , default='span', help='Column name for mention span (start, end)')
    
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use. Must be a sentence-transformers compatible model.')
    parser.add_argument('--use_span_embeddings', action='store_true', help='Whether to use custom SeFitForSpanClassification Trainer instead of mention and text concatenation or mention-only strategies')
    parser.add_argument('--concat_strategy', type=str, choices=[None, 'prefix', 'suffix'], default=None, help='If not None, concatenate the mention text as prefix or suffix to the context text using --concat_sep_token')
    parser.add_argument('--concat_sep_token', type=str, default=': ', help='Separator token to use when concatenating mention text to context text')
    
    parser.add_argument('--class_weighting_strategy', type=str, choices=[None, 'balanced', 'inverse_proportional'], default=None, help='Class weighting strategy to use during training')
    parser.add_argument('--class_weighting_smooth_exponent', type=float, default=0.5, help='Smoothing exponents to use when computing class weights (only relevant if --class_weighting_strategy is set to "inverse_proportional")')

    # hyperparameter search spaces
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials to run for hyperparameter search')
    parser.add_argument('--body_train_batch_sizes', type=int, nargs='+', default=[32], help='List of batch sizes to use for embedding model training')
    parser.add_argument('--head_train_batch_sizes', type=int, nargs='+', default=[8], help='List of batch sizes to use for classifier head training')
    parser.add_argument('--head_learning_rates', type=float, nargs='+', default=[0.01], help='Learning rates to use for classifier head training')
    parser.add_argument('--l2_weight_decays', type=float, nargs='+', default=[0.01], help='L2 weight decay to use for classifier head training')
    parser.add_argument('--warmup_proportions', type=float, nargs='+', default=[0.1], help='Warmup proportion to use for classifier head training')
    
    parser.add_argument('--body_early_stopping_patience', type=int, default=2, help='Early stopping patience for sentence transformer finetuning')
    parser.add_argument('--body_early_stopping_threshold', type=float, default=0.01, help='Early stopping threshold for sentence transformer finetuning')
    parser.add_argument('--head_early_stopping_patience', type=int, default=5, help='Early stopping patience for classifier head finetuning')
    parser.add_argument('--head_early_stopping_threshold', type=float, default=0.015, help='Early stopping threshold for classifier head finetuning')

    parser.add_argument('--save_results_to', type=str, required=True, help='Directory to save evaluation results to')
    parser.add_argument('--overwrite_results', action='store_true', help='Whether to overwrite existing evaluation results')
    
    parser.add_argument('--train_best_model', action='store_true', help='Whether to train the best model on the combined train and validation sets after hyperparameter search')
    parser.add_argument('--save_eval_results', action='store_true', help='Whether to save evaluation results to disk')
    parser.add_argument('--save_eval_predictions', action='store_true', help='Whether to save evaluation predictions to disk')

    parser.add_argument('--train_final_model', action='store_true', help='Whether to train the final model on the combined data splits with "optimal" hyperparameters')
    parser.add_argument('--save_final_model_to', type=str, help='Directory to save the trained model to')
    args = parser.parse_args()

else:
    from types import SimpleNamespace
    args = SimpleNamespace()

    args.data_splits_path =  '../../data/annotations/group_mention_categorization/splits/model_selection/fold01/'
    args.combine_train_val = True

    # args.label_cols = 'economic,noneconomic'
    args.label_cols = 'economic__*'
    # args.label_cols = 'noneconomic__*'
    args.exclude_label_cols = None  # e.g., 'economic__employment_status'
    
    args.id_col = 'mention_id'
    args.text_col = 'text'
    args.mention_col = 'mention'
    args.span_col = 'span'

    args.model_name = 'sentence-transformers/all-mpnet-base-v2'
    # args.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # args.model_name = "ibm-granite/granite-embedding-english-r2" # can't use because it uses `pooling_mode_cls_token`
    # args.model_name = "nomic-ai/modernbert-embed-base"
    # args.model_name = "Alibaba-NLP/gte-modernbert-base" # can't use because it uses `pooling_mode_cls_token`
    # args.model_name = "google/embeddinggemma-300m"
    # args.model_name = "Qwen/Qwen3-Embedding-0.6B"


    args.use_span_embeddings = False # or True
    args.concat_strategy = None # 'prefix', 'suffix' or None
    args.concat_sep_token = ': '  # separator token for prefix/suffix concatenation
    
    args.class_weighting_strategy = 'inverse_proportional'  # or 'balanced' or None
    args.class_weighting_smooth_exponent = 0.5

    args.n_trials = 2 # !!!
    args.body_train_batch_sizes = [64, 32, 16] # embeddinggemma-300m needs smaller batch size due to memory constraints
    args.head_train_batch_sizes = [16, 8, 4]
    args.head_learning_rates = [0.03, 0.01, 0.003, 0.0001]
    args.l2_weight_decays = [0.01, 0.015, 0.2]
    args.warmup_proportions = [0.1, 0.15]
    
    args.body_early_stopping_patience = 5
    args.body_early_stopping_threshold = 0.01
    args.head_early_stopping_patience = 5
    args.head_early_stopping_threshold = 0.015

    strategy = 'span_embedding' if args.use_span_embeddings else 'mention_text' if args.concat_strategy is None else f'concat_{args.concat_strategy}'
    args.save_results_to = f'../../results/classifiers/attribute_dimensions_classification/hp_search/setfit/{args.model_name.replace("/", "--")}/fold01/{strategy}'
    args.overwrite_results = True
    
    args.train_best_model = True
    args.save_eval_results = True
    args.save_eval_predictions = True

    args.train_final_model = True
    mn = args.model_name.split('/')[-1]
    args.save_final_model_to = f'../../models/{mn}_mention-economic-attributes-classifier'
    args.save_final_model_to




# ## Setup


import os
from pathlib import Path
import shutil
import warnings
import json

import numpy as np
np.set_printoptions(precision=4, suppress=True)
import pandas as pd
import regex

import torch
torch.set_float32_matmul_precision('high')
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, set_seed
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # to enable deterministic behavior with CuBLAS
SEED = 42
set_seed(SEED, deterministic=True) # for reproducibility

from src.finetuning.setfit_extensions import model_init
from src.finetuning.setfit_extensions.class_weights_head import compute_class_weights
from src.finetuning.setfit_extensions.early_stopping import (
    EarlyStoppingTrainingArguments,
    EarlyStoppingCallback,
    SetFitEarlyStoppingTrainer
)
from src.finetuning.setfit_extensions.span_embedding import SetFitTrainerForSpanClassification
from setfit import Trainer

from optuna.samplers import TPESampler # for seeting sampler seed

from sklearn.metrics import classification_report


args.data_splits_path = Path(args.data_splits_path)

if isinstance(args.label_cols, str):
    args.label_cols = [col.strip() for col in args.label_cols.split(',')]
if args.exclude_label_cols is not None:
    if isinstance(args.exclude_label_cols, str):
        args.exclude_label_cols = [col.strip() for col in args.exclude_label_cols.split(',')]
    args.label_cols = [col for col in args.label_cols if col not in args.exclude_label_cols]

if (args.save_eval_results or args.save_eval_predictions) and not args.train_best_model:
    raise ValueError("'save_eval_results' or 'save_eval_predictions' is set to True but 'train_best_model' is False. Cannot save evaluation results for a model that was not trained.")

if args.save_results_to is not None:
    args.save_results_to = Path(args.save_results_to)
    if args.save_results_to.exists() and not args.overwrite_results:
        raise ValueError(f"The directory '{args.save_results_to}' already exists. To avoid overwriting, please specify a different path or set 'overwrite_results'")
    else:
        args.save_results_to.mkdir(parents=True, exist_ok=True)

if args.save_final_model_to and not args.train_best_model:
    raise ValueError("'save_final_model' is set to True but 'train_best_model' is False. Cannot save a model that was not trained.")

if args.train_final_model and not args.train_best_model:
    raise ValueError("'train_final_model' is set to True but 'train_best_model' is False. Cannot train final model without training best model first.")

if args.save_final_model_to:
    if not args.train_best_model:
        raise ValueError("'save_final_model' is set to True but 'train_best_model' is False. Cannot save a model that was not trained.")
    if args.save_final_model_to is None or args.save_final_model_as is None:
        raise ValueError("Both 'save_model_to' and 'save_model_as' must be specified if 'save_final_model' is True.")
    args.save_final_model_to = Path(args.save_final_model_to)


from optuna import Trial
from typing import Dict, Union, Tuple
from itertools import product
import warnings

def hp_space(trial: Trial) -> Dict[str, Union[float, int, Tuple[int, int]]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trial_args = {
            "batch_size": trial.suggest_categorical("batch_size", list(product(args.body_train_batch_sizes, args.head_train_batch_sizes))),
            "head_learning_rate": trial.suggest_categorical("head_learning_rate", args.head_learning_rates),
            "l2_weight": trial.suggest_categorical("l2_weight", args.l2_weight_decays),
            "warmup_proportion": trial.suggest_categorical("warmup_proportion", args.warmup_proportions),
        }
    return trial_args


# ## Prepare the datasets


# ### Load the splits


df = pd.concat({split: pd.read_pickle(args.data_splits_path / f"{split}.pkl") for split in ['train', 'val', 'test']})
df.reset_index(level=0, names='split', inplace=True)
if args.combine_train_val:
    df.loc[df['split'] == 'val', 'split'] = 'train'
    df.loc[df['split'] == 'test', 'split'] = 'val'
    df['split'] = pd.Categorical(df['split'], categories=['train', 'val'], ordered=True)
else:
    df['split'] = pd.Categorical(df['split'], categories=['train', 'val', 'test'], ordered=True)


# ### prepare the label column


# consider that entries in args.label_cols may be glob patterns
import fnmatch
expanded_label_cols = []
for lab in args.label_cols:
    matched = fnmatch.filter(df.columns, lab)
    if matched:
        expanded_label_cols.extend(matched)
    elif lab in df.columns:
        expanded_label_cols.append(lab)
    else:
        raise ValueError(f"Label column '{lab}' not found in dataframe columns and did not match any glob patterns.")

args.label_cols = expanded_label_cols

# print info about label columns
print(f"Using label columns: {args.label_cols}")


df['labels'] = df[args.label_cols].apply(list, axis=1)


# ### format inputs


tokenizer = AutoTokenizer.from_pretrained(args.model_name)

if args.use_span_embeddings:
    if "span" not in df.columns:
    # using span embedding strategy
        df['span'] = df.apply(lambda x: regex.search(regex.escape(x[args.mention_col]), x[args.text_col]).span(), axis=1)
    max_length_ = max(tokenizer(df[args.text_col].to_list(), truncation=False, padding=False, return_length=True).length)
    cols = [args.text_col, 'span', 'labels']
    cols_mapping = {args.text_col: 'text', 'span': 'span', 'labels': 'label'}
elif args.concat_strategy is None:
    # default: just the mention text
    max_length_ = max(tokenizer(df[args.mention_col].to_list(), truncation=False, padding=False, return_length=True).length)
    cols = [args.mention_col, 'labels']
    cols_mapping = {args.mention_col: 'text', 'labels': 'label'}
else:
    # using concat strategy
    sep_tok = tokenizer.sep_token if args.concat_sep_token is None else args.concat_sep_token
    if args.concat_strategy == 'prefix':
        df['input'] = df[args.mention_col] + sep_tok + df[args.text_col]
    elif args.concat_strategy == 'suffix':
        df['input'] = df[args.text_col] + sep_tok + df[args.mention_col]
    else:
        raise ValueError(f"Unknown concat strategy: {args.concat_strategy}")
    max_length_ = max(tokenizer(df['input'].to_list(), truncation=False, padding=False, return_length=True).length)
    cols = ['input', 'labels']
    cols_mapping = {"input": "text", "labels": "label"}


# ### split the data


datasets = DatasetDict({
    s: Dataset.from_pandas(d, preserve_index=False)
    for s, d in df.groupby('split', observed=True)
})
datasets = datasets.remove_columns(set(df.columns)-set(cols))
datasets = datasets.rename_columns(column_mapping=cols_mapping)


# ## HP search


# ### Prepare fine-tuning


id2label = {i: l for i, l in enumerate(args.label_cols)}
label2id = {l: i for i, l in id2label.items()}


class_weights = None
if args.class_weighting_strategy in ['inverse_proportional']:
    class_weighting_args = {
        "multitarget": len(args.label_cols) > 1,
        "smooth_weights": args.class_weighting_strategy is not None and args.class_weighting_strategy != 'balanced',
        "smooth_exponent": args.class_weighting_smooth_exponent,
    }
    class_weights = compute_class_weights(datasets['train']['label'], **class_weighting_args)


from sentence_transformers.losses import ContrastiveLoss

from tempfile import TemporaryDirectory
with TemporaryDirectory() as tmpdirname:
    model_dir = tmpdirname

training_args = EarlyStoppingTrainingArguments(
    output_dir=model_dir,
    
    # set non-tunable hyperparameters
    loss=ContrastiveLoss,
    num_epochs=(1, 25),

    body_learning_rate=(2e-04, 1e-05), # default is (2e-05, 1e-05)

    # sentence transformer (embedding) finetuning args
    logging_first_step=False,
    eval_strategy="steps",
    eval_steps=50,
    max_steps=750,
    eval_max_steps=250,
    
    # early stopping config
    metric_for_best_model=("embedding_loss", "f1"),
    greater_is_better=(False, True),
    load_best_model_at_end=True,
    save_total_limit=2, # NOTE: currently no effect on (early stopping in) classification head training
    
    # misc
    end_to_end=True,
)

# NOTE: needed to avoid version mismatch issues during HP search
if not hasattr(training_args, 'process_index'):
    training_args.process_index = 0  # force single-process training for determinism
if not hasattr(training_args, 'world_size'):
    training_args.world_size = 1

training_callbacks = [
    # for sentence transformer finetuning
    EarlyStoppingCallback(
        early_stopping_patience=args.body_early_stopping_patience,
        early_stopping_threshold=args.body_early_stopping_threshold,
    ), 
    # for classifier finetuning
    EarlyStoppingCallback(
        early_stopping_patience=args.head_early_stopping_patience,
        early_stopping_threshold=args.head_early_stopping_threshold,
    ), 
]


from typing import Any
def hp_search_model_init(params: Dict[str, Any]):
    # NOTE: not used
    params = params or {}
    
    model = model_init(
        model_name=args.model_name,
        id2label=id2label, #num_classes=len(id2label),
        multilabel=True,
        class_weights=class_weights,
        use_span_embedding=args.use_span_embeddings,
    )

    return model


trainer_class = SetFitTrainerForSpanClassification if args.use_span_embeddings else SetFitEarlyStoppingTrainer
trainer = trainer_class(
    model_init=hp_search_model_init,
    metric="f1",
    metric_kwargs={"average": "macro" if args.label_cols and len(args.label_cols) > 1 else "binary"},
    args=training_args,
    train_dataset=datasets['train'],
    eval_dataset=datasets['val'],
    callbacks=training_callbacks,
)

# fix max_length issue
trainer._args.max_length = min(trainer.st_trainer.model.tokenizer.model_max_length, int(max_length_*1.1))

# set seeds for reproducibility
trainer._args.seed = SEED
trainer.st_trainer.args.seed = SEED
trainer.st_trainer.args.data_seed = SEED
trainer.st_trainer.args.full_determinism = True

# don't report to wandb or other experiment trackers
trainer._args.report_to = 'none'
trainer.st_trainer.args.report_to = 'none'


# ### Perform HP search

# TODO: create `study_name` based on args`
best_run = trainer.hyperparameter_search(hp_space=hp_space, n_trials=args.n_trials, direction="maximize", sampler = TPESampler(seed=trainer.args.seed))


# clean up
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)


#from optuna import TrialFrozen
def parse_frozen_trial(t, metric_name='f1'):
    return {
        "trial_id": t._trial_id,
        "started_at": t.datetime_start.isoformat(timespec='seconds'),
        "finished_at": t.datetime_complete.isoformat(timespec='seconds'),
        "duration": int(t.duration.total_seconds()),
        **t.params,
        metric_name: t.values[0],
    }


trial_results = pd.DataFrame([parse_frozen_trial(trial) for trial in best_run.backend.get_trials()])

fp = args.save_results_to / 'trial_results.csv'
trial_results.to_csv(fp, index=False)


trial_results


# ### Evaluate


if not args.train_best_model:
    exit(0)


# ## Train model with best hyperparams 


print("training best model with hyperparameters:", best_run.hyperparameters)


# apply best hyper parameters
trainer.apply_hyperparameters({**best_run.hyperparameters, 'eval_steps': 50, "max_eval_steps": 500}, final_model=True)


trainer.train()


# clean up
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)


# #### Evaluate


def get_predictions_df(split: str = 'val') -> pd.DataFrame:
    preds_df = df.loc[df['split'] == split, [args.id_col, args.text_col, args.mention_col, args.span_col, *args.label_cols]].copy()
    
    inputs = trainer.model._normalize_inputs(texts=datasets[split]['text'], spans=datasets[split]['span']) if args.use_span_embeddings else datasets[split]['text']
    
    probs = trainer.model.predict_proba(inputs, as_numpy=True)
    prob_cols = [f"prob_{col}" for col in trainer.model.labels]
    preds_df[prob_cols] = probs
    
    preds = np.where(probs > 0.5, 1, 0)
    pred_cols = [f"pred_{col}" for col in trainer.model.labels]
    preds_df[pred_cols] = preds
    
    for lab in trainer.model.labels:
        preds_df[f"error_{lab}"] = preds_df[f"pred_{lab}"] != preds_df[lab]
    
    return preds_df


inputs = trainer.model._normalize_inputs(texts=datasets['val']['text'], spans=datasets['val']['span']) if args.use_span_embeddings else datasets['val']['text']
preds = trainer.model.predict(inputs, as_numpy=True)
print(classification_report(y_pred=preds, y_true=datasets['val']['label'], target_names=args.label_cols, zero_division=0))


if args.save_eval_results:
    res = classification_report(y_pred=preds, y_true=datasets['val']['label'], target_names=args.label_cols, zero_division=0, output_dict=True)
    fp = args.save_results_to / 'eval_results.json'
    with open(fp, 'w') as f:
        json.dump(res, f)


if args.save_eval_predictions:
    preds_df = get_predictions_df()
    fp = args.save_results_to / 'eval_predictions.pkl'
    preds_df.to_pickle(fp)


# ## Train final model


if not args.train_final_model:
    exit(0)


# ### get optimal number of training steps for body and head


# determine "optiomal" embedding model training steps
body_log_history = [{**trn, **evl} for trn, evl in zip(trainer.st_trainer.state.log_history[0::2], trainer.st_trainer.state.log_history[1::2])]
body_log_history = pd.DataFrame(body_log_history)
body_metric = trainer.args.metric_for_best_model[0]
idx = body_log_history[f"eval_{body_metric}"].idxmin()
optimal_body_steps = body_log_history.step[idx].item()
print(f"training final model's body for {optimal_body_steps} steps")


head_log_history = pd.DataFrame(trainer.model.head_log_history)
head_metric = trainer.args.metric_for_best_model[1]
idx = head_log_history[f"eval_{head_metric}"].idxmax()
optimal_head_epochs = head_log_history.epoch[idx].item()
print(f"training final model's head for {optimal_head_epochs} epochs")


# update training args to optimal steps/epochs
training_args = trainer._to_setfit_train_args(training_args)

training_args.eval_strategy = "none"
training_args.max_steps = optimal_body_steps
training_args.num_epochs = (1, optimal_head_epochs)
training_args.save_total_limit = None


# combine dev ant train dataset
train_ds = concatenate_datasets([datasets['train'], datasets['val']])


trainer = Trainer(
    model_init=hp_search_model_init,
    args=training_args,
    train_dataset=train_ds
)

# fix max_length issue
trainer._args.max_length = min(trainer.st_trainer.model.tokenizer.model_max_length, int(max_length_*1.1))

# set seeds for reproducibility
trainer._args.seed = SEED
trainer.st_trainer.args.seed = SEED
trainer.st_trainer.args.data_seed = SEED
trainer.st_trainer.args.full_determinism = True

# don't report to wandb or other experiment trackers
trainer._args.report_to = 'none'
trainer.st_trainer.args.report_to = 'none'


trainer.train()


trainer.model.model_body.eval();
trainer.model.model_head.eval();


# ## Save the model



if args.save_final_model_to:
    trainer.model.save_pretrained(args.save_final_model_to)



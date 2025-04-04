
import os
import pandas as pd
import numpy as np
import torch

from transformers import pipeline

from tqdm import tqdm
from utils.nli import (
    STANCE_HYPOTHESIS_TEMPLATE,
    STANCE_LABEL_CLASSES,
    clean_text,
    ZeroShotMentionClassificationArgumentHandler,
    clean_memory
)


from types import SimpleNamespace

args = SimpleNamespace()

args.model_path = '../../models/nli_stance_classifier'


args.input_file = '../../data/labeled/manifesto_sentences_predicted_group_mentions_spans.tsv'
args.entity_type_col = 'label'
args.entity_type = 'social group'
args.sentence_text_col = 'sentence_text'
args.mention_text_col = 'text'
args.id_cols = 'sentence_id,span_nr'

args.batch_size = 64
args.chunk_size = 640

args.output_file = '../../data/labeled/manifesto_sentences_predicted_social_group_mentions_stance.tsv'

args.test = True
args.n_test = 1000

# parse
args.id_cols = [c.strip() for c in args.id_cols.split(',')]

# process

df = pd.read_csv(args.input_file, sep="\t")

df = df[df[args.entity_type_col]==args.entity_type]
del df[args.entity_type_col]

if args.test:
    df = df.iloc[:args.n_test]
df = df.reset_index(drop=True)

cols = args.id_cols + [args.sentence_text_col, args.mention_text_col]
df = df[cols]
df['input'] = df[args.sentence_text_col].apply(clean_text)
df.loc[:, "sentence_text_prepared"] = 'The quote: """' + df['input'].fillna("") + '""" - end of the quote.'
df = df[cols + ['sentence_text_prepared']]


classifier = pipeline(
    task='zero-shot-classification',
    model=args.model_path,
    framework='pt',
    args_parser=ZeroShotMentionClassificationArgumentHandler()
)


def _predict_batch(df, batch_size=64):
    inputs = df[['sentence_text_prepared', args.mention_text_col]].apply(tuple, axis=1).to_list()
    preds = classifier(
        inputs,
        candidate_labels=STANCE_LABEL_CLASSES,
        hypothesis_template=STANCE_HYPOTHESIS_TEMPLATE,
        multi_label=False,
        batch_size=batch_size
    )
    preds_df = pd.concat(pd.DataFrame(pred['scores'], index=pred['labels']).T for pred in preds)
    preds_df['pred'] = preds_df.columns[preds_df.values.argmax(axis=1)]

    preds_df = pd.concat([
        df[cols].reset_index(drop=True),
        preds_df.reset_index(drop=True)
    ], axis=1)

    return preds_df


n = len(df)
for i in tqdm(range(0, n, args.chunk_size)):
    if i + chunk_size > n:
        chunk_size = n - i
    preds_df = _predict_batch(df.iloc[i:i+chunk_size], batch_size=args.batch_size)
    if i == 0:
        preds_df.to_csv(args.output_file, sep="\t", index=False)
    else:
        preds_df.to_csv(args.output_file, sep="\t", mode='a', header=False, index=False)
    del preds_df
    clean_memory()
    

import re
import warnings

import numpy as np
import pandas as pd

from seqeval import metrics as seqeval_metrics
from sklearn import metrics as sklearn_metrics

from typing import List,  Dict, Tuple, Optional, Union, Literal

# TODO: there should be an equivalent implementation in seqeval utils 
def extract_spans(labels: List[str], tokens: Optional[List[str]]=None):
    """
    Extracts all spans from a sequence of predicted token labels.

    This function assumes that the labels are in IOB2 scheme, where each span
    starts with a 'B' tag (for beginning) and subsequent tokens in the same span
    are tagged with an 'I' tag (for inside). Tokens outside of any span are
    tagged with an 'O' tag (for outside).

    Args:
        labels (list): A list of predicted labels for each token in the input
            sequence.
        tokens (list, optional): A list of tokens in the input sequence. If
            provided, the function will return the tokens for each span in the
            output. If not provided, the function will only return the start and
            end indices of each span in the input sequence.

    Returns:
        list: A list of tuples, where each tuple contains the following
            elements:
            - A list of tokens in the span (None if `tokens=None`).
            - The type of the span.
            - The start index of the span in the input sequence.
            - The (exclusive) end index of the span in the input sequence.
    """
    spans = []
    current_span = []
    current_type = None
    current_start = None
    
    for i, label in enumerate(labels):
        if label == 'O':
            if current_span:
                # End the current span and add it to the list of spans.
                spans.append([current_span, current_type, current_start, i])
                current_span = []
                current_type = None
                current_start = None
        elif label.startswith('B-'):
            if current_span:
                # End the current span and add it to the list of spans.
                spans.append([current_span, current_type, current_start, i])
            # Start a new span.
            current_span = [tokens[i]] if tokens is not None else [None]
            current_type = label[2:]  # Remove the 'B-' prefix.
            current_start = i
        elif label.startswith('I-'):
            if current_span and current_type == label[2:]:
                # Add the current token to the current span.
                if tokens is not None:
                    current_span.append(tokens[i])
                else:
                    current_span.append(None)
            else:
                if current_span:
                    # End the current span and add it to the list of spans.
                    spans.append([current_span, current_type, current_start, i])
                # Start a new span.
                current_span = [tokens[i]] if tokens is not None else [None]
                current_type = label[2:]  # Remove the 'I-' prefix.
                current_start = i - 1
    
    if current_span:
        # End the final span and add it to the list of spans.
        spans.append([current_span, current_type, current_start, len(labels)])
    
    if tokens is None:
        for span in spans:
            span[0] = None
    return [tuple(span) for span in spans]

# # test
# tokens = ['Today', ',', 'Barack', 'Obama', 'and', 'Justin', 'Trudeau', 'meet', 'in', 'Osnabrügge', '.']
# labels = ['O',     'O', 'B-PER',  'I-PER', 'O',   'B-PER',  'I-PER',   'O',    'O',  'B-LOC',      'O']
# print(extract_spans(labels, tokens))
# print(extract_spans(labels))



def compute_span_score(spans: List[Tuple], ref: List[str], average: Literal[None, 'micro', 'macro']=None, use_length=False):
    """
    Compute score for spans in a sequence

    Given a list of spans and a list of reference labels, this function computes the
    accuracy of spans given the reference labels.

    Using predicted spans and true sequence labels as reference, this function computes the precision score.
    Using observed ("true") spans and predicted sequence labels as reference, this function computes the recall score.

    The function first computes the accuracy for each span against the labels of the corresponding elements in the reference sequence.
    It then averages this span-wise scores (weighting by spans' length if `use_length=True`) across all spans in the sequence.

    Args:
        spans (list): A list of spans, recording for each span (as a 4-tuple)
            the list of its tokens (or None), the span's type (str),
            and the span's start and end index (ints).
        ref (list): A list of predicted or observed labels for each token in the input
            sequence.
        average (None, 'micro', or 'macro'): how to aggregate the span-wise accuracy scores:
           - `None`: by span type
           - `'micro'`: ignoring span types
           - `'macro'`: first by span types, then across span types
        use_length (bool): whether or not to weight span-wise scores by span length
    
    Returns:
        float if `average='micro'`, the average span-wise accuracy scores (ignoring spans' types).
        float if `average='macro'`, the average of type-specific averages of span-wise accuracy scores (first by span types, then across span types).
        dict if `average=None`, mapping span types to their average span-wise accuracy scores.
    """
    # TODO: Decide whether to weight span-wise scores by span length.
    # Why? Consider the following example:
    #
    #   obs    = ['O',     'O', 'B-PER',  'I-PER', 'O',   'B-PER',  'I-PER',   'O',    'O',     'B-LOC' ]
    #   pred   = ['B-LOC', 'O', 'B-PER',  'I-PER', 'O',   'B-PER',  'O',       'O',    'B-LOC', 'I-LOC' ]
    #              ^ FP          ^ TP      ^ TP            ^ TP      ^ FN               ^ FP     ^ TP
    #
    # Focus on the LOC type:
    #  - Precision is 0.0 for the first predicted span and 0.5 for the second.
    #  - The average of these two span-wise scores is 0.25 this seems odd give that we got 1 out of 3 tokens right
    # Caveat: But if we weight by length, it might be the same as token-level (cross-sequence) precision, recall, and F1

    if average: assert average in ['micro', 'macro'], "`average` must be None, 'micro' or 'macro'"

    if len(spans) == 0:
        # return {} if average is None else 0.0
        return {} if average is None else np.nan
    # note: if no spans have been retrieved, we cannot evaluate correctness against the reference labels
    #  - if we focus on _predicted_ spans (and take the true sequence labels as a reference), this conforms 
    #    with the defintion of precision = TP/(TP+FP), which is undefined if there are no positive labels predicted 
    #    (because in this case there will be neither true nor false positives)
    #  - if we focus on _true_ spans (and take the predicted sequence labels as a reference), this conforms 
    #    with the defintion of recall = TP/(TP+FN), which is undefined if there are no positive labels (and 
    #    hence no "true" spans) because in this case there will be neither true positives (since there are no spans 
    #    in the ground truth) nor false negatives (since any negatice is correct)
    # caveat: sklearn handles it differently
    #  - `sklean.metrics.precision_score(y_true=[], y_pred=[], average='micro')` raises warning but returns 0.0
    #  - `sklean.metrics.recall_score(y_true=[], y_pred=[], average='micro')` raises warning but returns 0.0

    if average == 'micro':
        scores = list()
        total_length = 0 if use_length else len(spans)
        for _, typ, s, e in spans:
            span_length = e - s
            if use_length:
                total_length += span_length
            # compute number of tokens in the reference sequence for which the span label matches
            n_correct = sum(label[2:] == typ for label in ref[s:e])  # IMPORTANT: this ignores the B-/I- distinction
            # compute agreement share
            score = n_correct / span_length
            scores.append(score * span_length if use_length else score)  # weight by span length if use_length
        return np.sum(scores) / total_length  # weighted or unweighted average
    else:
        types = set(s[1] for s in spans)
        scores = {t: [] for t in types}
        lengths = {t: [] for t in types}
        for _, typ, s, e in spans:
            span_length = e - s
            # compute number of tokens in the reference sequence for which the span label matches
            n_correct = sum(label[2:] == typ for label in ref[s:e])  # IMPORTANT: this ignores the B-/I- distinction
            # compute agreement share
            score = n_correct / span_length
            scores[typ].append(score * span_length if use_length else score)  # weight by span length if use_length
            lengths[typ].append(span_length)

        # average within type (weighting by span lengths if needed)
        scores = {t: np.sum(s) / np.sum(lengths[t]) for t, s in scores.items()} if use_length else {t: np.mean(s) for t, s in scores.items()}

        # if no (macro) aggregation, return type-specific averages
        if average is None:
            return scores
        
        # otherwise average across types (weighting by span lengths if needed)
        total_weight = sum(np.sum(lengths[t]) for t in types)
        scores = sum((np.sum(lengths[t]) / total_weight) * scores[t] for t in types) if use_length else np.mean(list(scores.values()))
        return scores
# # test
# obs    = ['O',     'O', 'B-PER',  'I-PER', 'O',   'B-PER',  'I-PER',   'O',    'O',  'B-LOC']
# pred   = ['B-LOC', 'O', 'B-PER',  'I-PER', 'O',   'B-PER',  'O',       'O',    'O',  'B-LOC']
# #          ^FP           ^TP       ^TP             ^TP       ^FN                      ^TP                
# print('Recall:')
# print(compute_span_score(extract_spans(obs), ref=pred))
# # precision
# print('Precision:')
# print(compute_span_score(extract_spans(pred), ref=obs))


def spanwise_scores(y_true: List[str], y_pred: List[str], average: Literal[None, 'micro', 'macro']=None, zero_division: Union[None, float]=0.0, **kwargs):
    """
    Compute cross-sequence average spanwise precision, recall, and F1 scores.

    Given a lists of list of samples' observed sequence labels and a corresponding list of list of predicted sequence labels,
    this function computes the cross-sequence average precision, recall, and F1 score.

    The cross-sequence average of a score is an estimated for the expected classification performance for a randomly sampled
     sequence from the population of sequences because it first computes sequence-specific average scores that average across 
     predicted and observed spans in a sequence and only then averages these sequence-specific average scores.
     Hence, the name cross-sequence average.
    
    Args:
        y_true (list): A list of samples' observed labels (one list of labels per sample).
        y_pred (list): A list of samples' predicted labels (one list of labels per sample).
        average (None, 'micro', or 'macro'): how to aggregate the span-wise scores:
           - `None`: by span type
           - `'micro'`: ignoring span types
           - `'macro'`: first by span types, then across span types
        zero_division: how to handle zero devision. Pass None, float value (default: 0.0), or `np.nan` to return 
           in the case of zero division at the batch level
        **kwargs: forwarded to compute_span_score
    Returns:
        if `average='micro'`, a 3-tuple of the cross-sequence average of sequence-specific average precision, recall, and F1 scores (ignoring spans' types).
        if `average='macro'`, a 3-tuple of the cross-sequence average of sequence- and type-specific averages of span-wise precision, recall, and F1 scores.
        if `average=None` (default), a mapping of span types to their cross-sequence average of sequence-specific average span-wise precision, recall, and F1 scores.
    """
    # TODO: Here we do a lot of averaging across spans (by type) within individual sequences
    #        and then averaging of these within-averages across sequences.
    #       Maybe it'd be better to collect span-wise metrics and do a global average 
    #        (maybe weighted by spans' number of tokens)
    #       But this would require to modify the output of `compute_span_score`, too

    if average: assert average in ['micro', 'macro'], "`average` must be None, 'micro' or 'macro'"
    
    # helper
    def f1score(p, r): 
        if np.isnan(p) or np.isnan(r):
            return np.nan
        elif p+r == 0:
            return zero_division
        else:
            return 2*((p*r)/(p+r))

    if average is not None:
        # note: handling of micro/macro average distinction occurs in compute_span_score

        # create list to store sequences' averages of span-wise scores
        scores = list()
        for o, p in zip(y_true, y_pred):
            # compute spans' average precision
            prec = compute_span_score(extract_spans(p), ref=o, average=average, **kwargs)
            # compute spans' average recall
            recl = compute_span_score(extract_spans(o), ref=p, average=average, **kwargs)
            # compute spans' F1 (note: undefined if precision and/or recall undefined)
            f1 = f1score(prec, recl)
            # add to collector
            scores.append((prec, recl, f1))
        # average across sequences
        scores = np.array([*scores])
        if np.isnan(scores).all():
            return np.nan, np.nan, np.nan
        scores = np.nanmean(scores, axis=0)
        # compute f1 score from aggregated precision and recall scores
        scores = np.append(scores, f1score(scores[0], scores[1]))
        return tuple(scores)
    else:
        # get all types in y_true and pred labels
        types = set(l[2:] for o in y_true for l in o if l != 'O')
        types.update(set(l[2:] for o in y_pred for l in o if l != 'O'))
        # create dicts to store sequences' averages of span-wise scores
        precs, recls, f1s = [{t: [] for t in types} for _ in range(3)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)    
            for o, p in zip(y_true, y_pred):
                # compute type-specific average of span-wise precisions
                prec = compute_span_score(extract_spans(p), ref=o, average=None)
                for k, v in prec.items(): precs[k].append(v)
                # compute type-specific average of span-wise recall
                recl = compute_span_score(extract_spans(o), ref=p, average=None)
                for k, v in recl.items(): recls[k].append(v)
                # compute type-specific F1 for sequence (note: undefined if precision and/or recall undefined)
                f1 = {t: f1score(prec[t], recl[t]) for t in types if t in prec and t in recl}
                for k, v in f1.items(): f1s[k].append(v)
            # compute cross-sequence averages of type- and sequence-specific cross-span averages
            precs = {k: np.nanmean(np.array(v)) if len(v) > 0 else np.nan for k,v in precs.items()}
            recls = {k: np.nanmean(np.array(v)) if len(v) > 0 else np.nan for k,v in recls.items()}
            f1s   = {k: np.nanmean(np.array(v)) if len(v) > 0 else np.nan for k,v in f1s.items()}
        # transpose to dict
        scores = {
            t: (
                precs[t] if t in precs else None, 
                recls[t] if t in recls else None, 
                f1s[t] if t in f1s else None,
                # compute f1 score from aggregated precision and recall scores
                f1score(precs[t], recls[t]) if t in precs and t in recls else None
                ) 
            for t in types
        }
        return scores
# # test
# o = ['O',     'O', 'B-PER',  'I-PER', 'O',   'B-PER',  'I-PER',   'O',    'O',     'B-LOC' ]
# p = ['B-LOC', 'O', 'B-PER',  'I-PER', 'O',   'B-PER',  'O',       'O',    'B-LOC', 'I-LOC' ]
# #     ^ FP          ^ TP      ^ TP            ^ TP      ^ FN               ^ FP     ^ TP
# y_true = [o, o[:4], o[4:]]
# y_pred = [p, p[:4], p[4:]]
# # LOC 
# #  - precision: 0.25 in 1st sequence, 0.00 in 2nd sequence and 0.50 in 3rd sequence => (0.25 + 0.00 + 0.50)/3 = 0.25 
# #  - recall: 1.00 in 1st sequence and 3rd sequence => (1.00 + 1.00)/2 = 1.00
# # PER
# #  - precision: 1.00 in 1st, 2nd and 3rd sequence => (1.00 + 1.00 + 1.00)/3 = 1.00
# #  - recall: 0.75 (1.0 + 0.5) in 1st, 2nd and 3rd sequence => (1.00 + 1.00)/2 = 1.00
# print(spanwise_scores(y_true, y_pred, average=None))



def _correct_iob2(labels: List[str]):
    prev = None
    edit = list()
    for i, l in enumerate(labels):
        if (i == 0 or prev == 'O') and l[0] == 'I':
            edit.append(i)
        prev = l
    if len(edit) > 0:
        labels = [l.replace('I-', 'B-') if i in edit else l for i, l in enumerate(labels)]
    return labels

def _validate_eval_fun_inputs(
        y_true: Union[List[Union[int,str]], List[List[Union[int,str]]]], 
        y_pred: Union[List[Union[int,str]], List[List[Union[int,str]]]], 
        id2label: Dict[int, str]
    ):

    # add "outside" token label if needed
    if 0 not in id2label:
        id2label[0] = 'O'

    # convert to list of list if needed
    if isinstance(y_true[0], int):
        y_true = [y_true]
    if isinstance(y_pred[0], int):
        y_pred = [y_pred]
    
    if (
        # check if true and predicted labels already encoded as strings
        all(isinstance(l, str) for labs in y_true for l in labs)
        and
        all(isinstance(l, str) for labs in y_pred for l in labs)
    ):
        pass
    else:
        # check the labels
        unknown = set([l for x in y_true for l in x if l != -100 and l not in id2label])
        if len(unknown) > 0: raise ValueError(f'Unknown label(s) in y_true: {list(unknown)}')
        
        # check the predictions
        unknown = set([l for x in y_pred for l in x if l != -100 and l not in id2label])
        if len(unknown) > 0: raise ValueError(f'Unknown label(s) in y_pred: {list(unknown)}')
        
        # discard labels for inside-word tokens (for compatibility with output from fine-tuned token classifier)
        preds = [ [p for (p, o) in zip(preds, obs) if o != -100] for preds, obs in zip(y_pred, y_true) ]
        obs = [ [o for (_, o) in zip(preds, obs) if o != -100] for preds, obs in zip(y_pred, y_true) ]

        # apply string labels
        y_true = [[id2label[l] for l in x] for x in obs]
        y_pred = [[id2label[l] for l in x] for x in preds]

    # TODO: consider applying viterbi decoding (see https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling?tab=readme-ov-file#viterbi-decoding)
    # y_true = _correct_iob2(y_true)
    # y_pred = _correct_iob2(y_pred)
    
    return y_true, y_pred, id2label

def compute_seqeval_metrics(
        y_true: Union[List[Union[int,str]], List[List[Union[int,str]]]], 
        y_pred: Union[List[Union[int,str]], List[List[Union[int,str]]]], 
        id2label: Dict[int, str]
    ) -> Dict:
    
    y_true, y_pred, id2label = _validate_eval_fun_inputs(y_true, y_pred, id2label)
    
    return seqeval_metrics.classification_report(y_true=y_true, y_pred=y_pred, output_dict=True, zero_division=0.0)
# # test
# o = ['O',     'O', 'B-PER',  'I-PER', 'O',   'B-PER',  'I-PER',   'O',    'O',     'B-LOC' ]
# p = ['B-LOC', 'O', 'B-PER',  'I-PER', 'O',   'B-PER',  'O',       'O',    'B-LOC', 'I-LOC' ]
# y_true = [o, o[:4], o[4:]]
# y_pred = [p, p[:4], p[4:]]
# 
# id2label = {1: 'I-LOC', 2: 'I-PER', 3: 'B-LOC', 4: 'B-PER'}
# 
# compute_seqeval_metrics(y_true, y_pred, id2label)


def compute_sequence_metrics(
        y_true: Union[List[int], List[List[int]]], 
        y_pred: Union[List[int], List[List[int]]], 
        id2label: Dict[int, str],
        flatten_output: bool=False
    ) -> Dict:
    
    # define metrics to report
    mets=['precision', 'recall', 'f1-score']

    types = list(set([l[2:] for l in id2label.values() if l != 'O']))

    y_true, y_pred, id2label = _validate_eval_fun_inputs(y_true, y_pred, id2label)
    
    # dict for collecting the metrics
    results = {}    

    # Span level (Seqeval)
    result = seqeval_metrics.classification_report(y_true, y_pred, output_dict=True, zero_division=0.0)
   
    keys = ['macro avg', 'micro avg'] + types
    result = {k: result[k] for k in keys if k in result}
    result = {
        # format: metric name <=> metric value
        str(f"{k.replace(' avg', '')}_{m.replace('-score', '')}"): res[m] 
        # iterate over class-wise results
        for k, res in result.items()
        # iterate over metrics
        for m in mets
    }
    results['seqeval'] = result
    
    # Span level (relaxed averages)
    m = ['precision', 'recall', 'f1-cross', 'f1']
    result = {}
    for avg in ['macro', 'micro']:
        result |= {f'{avg}_{m[i]}': s for i, s in enumerate(spanwise_scores(y_true, y_pred, average=avg))}
    result |= {f'{typ}_{m[i]}': score for typ, scores in spanwise_scores(y_true, y_pred).items() for i, score in enumerate(list(scores))}
    results['spanlevel'] = result
    
    # Document level
    overall = [[], []]
    by_type = {t: [[], []] for t in types}
    for o, p in zip(y_true, y_pred):
        overall[0].append(int(any(l != 'O' for l in o)))
        overall[1].append(int(any(l != 'O' for l in p)))
        for t in types:
            by_type[t][0].append(int(any(t in l for l in o)))
            by_type[t][1].append(int(any(t in l for l in p)))
    result = dict()
    p, r, f1, _ = sklearn_metrics.precision_recall_fscore_support(overall[0], overall[1], average='micro', zero_division=0.0)
    result['micro_precision'] = p
    result['micro_recall'] = r
    result['micro_f1'] = f1
    for t in types:
        p, r, f1, _ = sklearn_metrics.precision_recall_fscore_support(by_type[t][0], by_type[t][1], average='micro', zero_division=0.0)
        result[t+'_precision'] = p
        result[t+'_recall'] = r
        result[t+'_f1'] = f1
    results['doclevel'] = result
    
    # Word level
    # flatten sequence labels
    y_pred = [l if l=='O' else l[2:] for labs in y_pred for l in labs]
    y_true = [l if l=='O' else l[2:] for labs in y_true for l in labs]

    result = sklearn_metrics.classification_report(y_true, y_pred, output_dict=True, zero_division=0.0)
    out = {'accuracy': result['accuracy']}
    keys = ['macro avg'] + ['O'] + types
    result = {k: result[k] for k in keys if k in result}
    tmp = {
        # format: metric name <=> metric value
        str(f"{c.replace(' avg', '')}_{m.replace('-score', '')}"): res[m] 
        # iterate over class-wise results
        for c, res in result.items()
        # iterate over metrics
        for m in mets
    }
    out.update(tmp)
    results['wordlevel'] = out
    
    if flatten_output:
        results = {k+'-'+m: v for k, res in results.items() for m, v in res.items()}
    return results
# # test
# o = ['O',     'O', 'B-PER',  'I-PER', 'O',   'B-PER',  'I-PER',   'O',    'O',     'B-LOC' ]
# p = ['B-LOC', 'O', 'B-PER',  'I-PER', 'O',   'B-PER',  'O',       'O',    'B-LOC', 'I-LOC' ]
# y_true = [o, o[:4], o[4:]]
# y_pred = [p, p[:4], p[4:]]
# 
# id2label = {0: 'O', 1: 'I-LOC', 2: 'I-PER', 3: 'B-LOC', 4: 'B-PER'}
# 
# compute_sequence_metrics(y_true, y_pred, id2label)


def parse_metrics(x: Dict, order: Optional[List[str]]=None):
    out = pd.DataFrame(x, index=['value']).T
    out = out.reset_index().rename(columns={'index': 'cat'})
    out[['label', 'metric']] = out.cat.str.split('_', expand=True)
    # pivot table using 'metric' as columns and 'value' as values (like tidyr::pivot_wider)
    out = out.pivot(index='label', columns='metric', values='value').reset_index()
    # drop the name from the index axis
    out = out.rename_axis(None, axis=1)
    # set values in 'label' as index
    out = out.set_index('label')
    # drop the name from the index axis
    out.index.name = None
    if order:
        out = out.loc[order, ]

    return out
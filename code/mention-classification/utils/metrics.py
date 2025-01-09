import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

def compute_metrics_binary(y_pred, y_true):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    out = {
        'accuracy': acc,
        'balanced_accuracy': bacc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    return out

def compute_metrics_multiclass(y_pred, y_true, id2label):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0.0)
    out = {
        'accuracy': acc,
        'balanced_accuracy': bacc,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall
    }
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(id2label.keys()))
    for i, l in id2label.items():
        out[f'precision_{l}'] = precision[i]
        out[f'recall_{l}'] = recall[i]
        out[f'f1_{l}'] = f1[i]

    return out

def compute_metrics_multilabel(y_pred, y_true, id2label):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    scores = {}
    for i, l in id2label.items():
        p, r, f1, _ = precision_recall_fscore_support(y_true[:, i], y_pred[:, i], average='binary', zero_division=0.0)
        scores[l] = {'f1': f1, 'precision': p, 'recall': r, 'support': np.sum(y_true[:, i])}
    macros = {m: np.mean([d[m] for d in scores.values()]) for m in ['f1', 'precision', 'recall']}
    scores = {'macro': macros} | scores
    # flatten
    scores = {f'{m}_{l}': v for l, d in scores.items() for m, v in d.items()}
    return scores


def compute_metrics_hierarchical_multilabel(y_pred, y_true, id2label, label_sep=': '):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # granularly
    granular_scores = {}
    for i, l in id2label.items():
        p, r, f1, _ = precision_recall_fscore_support(y_true[:, i], y_pred[:, i], average='binary', zero_division=0.0)
        scores[l] = {'f1': f1, 'precision': p, 'recall': r, 'support': np.sum(y_true[:, i])}
    macros = {m: np.mean([d[m] for d in granular_scores.values()]) for m in ['f1', 'precision', 'recall']}
    granular_scores = {'macro_granular': macros} | granular_scores
    # coarse
    granular2coarse = {i: l.split(label_sep)[0] for i, l in id2label.items()}
    coarse_cats = set(granular2coarse.values())
    y_true = [granular2coarse[i] for i in y_true]
    y_pred = [granular2coarse[i] for i in y_pred]
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0.0)
    coarse_scores = {}
    coarse_scores['macro_coarse'] = {'f1': np.mean(f1), 'precision': np.mean(p), 'recall': np.mean(r)}
    for i, l in coarse_cats:
        coarse_scores[l] = {'f1': f1[i], 'precision': p[i], 'recall': r[i]}
    # flatten
    scores = coarse_scores | granular_scores
    scores = {f'{m}_{l}': v for l, d in scores.items() for m, v in d.items()}
    return scores
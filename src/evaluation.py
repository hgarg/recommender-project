"""
Evaluation utilities for recommendation quality.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from pipeline import topk


def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=42):
    """Bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return np.nan, np.nan, np.nan

    sample_idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    sample_means = values[sample_idx].mean(axis=1)

    mean = values.mean()
    lower = np.percentile(sample_means, 100 * alpha / 2)
    upper = np.percentile(sample_means, 100 * (1 - alpha / 2))

    return mean, lower, upper


def sampled_roc(scores, state, eval_df, n_neg=100, seed=0):
    """ROC-AUC over liked items against sampled unseen negatives."""
    rng = np.random.default_rng(seed)
    truth = eval_df[eval_df.rating >= 4].groupby("user_id")["product_id"].apply(set).to_dict()

    y_true = []
    y_score = []

    for user_id, relevant in truth.items():
        user_idx = state["u_idx"][user_id]
        positives = [state["p_idx"][product_id] for product_id in relevant]

        candidate_idx = np.setdiff1d(np.where(~state["seen_mask"][user_idx])[0], positives)
        negatives = rng.choice(candidate_idx, size=min(n_neg, len(candidate_idx)), replace=False)

        y_true.extend([1] * len(positives) + [0] * len(negatives))
        y_score.extend(list(scores[user_idx, positives]) + list(scores[user_idx, negatives]))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr, roc_auc_score(y_true, y_score)


def topk_confusion(scores, state, eval_df, k=10):
    """Top-k confusion matrix for liked items among unseen candidates."""
    truth = eval_df[eval_df.rating >= 4].groupby("user_id")["product_id"].apply(set).to_dict()
    recs = topk(scores, state["seen_mask"], k)

    tp = fp = fn = tn = 0

    for user_id, relevant in truth.items():
        user_idx = state["u_idx"][user_id]
        candidate_items = set(np.where(~state["seen_mask"][user_idx])[0])
        relevant_items = {state["p_idx"][product_id] for product_id in relevant}
        recommended = set(recs[user_idx])

        tp += len(recommended & relevant_items)
        fp += len(recommended - relevant_items)
        fn += len(relevant_items - recommended)
        tn += len(candidate_items - recommended - relevant_items)

    return np.array([[tn, fp], [fn, tp]])


def metrics_at_k(scores, state, eval_df, ks=range(1, 21)):
    """Precision, recall, and NDCG across a range of list lengths."""
    from pipeline import per_user_metrics

    rows = []
    for k in ks:
        metrics = per_user_metrics(scores, state, eval_df, k=k)
        rows.append(
            {
                "K": k,
                "precision": metrics.precision.mean(),
                "recall": metrics.recall.mean(),
                "ndcg": metrics.ndcg.mean(),
            }
        )

    return pd.DataFrame(rows)
"""
recommendation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PerUserMetrics:
    precision: np.ndarray
    recall: np.ndarray
    ndcg: np.ndarray


def topk(scores, seen_mask, k=10):
    """Return top-k unseen item indices for each user."""
    scores = np.asarray(scores)
    seen_mask = np.asarray(seen_mask, dtype=bool)
    if scores.ndim != 2 or seen_mask.shape != scores.shape:
        raise ValueError("scores and seen_mask must have the same 2D shape")
    if k < 1:
        raise ValueError("k must be positive")

    masked_scores = np.where(seen_mask, -np.inf, scores)
    return np.argsort(masked_scores, axis=1)[:, ::-1][:, :k]


def per_user_metrics(scores, state, eval_df, k=10):
    """Calculate ranking metrics for users with held-out positive items."""
    from metrics import ndcg_at_k, precision_at_k, recall_at_k

    truth = eval_df[eval_df.rating >= 4].groupby("user_id")["product_id"].apply(set).to_dict()
    recommendations = topk(scores, state["seen_mask"], k)
    precisions = []
    recalls = []
    ndcgs = []

    for user_id, relevant in truth.items():
        user_idx = state["u_idx"][user_id]
        relevant_items = {state["p_idx"][product_id] for product_id in relevant}
        recommended = recommendations[user_idx].tolist()
        precisions.append(precision_at_k(recommended, relevant_items, k=k))
        recalls.append(recall_at_k(recommended, relevant_items, k=k))
        ndcgs.append(ndcg_at_k(recommended, relevant_items, k=k))

    return PerUserMetrics(
        precision=np.asarray(precisions),
        recall=np.asarray(recalls),
        ndcg=np.asarray(ndcgs),
    )


def load_events(path):
    """Load event data from CSV."""
    return pd.read_csv(path)


def make_user_item_matrix(events, user_col="user_id", item_col="item_id"):
    """Group items by user while preserving a stable item list."""
    return (
        events[[user_col, item_col]]
        .drop_duplicates()
        .groupby(user_col, as_index=False)[item_col]
        .agg(list)
        .rename(columns={item_col: "items"})
    )


def rank_items(events, user_id, top_n=10, user_col="user_id", item_col="item_id", score_col="score"):
    """Return the top-scoring items for a single user."""
    subset = events.loc[events[user_col] == user_id, [item_col, score_col]]

    if subset.empty:
        return []

    return subset.nlargest(top_n, score_col)[item_col].tolist()


def build_recommendations(events, user_id=None, top_n=10, user_col="user_id", item_col="item_id", score_col="score"):
    """Build recommendation lists for one user or for all users."""
    if user_id is not None:
        return {
            user_id: rank_items(
                events,
                user_id,
                top_n=top_n,
                user_col=user_col,
                item_col=item_col,
                score_col=score_col,
            )
        }

    return {
        user: group.nlargest(top_n, score_col)[item_col].tolist()
        for user, group in events.groupby(user_col)
    }
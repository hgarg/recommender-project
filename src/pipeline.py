"""
recommendation pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def make_data(seed=42, n_users=5000, n_products=1988, n_interactions=67967):
    """Generate the synthetic dataset used by the implementation notebook."""
    from generate_synthetic_data import generate_interactions, generate_products

    rng = np.random.default_rng(seed)
    products = generate_products(n_products, rng)
    events = generate_interactions(
        n_users, n_products, n_interactions, 0.722, products, rng
    )
    average_rating = events.groupby("product_id")["rating"].mean()
    products["avg_rating"] = products["product_id"].map(average_rating).fillna(
        events["rating"].mean()
    ).round(2)
    return events, products


def _row_normalize(scores):
    minimum = scores.min(axis=1, keepdims=True)
    maximum = scores.max(axis=1, keepdims=True)
    span = np.where(maximum - minimum == 0, 1, maximum - minimum)
    return (scores - minimum) / span


def run_pipeline(events, products, k=12, n_full=8, n_min=3):
    """Build the score matrices and evaluation state used by the notebook."""
    events = events.drop_duplicates(
        subset=["user_id", "product_id", "timestamp"]
    ).copy()
    user_counts = events["user_id"].value_counts()
    product_counts = events["product_id"].value_counts()
    events = events[
        events["user_id"].isin(user_counts[user_counts >= n_min].index)
        & events["product_id"].isin(product_counts[product_counts >= n_min].index)
    ].sort_values("timestamp")

    split_train = int(len(events) * 0.70)
    split_validation = int(len(events) * 0.80)
    train = events.iloc[:split_train].copy()
    validation = events.iloc[split_train:split_validation].copy()
    test = events.iloc[split_validation:].copy()
    train_users = set(train["user_id"])
    train_products = set(train["product_id"])
    validation = validation[
        validation["user_id"].isin(train_users)
        & validation["product_id"].isin(train_products)
    ]
    test = test[
        test["user_id"].isin(train_users)
        & test["product_id"].isin(train_products)
    ]

    train_agg = train.groupby(["user_id", "product_id"], as_index=False)["rating"].mean()
    user_ids = sorted(train_agg["user_id"].unique())
    product_ids = sorted(train_agg["product_id"].unique())
    user_index = {user_id: index for index, user_id in enumerate(user_ids)}
    product_index = {product_id: index for index, product_id in enumerate(product_ids)}

    rows = train_agg["user_id"].map(user_index)
    columns = train_agg["product_id"].map(product_index)
    matrix = csr_matrix(
        (train_agg["rating"].values, (rows, columns)),
        shape=(len(user_ids), len(product_ids)),
    )
    latent_factors = min(k, min(matrix.shape) - 1)
    left, singular_values, right = svds(matrix.astype(float), k=latent_factors)
    svd_scores = left @ np.diag(singular_values) @ right
    svd_scores = _row_normalize(svd_scores)

    products_indexed = products.set_index("product_id")
    product_text = [
        f"{products_indexed.loc[product_id, 'category']} "
        f"{products_indexed.loc[product_id, 'price_bucket']}"
        for product_id in product_ids
    ]
    tfidf_matrix = TfidfVectorizer().fit_transform(product_text)
    item_similarity = cosine_similarity(tfidf_matrix)

    train_indexed = train_agg.copy()
    train_indexed["user_index"] = train_indexed["user_id"].map(user_index)
    train_indexed["product_index"] = train_indexed["product_id"].map(product_index)
    content_scores = np.zeros((len(user_ids), len(product_ids)))
    for user_index_value, group in train_indexed.groupby("user_index"):
        liked = group.loc[group["rating"] >= 4, "product_index"].values
        if len(liked):
            content_scores[user_index_value] = item_similarity[liked].mean(axis=0)
    content_scores = _row_normalize(content_scores)

    n_history = np.asarray(matrix.getnnz(axis=1)).ravel()
    collaborative_weight = np.clip(
        (n_history - n_min) / (n_full - n_min), 0, 1
    )
    fixed_scores = 0.5 * svd_scores + 0.5 * content_scores
    adaptive_scores = (
        collaborative_weight[:, None] * svd_scores
        + (1 - collaborative_weight[:, None]) * content_scores
    )
    popularity = train_agg["product_id"].value_counts()
    popularity_vector = np.array(
        [popularity.get(product_id, 0) for product_id in product_ids], dtype=float
    )
    popular_scores = np.tile(popularity_vector, (len(user_ids), 1))
    seen_mask = matrix.toarray() > 0

    return {
        "events_f": events,
        "train": train,
        "validation": validation,
        "test": test,
        "train_agg": train_agg,
        "user_ids": user_ids,
        "prod_ids": product_ids,
        "u_idx": user_index,
        "p_idx": product_index,
        "seen_mask": seen_mask,
        "n_hist": n_history,
        "w_cf": collaborative_weight,
        "scores": {
            "Hybrid (adaptive)": adaptive_scores,
            "SVD only": svd_scores,
            "TF-IDF only": content_scores,
            "Fixed 50/50 hybrid": fixed_scores,
            "Most popular": popular_scores,
        },
    }


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

    return pd.DataFrame(
        {
            "precision": precisions,
            "recall": recalls,
            "ndcg": ndcgs,
            "n_hist": [state["n_hist"][state["u_idx"][user_id]] for user_id in truth],
        }
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
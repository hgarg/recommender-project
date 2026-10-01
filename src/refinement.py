"""Helpers for trying small changes in the recommendation pipeline.

This file is built for quick experiments. The idea is simple: keep the main
pipeline in pipeline.py as the reference version, then rebuild one part at a
 time here to compare different settings.
"""

import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def row_normalize(scores):
    """Scale each row to the range [0, 1]."""
    lo = scores.min(axis=1, keepdims=True)
    hi = scores.max(axis=1, keepdims=True)
    span = np.where(hi - lo == 0, 1, hi - lo)
    return (scores - lo) / span


def train_matrix(state, signal="rating"):
    """Build the user-by-item training matrix.

    signal = "rating": use the raw 1-5 rating
    signal = "binary": treat every interaction as 1
    signal = "centered": subtract each user's average rating
    """
    agg = state["train_agg"]
    rows = agg["user_id"].map(state["u_idx"]).values
    cols = agg["product_id"].map(state["p_idx"]).values

    if signal == "rating":
        values = agg["rating"].values.astype(float)
    elif signal == "binary":
        values = np.ones(len(agg), dtype=float)
    elif signal == "centered":
        user_mean = agg.groupby("user_id")["rating"].transform("mean")
        # keep a tiny offset so values at the mean are still stored as observed
        values = (agg["rating"] - user_mean).values + 0.01
    else:
        raise ValueError(f"unknown signal: {signal}")

    shape = (len(state["user_ids"]), len(state["prod_ids"]))
    return csr_matrix((values, (rows, cols)), shape=shape)


def svd_scores(matrix, k=12, seed=0):
    """Make an SVD-based score matrix and normalize the rows."""
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return np.zeros(matrix.shape, dtype=float)

    n_factors = min(k, min(matrix.shape) - 1)
    if n_factors <= 0:
        return np.zeros(matrix.shape, dtype=float)

    v0 = np.random.default_rng(seed).random(min(matrix.shape))
    u, s, vt = svds(matrix.astype(float), k=n_factors, v0=v0)
    return row_normalize(u @ np.diag(s) @ vt)


def item_text(products, prod_ids, fields, avg_rating=None):
    """Build the text for each product so TF-IDF can compare product similarity."""
    if avg_rating is None:
        avg_rating = {}

    product_table = products.set_index("product_id")
    texts = []

    for product_id in prod_ids:
        parts = []
        for field in fields:
            if field == "rating_tier":
                rating = avg_rating.get(product_id, np.nan)
                if np.isnan(rating):
                    tier = "unrated"
                elif rating >= 4:
                    tier = "rating_high"
                elif rating >= 3:
                    tier = "rating_mid"
                else:
                    tier = "rating_low"
                parts.append(tier)
            else:
                parts.append(str(product_table.loc[product_id, field]))
        texts.append(" ".join(parts))

    return texts


def content_scores(state, texts, profile="liked_mean"):
    """Build item-content scores from cosine similarity between product texts."""
    tfidf = TfidfVectorizer().fit_transform(texts)
    similarity = cosine_similarity(tfidf)
    agg = state["train_agg"]

    out = np.zeros((len(state["user_ids"]), len(state["prod_ids"])))
    ui = agg["user_id"].map(state["u_idx"]).values
    pi = agg["product_id"].map(state["p_idx"]).values
    ratings = agg["rating"].values
    df = pd.DataFrame({"u": ui, "p": pi, "r": ratings})

    for user_index, group in df.groupby("u"):
        if profile == "liked_mean":
            liked = group.loc[group.r >= 4, "p"].values
            if len(liked):
                out[user_index] = similarity[liked].mean(axis=0)
        elif profile == "rating_weighted":
            weights = (group.r - 3).values.astype(float)
            if np.abs(weights).sum() > 0:
                out[user_index] = (
                    weights[:, None] * similarity[group.p.values]
                ).sum(axis=0) / np.abs(weights).sum()
        else:
            raise ValueError(profile)

    return row_normalize(out), len(set(texts))


def popularity_vector(state):
    """Return the popularity count for every product in the training set."""
    counts = state["train_agg"]["product_id"].value_counts()
    return np.array([counts.get(product_id, 0) for product_id in state["prod_ids"]], dtype=float)


def adaptive_weight(n_hist, n_min=3, n_full=8, w_floor=0.0):
    """Weight each user toward collaborative filtering based on history length."""
    n_hist = np.asarray(n_hist, dtype=float)

    if n_full <= n_min:
        raise ValueError("n_full must be greater than n_min")

    ramp = np.clip((n_hist - n_min) / (n_full - n_min), 0, 1)
    return w_floor + (1 - w_floor) * ramp


def blend(a, b, w):
    """Blend two score matrices using a scalar or one weight per user."""
    w = np.asarray(w, dtype=float)
    if w.ndim == 1:
        w = w[:, None]
    return w * a + (1 - w) * b


def popularity_rerank(scores, pop, lam):
    """Subtract a popularity penalty before ranking items."""
    pop_rank = pd.Series(pop).rank(pct=True).values
    return scores - lam * pop_rank[None, :]


def fast_topk(scores, seen_mask, k=10):
    """Fast top-k selection that behaves like the original helper.

    This version uses argpartition instead of sorting every row.
    """
    scores = np.asarray(scores, dtype=float)
    seen_mask = np.asarray(seen_mask, dtype=bool)

    if scores.ndim != 2 or seen_mask.shape != scores.shape:
        raise ValueError("scores and seen_mask must have the same 2D shape")
    if k < 1:
        raise ValueError("k must be positive")
    if scores.shape[1] == 0:
        return np.empty((scores.shape[0], 0), dtype=int)

    k = min(int(k), scores.shape[1])
    masked = np.where(seen_mask, -np.inf, scores)
    part = np.argpartition(-masked, kth=k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(masked, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def time_it(fn, repeats=3):
    """Return the median time for a small timing experiment."""
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return float(np.median(times))

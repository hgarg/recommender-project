"""Final recommender used by the Streamlit app.

This module collects the final model from the refinement notebook in one
place, so the app does not need to re-run the notebook. Settings are the ones picked on the
validation set in advanced_analysis_refinement_final.ipynb:

- SVD on raw ratings, 12 factors
- TF-IDF on product category only (price bucket was dropped in R3)
- alpha = 0.1 (weight on SVD), picked for seed 42
- popularity penalty lambda = 0.0025 (R5)

The other models (SVD only, fixed 50/50, most popular) are kept so the app
can show them side by side.
"""

import numpy as np
import pandas as pd

from pipeline import make_data, run_pipeline, per_user_metrics
from refinement import (
    train_matrix,
    svd_scores,
    item_text,
    content_scores,
    popularity_vector,
    blend,
    popularity_rerank,
    fast_topk,
)

FINAL_ALPHA = 0.1
FINAL_LAMBDA = 0.0025
SVD_FACTORS = 12

MODEL_NAMES = ["Final hybrid", "SVD only", "Fixed 50/50 hybrid", "Most popular"]


def build_models(seed=42, alpha=FINAL_ALPHA, lam=FINAL_LAMBDA, k=SVD_FACTORS):
    """Generate the data, train all models once, and return the objects the app uses."""
    events, products = make_data(seed=seed)
    state = run_pipeline(events, products, k=k, n_full=8)

    svd = svd_scores(train_matrix(state, "rating"), k=k)
    pop = popularity_vector(state)

    # starting model: category + price, 50/50
    text_cp = item_text(products, state["prod_ids"], ["category", "price_bucket"])
    content_cp, _ = content_scores(state, text_cp)

    # final model: category only, then the small popularity penalty
    text_cat = item_text(products, state["prod_ids"], ["category"])
    content_cat, _ = content_scores(state, text_cat)
    final = popularity_rerank(blend(svd, content_cat, alpha), pop, lam)

    scores = {
        "Final hybrid": final,
        "SVD only": svd,
        "Fixed 50/50 hybrid": blend(svd, content_cp, 0.5),
        "Most popular": np.tile(pop, (len(state["user_ids"]), 1)),
    }
    return state, products, scores, pop


def recommend(scores, state, user_id, k=10):
    """Top-k unseen products for one customer, best first."""
    u = state["u_idx"][user_id]
    top = fast_topk(scores[u : u + 1], state["seen_mask"][u : u + 1], k=k)[0]
    return [state["prod_ids"][i] for i in top], scores[u, top]


def user_history(state, products, user_id):
    """Training-period ratings for one customer, with product details."""
    hist = state["train_agg"][state["train_agg"].user_id == user_id]
    hist = hist.merge(products, on="product_id", how="left")
    return hist.sort_values("rating", ascending=False).reset_index(drop=True)


def held_out_liked(state, user_id, split="test"):
    """Products the customer rated 4+ in the held-out split."""
    df = state[split]
    return set(df[(df.user_id == user_id) & (df.rating >= 4)].product_id)


SVD_REASON = "Rated highly by customers with similar rating patterns"


def _liked_text(n):
    return f"{n} of your 4-5★ products"


def explain(model_name, state, products, user_id, rec_ids, pop_rank):
    """One short reason per recommended product.

    The reasons follow how each model scores items. The content part of the
    final hybrid only uses category, so its score is driven by how many of the
    customer's 4+ rated products share the category, and SVD then orders the
    products within that category. The 50/50 hybrid also uses price bucket.
    """
    liked = user_history(state, products, user_id)
    liked = liked[liked.rating >= 4]
    info = products.set_index("product_id")

    reasons = []
    for p in rec_ids:
        cat, price = info.loc[p, "category"], info.loc[p, "price_bucket"]
        n_cat = int((liked.category == cat).sum())
        n_both = int(((liked.category == cat) & (liked.price_bucket == price)).sum())
        n_price = int((liked.price_bucket == price).sum())

        if model_name == "Most popular":
            reason = f"One of the most-rated products overall (popularity rank {pop_rank[p]})"
        elif model_name == "SVD only":
            reason = SVD_REASON
        elif model_name == "Fixed 50/50 hybrid":
            if n_both:
                reason = f"Same category and price range ({cat}, {price}) as {_liked_text(n_both)}"
            elif n_cat:
                reason = f"Same category ({cat}) as {_liked_text(n_cat)}"
            elif n_price:
                reason = f"Same price range ({price}) as {_liked_text(n_price)}"
            else:
                reason = SVD_REASON
        else:  # final hybrid
            if n_cat:
                reason = f"Same category ({cat}) as {_liked_text(n_cat)}"
            else:
                reason = SVD_REASON
        reasons.append(reason)
    return reasons


def evaluate_models(state, scores, split="test", k=10):
    """Mean precision/recall/NDCG and catalog coverage for each model."""
    rows = []
    n_items = len(state["prod_ids"])
    for name, s in scores.items():
        m = per_user_metrics(s, state, state[split], k=k)
        recs = fast_topk(s, state["seen_mask"], k=k)
        rows.append(
            {
                "model": name,
                f"precision@{k}": m.precision.mean(),
                f"recall@{k}": m.recall.mean(),
                f"NDCG@{k}": m.ndcg.mean(),
                "catalog coverage": len(np.unique(recs)) / n_items,
            }
        )
    return pd.DataFrame(rows).set_index("model")


def ndcg_by_tier(state, scores, split="test", k=10):
    """NDCG@k split by how many training interactions the customer has."""
    bins = [0, 2, 4, 9, 19, np.inf]
    labels = ["1-2", "3-4", "5-9", "10-19", "20+"]
    out = {}
    for name, s in scores.items():
        m = per_user_metrics(s, state, state[split], k=k)
        tier = pd.cut(m.n_hist, bins, labels=labels)
        out[name] = m.groupby(tier, observed=True).ndcg.mean()
    return pd.DataFrame(out)

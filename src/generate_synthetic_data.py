"""
Generate synthetic e-commerce data for the recommender project.

This script creates a product catalog and a set of user-product interactions
with a similar overall shape to the Amazon Electronics review data: sparse
interactions, a strong positive-rating bias, and a long-tail popularity pattern.
It does not use any real user data.

Usage:
    python generate_synthetic_data.py --out-dir ../data/raw --seed 42
"""

import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

CATEGORIES = ["Headphones", "Cameras", "Chargers", "Keyboards", "Speakers",
              "Monitors", "Cables", "Laptops", "Tablets", "SmartHome"]


def generate_products(n_products, rng):
    product_ids = [f"P{i:05d}" for i in range(1, n_products + 1)]
    categories = rng.choice(CATEGORIES, size=n_products)
    # The price distribution is skewed so most items are reasonably cheap,
    # while a smaller tail reaches much higher prices.
    prices = np.round(rng.lognormal(3.2, 0.9, size=n_products), 2)
    prices = np.clip(prices, 4.99, 1499.99)

    df = pd.DataFrame({"product_id": product_ids, "category": categories, "price": prices})
    df["price_bucket"] = pd.cut(df["price"], bins=[0, 20, 50, 120, 300, 1e6],
                                 labels=["budget", "low", "mid", "high", "premium"])
    return df


def generate_interactions(n_users, n_products, n_interactions, pos_target, products_df, rng):
    # A few users and products get more attention than the rest, which creates
    # the usual long-tail pattern in recommendation datasets.
    user_ids = np.array([f"U{i:05d}" for i in range(1, n_users + 1)])
    product_ids = np.array([f"P{i:05d}" for i in range(1, n_products + 1)])

    user_w = 1.0 / np.arange(1, n_users + 1) ** 0.9
    user_w /= user_w.sum()
    rng.shuffle(user_w)

    # This was tuned down from an earlier version so the popularity spike was
    # less extreme while still keeping the tail realistic.
    item_w = 1.0 / np.arange(1, n_products + 1) ** 0.78
    item_w /= item_w.sum()
    rng.shuffle(item_w)

    # The first draft sampled users and products independently, which meant the
    # data had no real personalization pattern. Each user is now assigned a
    # preferred category, so there is an actual signal for collaborative and
    # content-based models to learn from.
    user_pref_category = rng.choice(CATEGORIES, size=n_users)
    pref_lookup = dict(zip(user_ids, user_pref_category))
    cat_lookup = products_df.set_index("product_id")["category"].to_dict()
    products_by_cat = {c: products_df.loc[products_df.category == c, "product_id"].values
                        for c in CATEGORIES}
    item_w_lookup = dict(zip(product_ids, item_w))

    AFFINITY_STRENGTH = 0.55  # Share of interactions drawn from the user's preferred category.

    sampled_users = rng.choice(user_ids, size=n_interactions, p=user_w)
    sampled_products = np.empty(n_interactions, dtype=object)
    in_affinity = rng.random(n_interactions) < AFFINITY_STRENGTH

    for i in range(n_interactions):
        u = sampled_users[i]
        if in_affinity[i]:
            cat = pref_lookup[u]
            cat_products = products_by_cat[cat]
            cat_w = np.array([item_w_lookup[p] for p in cat_products])
            cat_w = cat_w / cat_w.sum()
            sampled_products[i] = rng.choice(cat_products, p=cat_w)
        else:
            sampled_products[i] = rng.choice(product_ids, p=item_w)

    # The rating distribution is shaped so the overall positive rate stays near
    # the target, with a slight bump when the chosen item matches the user's
    # preferred category.
    base_probs = np.array([0.08, 0.12, 0.27, 0.28, 0.25])
    boosted_probs = np.array([0.02, 0.03, 0.10, 0.30, 0.55])

    ratings = np.zeros(n_interactions, dtype=int)
    matches_pref = np.array([cat_lookup[p] == pref_lookup[u]
                              for u, p in zip(sampled_users, sampled_products)])
    n_match = matches_pref.sum()
    ratings[matches_pref] = rng.choice([1, 2, 3, 4, 5], size=n_match, p=boosted_probs)
    ratings[~matches_pref] = rng.choice([1, 2, 3, 4, 5], size=n_interactions - n_match, p=base_probs)

    # Timestamps are spread over an 18-month window so the data can later be
    # split into train and test by time rather than by random row selection.
    start = datetime(2024, 3, 1)
    day_offsets = rng.integers(0, 540, size=n_interactions)
    timestamps = [start + timedelta(days=int(d), seconds=int(rng.integers(0, 86400))) for d in day_offsets]

    events = pd.DataFrame({
        "user_id": sampled_users,
        "product_id": sampled_products,
        "rating": ratings,
        "timestamp": timestamps,
    })
    return events.sort_values("timestamp").reset_index(drop=True)


def summarize(events, products):
    n_int = len(events)
    n_users = events["user_id"].nunique()
    n_prod = events["product_id"].nunique()
    sparsity = 1 - (n_int / (n_users * n_prod))
    pos_rate = (events["rating"] >= 4).mean()
    top5_cut = max(1, int(0.05 * n_prod))
    top5_share = events["product_id"].value_counts().head(top5_cut).sum() / n_int

    print(f"interactions: {n_int}")
    print(f"distinct users: {n_users}")
    print(f"distinct products: {n_prod}")
    print(f"sparsity: {sparsity:.4%}")
    print(f"rating >=4 share: {pos_rate:.4%}")
    print(f"top 5% item share: {top5_share:.4%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="../data/raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-users", type=int, default=5000)
    parser.add_argument("--n-products", type=int, default=1988)
    parser.add_argument("--n-interactions", type=int, default=67967)
    parser.add_argument("--pos-target", type=float, default=0.722)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    products = generate_products(args.n_products, rng)
    events = generate_interactions(args.n_users, args.n_products, args.n_interactions,
                                    args.pos_target, products, rng)

    avg = events.groupby("product_id")["rating"].mean()
    products["avg_rating"] = products["product_id"].map(avg).fillna(events["rating"].mean()).round(2)

    os.makedirs(args.out_dir, exist_ok=True)
    events.to_csv(os.path.join(args.out_dir, "events.csv"), index=False)
    products.to_csv(os.path.join(args.out_dir, "products.csv"), index=False)

    summarize(events, products)
    print("done, wrote to", args.out_dir)

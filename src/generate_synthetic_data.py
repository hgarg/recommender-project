"""Quick script to make a fake e-commerce dataset.

Run:
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
    # keep prices mostly low, with a few expensive items
    prices = np.round(rng.lognormal(3.2, 0.9, size=n_products), 2)
    prices = np.clip(prices, 4.99, 1499.99)

    df = pd.DataFrame({"product_id": product_ids, "category": categories, "price": prices})
    df["price_bucket"] = pd.cut(df["price"], bins=[0, 20, 50, 120, 300, 1e6],
                                 labels=["budget", "low", "mid", "high", "premium"])
    return df


def generate_interactions(n_users, n_products, n_interactions, pos_target, rng):
    # skewed user/product popularity so it looks more like real data
    user_ids = np.array([f"U{i:05d}" for i in range(1, n_users + 1)])
    product_ids = np.array([f"P{i:05d}" for i in range(1, n_products + 1)])

    user_w = 1.0 / np.arange(1, n_users + 1) ** 0.9
    user_w /= user_w.sum()
    rng.shuffle(user_w)

    # tuned to make the product popularity less uniform
    item_w = 1.0 / np.arange(1, n_products + 1) ** 0.78
    item_w /= item_w.sum()
    rng.shuffle(item_w)

    users = rng.choice(user_ids, size=n_interactions, p=user_w)
    items = rng.choice(product_ids, size=n_interactions, p=item_w)

    # ratings skewed toward 4s and 5s
    star_probs = np.array([0.04, 0.06, 0.14, 0.32, 0.44])
    cur_pos = star_probs[3] + star_probs[4]
    if pos_target != cur_pos:
        diff = pos_target - cur_pos
        star_probs[3] += diff * 0.55
        star_probs[4] += diff * 0.45
        star_probs[:3] *= (1 - star_probs[3] - star_probs[4]) / star_probs[:3].sum()
    ratings = rng.choice([1, 2, 3, 4, 5], size=n_interactions, p=star_probs)

    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(days=int(d)) for d in rng.integers(0, 600, size=n_interactions)]

    df = pd.DataFrame({"user_id": users, "product_id": items, "rating": ratings,
                        "timestamp": timestamps})

    # keep duplicates on purpose; they get cleaned in prep later
    return df


def summarize(events, products):
    n_users = events["user_id"].nunique()
    n_products = events["product_id"].nunique()
    n = len(events)
    sparsity = 1 - n / (n_users * n_products)
    dupes = events.duplicated(subset=["user_id", "product_id", "timestamp"]).sum()
    pos = (events["rating"] >= 4).mean()
    counts = events["product_id"].value_counts()
    top5 = counts.iloc[:max(1, int(len(counts) * 0.05))].sum() / n

    print("interactions:", n)
    print("users:", n_users, " products:", n_products)
    print(f"sparsity: {sparsity:.4%}")
    print("dupe rows:", dupes)
    print(f"pct rating>=4: {pos:.4%}")
    print(f"top5% items share: {top5:.4%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="../data/raw")
    parser.add_argument("--n-users", type=int, default=5000)
    parser.add_argument("--n-products", type=int, default=1988)
    parser.add_argument("--n-interactions", type=int, default=67967)
    parser.add_argument("--positivity-target", type=float, default=0.722)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    products = generate_products(args.n_products, rng)
    events = generate_interactions(args.n_users, args.n_products, args.n_interactions,
                                    args.positivity_target, rng)

    # fill in average rating per product
    avg = events.groupby("product_id")["rating"].mean()
    products["avg_rating"] = products["product_id"].map(avg).fillna(events["rating"].mean()).round(2)

    os.makedirs(args.out_dir, exist_ok=True)
    events.to_csv(os.path.join(args.out_dir, "events.csv"), index=False)
    products.to_csv(os.path.join(args.out_dir, "products.csv"), index=False)

    summarize(events, products)
    print("done, wrote to", args.out_dir)

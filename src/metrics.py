import numpy as np


def precision_at_k(recommended, relevant, k=10):
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / k


def recall_at_k(recommended, relevant, k=10):
    if not relevant:
        return 0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k=10):
    rec_k = recommended[:k]
    hits = [1 if item in relevant else 0 for item in rec_k]
    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0

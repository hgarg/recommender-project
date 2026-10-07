# tests for the final model wrapper used by the app.
# building the models takes about 15 s, so it is done once for the whole file.
# the last test is a regression check: if the final NDCG changes, something in
# the pipeline has changed and the reported numbers would be out of date.

import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from recommender import build_models, recommend, evaluate_models, MODEL_NAMES


@pytest.fixture(scope="module")
def built():
    return build_models(seed=42)


def test_all_models_built(built):
    state, products, scores, pop = built
    assert set(scores) == set(MODEL_NAMES)
    for s in scores.values():
        assert s.shape == (len(state["user_ids"]), len(state["prod_ids"]))


def test_recommend_skips_already_rated(built):
    state, products, scores, pop = built
    user = state["user_ids"][0]
    rated = set(state["train_agg"][state["train_agg"].user_id == user].product_id)
    rec_ids, _ = recommend(scores["Final hybrid"], state, user, k=10)
    assert len(rec_ids) == 10
    assert len(set(rec_ids)) == 10  # no duplicates
    assert not rated & set(rec_ids)


def test_scores_come_back_sorted(built):
    state, products, scores, pop = built
    _, rec_scores = recommend(scores["Final hybrid"], state, state["user_ids"][5], k=15)
    assert all(rec_scores[i] >= rec_scores[i + 1] for i in range(len(rec_scores) - 1))


def test_final_numbers_match_report(built):
    state, products, scores, pop = built
    res = evaluate_models(state, scores)
    assert res.loc["Final hybrid", "NDCG@10"] == pytest.approx(0.1193, abs=5e-4)
    assert res.loc["SVD only", "NDCG@10"] == pytest.approx(0.1000, abs=5e-4)
    # final model should beat SVD alone
    assert res.loc["Final hybrid", "NDCG@10"] > res.loc["SVD only", "NDCG@10"]

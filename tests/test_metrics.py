import sys
sys.path.append("../src")
from metrics import precision_at_k, recall_at_k, ndcg_at_k


def test_precision_simple():
    recommended = ["a", "b", "c", "d", "e"]
    relevant = {"a", "c"}
    assert precision_at_k(recommended, relevant, k=5) == 2 / 5


def test_recall_simple():
    recommended = ["a", "b", "c", "d", "e"]
    relevant = {"a", "c", "z"}
    assert recall_at_k(recommended, relevant, k=5) == 2 / 3


def test_recall_no_relevant_items():
    assert recall_at_k(["a", "b"], set(), k=5) == 0


def test_ndcg_perfect_ranking():
    recommended = ["a", "b", "c"]
    relevant = {"a", "b"}
    assert abs(ndcg_at_k(recommended, relevant, k=3) - 1.0) < 1e-9


def test_ndcg_worse_when_hit_is_lower():
    recommended_good = ["a", "b", "c"]
    recommended_bad = ["c", "b", "a"]
    relevant = {"a"}
    good = ndcg_at_k(recommended_good, relevant, k=3)
    bad = ndcg_at_k(recommended_bad, relevant, k=3)
    assert good > bad


if __name__ == "__main__":
    test_precision_simple()
    test_recall_simple()
    test_recall_no_relevant_items()
    test_ndcg_perfect_ranking()
    test_ndcg_worse_when_hit_is_lower()
    print("all metric tests passed")

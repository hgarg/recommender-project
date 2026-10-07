# tests for the refinement helpers, mainly that the weight floor behaves
# correctly at the boundaries and that the faster top-k returns the same lists
# as the original version.

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from pipeline import topk
from refinement import adaptive_weight, blend, fast_topk, popularity_rerank


def test_floor_zero_matches_original_rule():
    n = np.array([0, 3, 5, 8, 20])
    w = adaptive_weight(n, n_min=3, n_full=8, w_floor=0.0)
    assert w[0] == 0 and w[1] == 0
    assert np.isclose(w[2], 0.4)
    assert w[3] == 1 and w[4] == 1


def test_floor_lifts_cold_users_only():
    n = np.array([1, 3, 8, 30])
    w = adaptive_weight(n, n_min=3, n_full=8, w_floor=0.5)
    assert w[0] == 0.5 and w[1] == 0.5  # low-history users now keep half of the CF weight
    assert w[2] == 1 and w[3] == 1      # warm users unchanged


def test_blend_scalar_and_per_user():
    a = np.ones((2, 3))
    b = np.zeros((2, 3))
    assert np.allclose(blend(a, b, 0.3), 0.3)
    out = blend(a, b, np.array([0.2, 0.9]))
    assert np.allclose(out[0], 0.2) and np.allclose(out[1], 0.9)


def test_rerank_zero_lambda_is_noop():
    s = np.random.default_rng(0).random((4, 6))
    pop = np.arange(6, dtype=float)
    assert np.allclose(popularity_rerank(s, pop, 0.0), s)


def test_fast_topk_matches_argsort():
    rng = np.random.default_rng(1)
    scores = rng.random((50, 200))
    seen = rng.random((50, 200)) < 0.05
    assert (fast_topk(scores, seen, 10) == topk(scores, seen, 10)).all()


def test_fast_topk_never_returns_seen():
    rng = np.random.default_rng(2)
    scores = rng.random((20, 40))
    seen = rng.random((20, 40)) < 0.3
    rec = fast_topk(scores, seen, 5)
    assert not np.take_along_axis(seen, rec, axis=1).any()


def test_adaptive_weight_rejects_invalid_window():
    with pytest.raises(ValueError, match="n_full"):
        adaptive_weight(np.array([5]), n_min=5, n_full=5, w_floor=0.0)


def test_fast_topk_allows_k_larger_than_columns():
    scores = np.random.default_rng(7).random((2, 3))
    seen = np.zeros_like(scores, dtype=bool)
    rec = fast_topk(scores, seen, 10)
    assert rec.shape == (2, 3)
    assert np.array_equal(np.sort(rec[0]), np.array([0, 1, 2]))

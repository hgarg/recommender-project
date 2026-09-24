import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation import bootstrap_ci


def test_bootstrap_ci_brackets_the_mean():
    vals = np.random.default_rng(0).random(500)
    m, lo, hi = bootstrap_ci(vals)
    assert lo < m < hi


def test_bootstrap_ci_collapses_for_constant_values():
    m, lo, hi = bootstrap_ci(np.full(100, 0.25))
    assert m == lo == hi == 0.25

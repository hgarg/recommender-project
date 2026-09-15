# quick tests for the cold-start weighting fn before wiring it into anything
# real - just want to know it behaves at the edges before trusting it in the
# full pipeline

import sys
sys.path.append("../src")
from blending import blend_weight


def test_below_min_is_zero():
    assert blend_weight(0)[0] == 0
    assert blend_weight(3)[0] == 0  # right at n_min, should still floor to 0


def test_above_full_is_one():
    assert blend_weight(20)[0] == 1
    assert blend_weight(50)[0] == 1  # way past n_full, still capped at 1


def test_midpoint():
    # halfway between n_min=3 and n_full=20 should land close to 0.5
    mid = (3 + 20) / 2
    cf, _ = blend_weight(mid)
    assert abs(cf - 0.5) < 0.01


def test_monotonic():
    # weight should never decrease as a user gets more history
    vals = [blend_weight(n)[0] for n in range(0, 30)]
    assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


if __name__ == "__main__":
    test_below_min_is_zero()
    test_above_full_is_one()
    test_midpoint()
    test_monotonic()
    print("all blending tests passed")

# linear blending rule: more past interactions give more weight to collaborative
# filtering. at or below n_min the score is content-only, at or above n_full it is CF-only.


def blend_weight(n_u, n_min=3, n_full=20):
    """Return (w_cf, w_content) for a user with n_u prior interactions."""
    if n_full <= n_min:
        raise ValueError("n_full has to be bigger than n_min")

    if n_u <= n_min:
        w_cf = 0.0
    elif n_u >= n_full:
        w_cf = 1.0
    else:
        w_cf = (n_u - n_min) / (n_full - n_min)

    return w_cf, 1.0 - w_cf


if __name__ == "__main__":
    # sanity check of the weights at selected history lengths
    for n in [0, 1, 3, 5, 10, 12, 15, 20, 25, 40]:
        cf, content = blend_weight(n)
        print(f"n_u={n}  w_cf={cf:.3f}  w_content={content:.3f}")

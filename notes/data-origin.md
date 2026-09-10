# data-origin

This is a synthetic dataset, not a real Amazon data dump. There are no real user records or product rows in here. The generator in src/generate_synthetic_data.py is just trying to match the general shape of the Amazon Electronics reviews dataset reported in the paper below:

McAuley, J., Targett, C., Shi, Q., & van den Hengel, A. (2015). Image-based recommendations on styles and substitutes. Proceedings of the 38th International ACM SIGIR Conference, 43-52.

The goal was to get something roughly similar in a few key ways: very sparse data, mostly 4-5 star ratings, and a long-tail popularity pattern. The category names, price ranges, and user/product counts were all made up for this project, not taken from any source.

How it works:
- products get a category and a log-normal price
- interactions are sampled with Zipf-style user activity and item popularity so a few products show up a lot and most are rare
- ratings are drawn from a tuned distribution to get around 72% positive ratings
- duplicate rows happen naturally because of sampling with replacement; they are left in on purpose because the preprocessing step is meant to clean them up

I ran it once with seed=42 using the default setup: 5000 users, 1988 products, 67967 interactions, positivity target 0.722.

Results:
- interactions: 67967
- distinct users: 4862
- distinct products: 1988
- sparsity: 99.30% vs target 99.32%
- rating >= 4: 72.1% vs target 72.2%
- top 5% item share: 42.4% vs target 47.2%

This is close overall. The main miss is how concentrated the top items are, but it is still good enough for a first pass. I would probably rerun it once the final dataset pipeline is locked in, instead of relying too much on this one random seed.

One thing still to check later is the raw-vs-filtered comparison from the design notes. That is separate from this generator check. This file only checks whether the generator hits its own target, not whether the later filtering step changes the distribution in a bad way.

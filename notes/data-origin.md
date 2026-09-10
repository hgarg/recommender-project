# data-origin

This dataset is synthetic, not a real Amazon dump. There are no actual user records or product rows in it. I made the generator in src/generate_synthetic_data.py to match the general shape of the Amazon Electronics reviews dataset described in the paper below:

McAuley, J., Targett, C., Shi, Q., & van den Hengel, A. (2015). Image-based recommendations on styles and substitutes. Proceedings of the 38th International ACM SIGIR Conference, 43-52.

I wanted something with a similar level of sparsity, a strong skew toward 4-5 star ratings, and a long-tail popularity pattern. The category names, price ranges, and user/product counts were all created for this project rather than taken from a real source.

How it works:
- products are assigned a category and a log-normal price
- interactions are sampled using Zipf-style user activity and item popularity so a few products appear a lot and most are quite rare
- ratings are drawn from a tuned distribution to get around 72% positive ratings
- duplicate rows happen naturally because sampling is done with replacement; I left them in on purpose because the preprocessing step is meant to clean them up

I ran it once with seed=42 using the default setup: 5000 users, 1988 products, 67967 interactions, and a positivity target of 0.722.

Results:
- interactions: 67967
- distinct users: 4862
- distinct products: 1988
- sparsity: 99.30% vs target 99.32%
- rating >= 4: 72.1% vs target 72.2%
- top 5% item share: 42.4% vs target 47.2%

Overall this is fairly close. The biggest mismatch is that the top products are a bit less concentrated than the target, but it is still good enough for a first pass. I would likely rerun it once the final dataset pipeline is fixed, instead of relying too much on one random seed.

One thing I still want to check later is the raw-vs-filtered comparison noted in the design writeup. That is separate from this generator check. This file only checks whether the generator is close to the target distribution, not whether the later filtering step changes the data in a problematic way.

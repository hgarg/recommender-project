# project notes

Working notes behind the planning phase report + this week's refinement assignment.
Not trying to make this read like a polished paper, just want my reasoning written down
somewhere I can actually find it again.

## scope / objectives (what changed)

Original pitch was pretty generic - hybrid recommender, SVD + TF-IDF, done on whatever
ratings dataset. After feedback I tightened it up: it's an e-commerce recommender now,
dataset is shaped like Amazon Electronics reviews, and the actual point of interest is
the blending. Instead of a fixed weight between the two models (like a flat 60/40) I'm
letting the weight move depending on how much history a user or item actually has. New
item, barely any interactions -> lean TF-IDF. Item's been around a while, has real
interaction data -> lean SVD. That's basically how you're supposed to attack cold start
per the hybrid recommender literature (Burke's 2002 taxonomy calls this a weighted
hybrid w/ an adaptive piece) instead of just picking a static ratio and hoping it works
across the board (Adomavicius & Tuzhilin, 2005 make this point too - static hybrids
mostly hide the cold start problem, they don't fix it).

## needs analysis / why this matters

Recommendation quality is one of the more directly measurable things in e-commerce -
even small relevance gains move conversion. That's the business case, not super deep but
it's real.

Bigger issue was feasibility. Full dataset is too big for my laptop or a normal colab
session, so I did stratified sampling instead - bucket users/items by how active they
are, sample proportionally out of each bucket rather than just cutting randomly. Raw
data was ~68k interactions / 5k users / ~2k products, sparsity over 99%. After filtering
out low-activity users and items it's down to ~56.4k interactions, ~1,750 users, ~756
products, 95.75% sparsity - better, still sparse, which is normal for this kind of data
(Koren et al., 2009).

risks I'm tracking:
- popularity bias - top 5% of products = ~47% of all interactions. left alone the model
  will just keep recommending the same handful of things. plan is diversity-aware
  re-ranking later, accuracy by itself isn't a good enough goal (McNee et al., 2006 make
  this argument well)
- no real ground truth for "would have bought" - going to lean on proxy ranking metrics
  instead (precision/recall/NDCG, catalog coverage) rather than pretend I'm predicting
  intent directly

## methodology (high level)

- SVD (matrix factorization) for the collaborative side - standard approach, goes back
  to the Netflix Prize era stuff (Koren et al., 2009)
- TF-IDF on product metadata (category, price bucket, avg rating) for content side
  (Ramos, 2003)
- blend the two with the dynamic weight described above
- tools: python, pandas, scikit-learn + surprise for SVD, sklearn's TF-IDF vectorizer,
  eventually streamlit for the demo
- dataset shaped like the He & McAuley (2016) Amazon dataset work
- preprocessing = the stratified sampling above + building the TF-IDF content field +
  correcting for a pretty heavy ratings positivity bias (72% of ratings are 4-5 stars) by
  weighting interactions instead of trusting raw stars

## credibility / ethics

Data's public, not real customer data, but treating it like it matters anyway - checked
the license before using it, keeping working copies in the project folder only (not some
random shared drive), documenting where it came from so that doesn't get lost later.

If this ever got extended to actual customer interaction data it'd need to be handled
under something like GDPR/CCPA - collection has to be justified, can't just retain
indefinitely, and there'd need to be a real way for someone to ask their data be removed
(Voigt & dem Bussche, 2017).

Also thinking about fairness a little here, not just privacy - a model that just
reinforces whatever's already popular effectively locks out newer/smaller sellers from
ever getting seen, which is part of why the diversity re-ranking thing isn't optional to
me, it's part of the actual design (McNee et al., 2006 again).

## refs (informal, not the full APA reference list - that's in the actual assignment doc)

Adomavicius & Tuzhilin (2005), Burke (2002), He & McAuley (2016), Koren, Bell & Volinsky
(2009), McNee, Riedl & Konstan (2006), Ramos (2003), Voigt & dem Bussche (2017)

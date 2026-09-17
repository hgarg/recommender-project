# Design notes

This document records the main design decisions for the recommender system and the rationale behind them. The notes reflect the working design process and should be read as ongoing documentation rather than a final report.

## Adaptive blending strategy

The first version of the design simply stated that "more weight should shift toward content-based recommendations for new users." While that description was directionally correct, it was not operationally precise and could not be reproduced. To make the idea measurable, I replaced the informal statement with a simple piecewise linear weighting rule.

The current formulation is a linear interpolation between two thresholds:

- below `n_min`: use content-based recommendations only
- above `n_full`: use collaborative filtering only
- between the two thresholds: linearly ramp the weight from content to collaborative filtering

This is simpler to explain and more transparent than a sigmoid or other nonlinear formulation. It also makes the threshold logic easier to reason about during implementation and evaluation. The values `n_min = 3` and `n_full = 20` were chosen heuristically rather than through tuning. They are not yet justified by empirical evaluation, so they should be treated as provisional design parameters rather than final model choices.

The actual function is implemented in `src/blending.py`, since it is a small, self-contained component that does not require the rest of the pipeline to exist first. The synthetic data generator in `src/generate_synthetic_data.py` and the implementation notebook in `notebooks/implementation.ipynb` now provide the working evidence that the design is being tested in practice. The rest of the source files remain lightweight at this stage, but the design is no longer purely conceptual.

## Baseline definition

The initial project objective was too broad when phrased as simply "predict what users might like." That statement is not sufficiently measurable for a rigorous evaluation. To make the task more concrete, I narrowed the comparison to a small set of interpretable baselines:

1. SVD-only recommendation
2. TF-IDF-only recommendation
3. a fixed 50/50 hybrid

The fixed hybrid is especially important because it provides a simple reference point. If the adaptive model does not outperform the fixed blend by a meaningful margin, then the added complexity of adaptive weighting may not be justified. In practice, the adaptive model should be considered valuable only if it produces a real gain beyond a simple fixed strategy.

The evaluation target is set as follows:

- improve NDCG@10 by at least 5% over the best baseline
- improve cold-start recall over the fixed hybrid baseline

This is deliberately a two-part criterion. A model that improves NDCG alone could still fail to address the true cold-start problem, so the cold-start recall condition is included to ensure the adaptive strategy is doing what it was intended to do.

## Sampling and validation plan

The raw-versus-filtered distribution comparison has not yet been run. This step is still part of the planned implementation work rather than a completed analysis. It is recorded in the project documentation and in the dataset provenance notes, but the actual numbers are still pending.

This is an important validation step because the final model will only be meaningful if the filtered sample preserves the distributional properties that matter for the recommendation task. Without this check, it is unclear whether the preprocessing step changes the underlying data in a way that would distort the evaluation.

## Synthetic data generation

I also implemented a synthetic data generator in `src/generate_synthetic_data.py` so the project has an actual reproducible data source instead of only a conceptual description. The goal was to approximate the broad statistical properties of the Amazon Electronics review dataset, including high sparsity, a strong 4-5 star skew, and a long-tail popularity distribution.

The first pass of the generator was too aggressive in its popularity skew. The top 5% of products accounted for more than 70% of all interactions, which was far above the target. After adjusting the Zipf-style weighting, the distribution moved much closer to the desired range. With seed 42, the final run produced a top 5% share of around 42%, compared with the target of roughly 47% from the earlier exploratory analysis.

This is close enough for the current stage of the project, and it is not worth spending additional time tuning the generator to an exact value unless the final model evaluation suggests that the mismatch materially affects performance. In contrast, the positivity rate and sparsity landed close to the target in the first realistic pass, which was encouraging.

## Practical next steps

A few small validation tasks are still needed:

- add a small set of unit tests for `blend_weight`
- run the raw-vs-filtered sampling comparison
- evaluate the fixed hybrid and adaptive hybrid models against the same metrics
- revisit the threshold values (`n_min`, `n_full`) once the results are available

At the moment, the main validation is still visual and heuristic rather than formal, but this is acceptable at the design stage. The important point is that the design choices are explicit and can be tested systematically once the full pipeline is running.

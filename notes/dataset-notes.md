# Dataset Notes

This file records the main findings from the preliminary exploratory data analysis. The
figures are working estimates until the preprocessing pipeline is formalised and rerun
from the original source data.

## Dataset profile

| Stage | Interactions | Users | Products | Sparsity |
| --- | ---: | ---: | ---: | ---: |
| Raw data | 67,967 | 4,862 observed | 1,988 | 99.30% |
| Filtered sample | 59,768 | 2,574 | 1,666 | 98.61% |

The raw interaction matrix is highly sparse, which is expected in review and e-commerce
datasets because most users interact with only a small portion of the catalogue. The
filtered sample remains sparse while being more practical for local experimentation.

## Sampling and filtering

The current working sample is produced using minimum activity thresholds: at least five
interactions per user and ten per product. This is threshold-based filtering rather than
formal stratified sampling. A later validation will compare the raw and filtered
distributions and determine whether activity-decile stratification is needed to retain
more lower-activity users and products for cold-start analysis.

The activity thresholds have not yet been finalised. They must be defined explicitly and
included in a reproducible preprocessing script before the modelling results can be
treated as final.

## Preliminary observations

- **Rating concentration:** 72.1% of ratings are four or five stars in the current
  generated data. This suggests a
  substantial positivity bias and raises questions about using raw star ratings as a
  direct measure of preference. Interaction weighting will be investigated.
- **Popularity concentration:** the most active five percent of products account for
  42.4% of all interactions in the current generated data. The earlier 47.2% figure is
  retained as a target for the generator rather than an observed result. This indicates a
  pronounced long-tail effect
  and provides a reason to evaluate catalogue coverage and diversity in addition to
  ranking accuracy.
- **Initial content representation:** the TF-IDF matrix is planned to use product category,
  price bucket, and average rating. Its final dimensions will be reported after the
  preprocessing and content-based pipeline are implemented.

## Outstanding data tasks

- Write a preprocessing script that reproduces the exploratory sample from the original
  data source.
- Confirm the source data, provenance, and licensing information.
- Define and record the activity thresholds used for filtering and cold-start groups.
- Recalculate all summary statistics after the reproducible pipeline is in place.

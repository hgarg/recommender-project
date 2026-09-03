# Dataset Notes

This file records the main findings from the preliminary exploratory data analysis. The
figures are working estimates until the preprocessing pipeline is formalised and rerun
from the original source data.

## Dataset profile

| Stage | Interactions | Users | Products | Sparsity |
| --- | ---: | ---: | ---: | ---: |
| Raw data | 67,967 | 5,000 | 1,988 | 99.32% |
| Filtered sample | 56,417 | 1,754 | 756 | 95.75% |

The raw interaction matrix is highly sparse, which is expected in review and e-commerce
datasets because most users interact with only a small portion of the catalogue. The
filtered sample remains sparse while being more practical for local experimentation.

## Sampling and filtering

The working sample would be produced using stratification by user and product activity.
Users and products would be grouped according to the amount of observed activity, and
observations would be sampled proportionally across those groups. This approach would be
chosen in preference to a single blanket threshold so that lower-activity users and
products remain represented in the data used for cold-start analysis.

The activity thresholds have not yet been finalised. They must be defined explicitly and
included in a reproducible preprocessing script before the modelling results can be
treated as final.

## Preliminary observations

- **Rating concentration:** 72.2% of ratings are four or five stars. This suggests a
  substantial positivity bias and raises questions about using raw star ratings as a
  direct measure of preference. Interaction weighting will be investigated.
- **Popularity concentration:** the most active five percent of products account for
  approximately 47.2% of all interactions. This indicates a pronounced long-tail effect
  and provides a reason to evaluate catalogue coverage and diversity in addition to
  ranking accuracy.
- **Initial content representation:** a TF-IDF matrix constructed from product category,
  price bucket, and average rating had shape `(756, 14)`. The small feature count is a
  consequence of the short content field. Product descriptions or other metadata may be
  added if they are available in the final source data.

## Outstanding data tasks

- Write a preprocessing script that reproduces the exploratory sample from the original
  data source.
- Confirm the source data, provenance, and licensing information.
- Define and record the activity thresholds used for filtering and cold-start groups.
- Recalculate all summary statistics after the reproducible pipeline is in place.

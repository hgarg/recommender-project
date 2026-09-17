# Project Notes

These notes document the reasoning behind the planning phase report and the subsequent
methodology refinement. They are intended to preserve the decisions, assumptions, and
open questions that will guide implementation; they are not a substitute for the final
capstone report.

## Scope and objectives

The initial project proposal described a general hybrid recommender using SVD and TF-IDF.
Following feedback, the scope would be narrowed to an e-commerce setting using a dataset
structured like Amazon Electronics review data. The main technical question is now the
combination of the two models rather than simply their independent performance.

The proposed system will use an activity-dependent blending weight rather than a fixed
ratio. Products with little interaction history will rely more heavily on the TF-IDF
model, while products with established histories will rely more heavily on collaborative
filtering. This is intended to address cold-start conditions directly. The approach is
consistent with the weighted hybrid systems discussed by Burke (2002), while responding
to the limitations of static combinations described by Adomavicius and Tuzhilin (2005).

## Implementation progress

The project has moved beyond pure planning and into a working implementation phase.
The repository now includes a reproducible synthetic data generator in
[`src/generate_synthetic_data.py`](../src/generate_synthetic_data.py), a small hybrid
weighting function in [`src/blending.py`](../src/blending.py), and an implementation
notebook in [`notebooks/implementation.ipynb`](../notebooks/implementation.ipynb).
These components provide the first end-to-end local workflow for data generation,
model setup, and preliminary experimentation. The current focus is on validating the
preprocessing assumptions and ensuring that the notebook and source files remain aligned
with the project documentation.

## Need and feasibility

Recommendation relevance is a practical concern in e-commerce because product discovery
and user engagement depend partly on the quality of the items presented. The more
immediate constraint for this project, however, is computational feasibility. The full
dataset is too large for the available laptop and a typical hosted notebook session.

The working sample would therefore be created through activity-based stratification.
Users and products would be grouped by activity level, and observations would be sampled
proportionally from those groups rather than removed through a single arbitrary cutoff.
This retains lower-activity cases that are important for evaluating cold-start behaviour.
The exploratory counts and sparsity measures are recorded in
[`dataset-notes.md`](dataset-notes.md).

## Risks and evaluation considerations

Two risks are central to the project:

- **Popularity bias:** the most active five percent of products account for approximately
  47% of all interactions. Without additional controls, the model may repeatedly return a
  small set of popular products. Diversity-aware re-ranking is therefore a planned part
  of the later implementation, not only a possible enhancement.
- **Limited ground truth:** the data records observed ratings and interactions, not
  whether a user would have purchased an item if it had been recommended. Evaluation will
  consequently use proxy ranking measures, including precision, recall, NDCG, and catalogue
  coverage, with this limitation reported explicitly.

Accuracy alone will not be treated as a sufficient measure of system quality. Coverage,
diversity, and performance on low-activity products will be considered alongside ranking
metrics, following the concerns raised by McNee, Riedl, and Konstan (2006).

## Proposed methodology

- SVD matrix factorisation will provide the collaborative filtering component.
- TF-IDF will represent product metadata, initially including category, price bucket, and
  average rating.
- An adaptive blending rule will combine the two recommendation scores according to
  available interaction history.
- Preprocessing will include stratified sampling, construction of the TF-IDF content
  field, and a response to the observed positivity bias in ratings. Interactions may need
  to be weighted rather than treating raw star ratings as a direct measure of preference.

The planned implementation uses Python, pandas, scikit-learn, scikit-surprise, and
eventually Streamlit for the demonstration application. The dataset choice is informed by
the Amazon review work discussed by He and McAuley (2016), while the TF-IDF component is
based on the approach described by Ramos (2003).

## Ethics and responsible use

The source data is public and is being used for academic experimentation rather than as
live customer data. Its provenance and licensing conditions will be recorded as the
project develops, and working copies will remain within the project environment.

An extension using identifiable customer data would require a clear legal basis,
appropriate retention limits, a deletion process, and a privacy review under applicable
regulations such as the GDPR and CCPA (Voigt and von dem Bussche, 2017).

Fairness is also relevant to the design. A system that continually reinforces existing
popularity can reduce exposure for newer or smaller sellers. This is one reason that
diversity and catalogue coverage are included in the intended evaluation rather than
leaving the system to optimise relevance in isolation.

## Open questions

- What activity thresholds should define the low-, medium-, and high-history groups?
- What weighting scheme best accounts for the strong concentration of four- and five-star
  ratings?
- How much does the adaptive blend improve cold-start performance relative to fixed-weight
  and single-model baselines?
- How should diversity be introduced without materially reducing recommendation relevance?

## Selected references

Adomavicius and Tuzhilin (2005); Burke (2002); He and McAuley (2016); Koren, Bell, and
Volinsky (2009); McNee, Riedl, and Konstan (2006); Ramos (2003); Voigt and von dem
Bussche (2017).

# Hybrid Recommender for E-Commerce

This repository contains the planning and implementation work for a capstone project on
hybrid recommendation systems. The project investigates whether an adaptive combination
of collaborative filtering and content-based recommendation can address the cold-start
problem more effectively than either approach used alone.

The intended outcome is a reproducible Python implementation, a small Streamlit
application through which users can inspect recommendations, and a deployable project
artifact. At present, the repository is in the planning and methodology phase. The
implementation, evaluation, and deployment materials will be developed incrementally as
the capstone progresses.

## Project status and roadmap

This repository is being developed across the full capstone lifecycle. The current phase
is **planning and methodology**, with the research scope, dataset considerations, and
evaluation strategy documented in `notes/`. The planned sequence is:

1. **Planning:** establish the research question, review relevant literature, document
	the dataset, and define the evaluation strategy.
2. **Implementation:** formalise preprocessing, build the collaborative and content-based
	models, and implement the adaptive hybrid recommender in `src/`.
3. **Evaluation:** compare the hybrid approach with appropriate baselines using ranking
	accuracy, coverage, diversity, and cold-start performance.
4. **Application:** integrate the final pipeline into the Streamlit interface in `app/`.
5. **Deployment:** package and deploy the application, document the runtime environment,
	and verify that the deployed artefact behaves consistently with the evaluated system.

The implementation and deployment stages are intentionally not represented as complete
while the project is still in planning. This status will be updated as each milestone is
completed.

## Current implementation status

The project has progressed from the planning stage into an initial implementation phase.
A reproducible synthetic dataset generator is now in place, and a first version of the
adaptive blending rule has been implemented. These components provide a working foundation
for local experimentation while the remainder of the pipeline remains under active development.

The synthetic dataset is generated in `src/generate_synthetic_data.py` and is designed to
approximate the statistical properties of a sparse e-commerce review dataset, including
high sparsity, a strong positive rating skew, and long-tail product popularity. A brief
summary of the data provenance and validation checks is recorded in `notes/data-origin.md`
and `notes/design-notes.md`.

The adaptive recommendation rule is implemented in `src/blending.py`. This component
adjusts the relative contribution of collaborative filtering and content-based filtering
according to user activity, allowing a stronger content-based signal for cold-start users
and a stronger collaborative signal for users with more interaction history. At this
stage, the implementation is intentionally lightweight and serves as a working prototype
for evaluation rather than a final production design.

The repository also includes preliminary project notes documenting the design rationale,
baseline strategy, and validation criteria. The current implementation remains incomplete,
and the preprocessing pipeline, final model training, and evaluation results will be added
as the capstone progresses.

## Research question

How does a hybrid recommender with an activity-dependent blending weight perform on a
sparse e-commerce interaction dataset, particularly when products have limited or no
interaction history?

## Project rationale

Collaborative filtering can capture patterns in user behaviour, but it depends on
sufficient interaction history. Content-based methods can recommend newer products from
their metadata, although they do not learn the broader preferences represented in the
interaction matrix. This project combines the two approaches and varies their relative
contribution according to product activity:

- products with limited history receive greater weight from the content-based model;
- products with stronger interaction histories receive greater weight from the
	collaborative model.

This design treats cold start as a modelling consideration rather than as a limitation
reported only after evaluation. The project will also examine catalogue coverage and
recommendation diversity alongside ranking accuracy, since a recommender that repeatedly
returns only the most popular products is not sufficient for a long-tail marketplace.

## Proposed methodology

The planned system consists of three components:

1. **Collaborative filtering:** SVD matrix factorisation learned from user-product
	 interactions.
2. **Content-based filtering:** TF-IDF representations built from product metadata,
	 including category, price bucket, and average rating.
3. **Adaptive hybridisation:** a blending rule that adjusts the contribution of each
	 model according to available interaction history.

The analysis will use ranking-oriented evaluation measures such as precision, recall, and
NDCG, together with catalogue coverage. A later stage of the project will investigate
diversity-aware re-ranking to reduce the effect of popularity bias.

### Data preparation

The dataset is shaped like an Amazon Electronics review dataset and is used for academic
experimentation rather than production deployment. Because the full dataset is not
practical to process in the current development
environment, the working sample would be constructed using activity-based stratification.
This retains users and products across activity levels instead of applying an
undifferentiated random cut. The sample also preserves the conditions relevant to the
cold-start analysis.

Two characteristics require particular care during modelling:

- 72.2% of ratings are four or five stars, indicating a strong positivity bias;
- the most active five percent of products account for approximately 47.2% of all
	interactions, indicating a substantial long-tail effect.

For this reason, raw star ratings will not be treated as an unqualified measure of
preference. Interaction weighting and diversity-aware analysis will be considered as
part of the evaluation design.

## Repository structure

```text
recommender-project/
├── app/                    # Streamlit application and deployment entry point (planned)
├── data/
│   ├── processed/          # Generated, cleaned data (not committed)
│   └── raw/                # Source data (not committed)
├── notebooks/              # Exploratory analysis and experiments
├── notes/                  # Project rationale, data notes, and methodology
├── src/                    # Reusable preprocessing and modelling code (planned)
├── tests/                  # Tests for the implementation
├── requirements.txt        # Python dependencies
└── README.md
```

The current substantive material is in [project-notes.md](notes/project-notes.md) and
[dataset-notes.md](notes/dataset-notes.md). These documents record the scope, exploratory
findings, methodological decisions, and ethical considerations that will guide the
implementation.

## Reproducibility

The environment can be prepared with Python and a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dependency versions are not yet pinned. They will be recorded once the implementation
environment is established. The dependency list currently reflects the libraries used in
this phase, including `pandas`, `numpy`, `scikit-learn`, and `matplotlib`.

The synthetic dataset can be regenerated from the project source folder:

```bash
cd src
python generate_synthetic_data.py --out-dir ../data/raw --seed 42
```

This produces the local raw data files used for the current working sample. The generated
files are not committed to the repository and can be replaced by changing the random seed
or by updating the generation parameters.

## Ethics and limitations

The source data is public and is being used for academic purposes. It does not represent
live customer data, but data provenance, licensing, and local handling will be documented
as the project develops. Any extension to identifiable customer data would require an
appropriate legal basis, retention policy, deletion process, and privacy review.

The project also treats popularity bias as a substantive design issue. A model that
reinforces existing exposure can make it harder for newer or less-established products to
be discovered. Accuracy will therefore be reported alongside coverage and diversity, with
the limitations of implicit behavioural data stated explicitly.

## Selected references

- Adomavicius, G., & Tuzhilin, A. (2005). Toward the next generation of recommender
	systems: A survey of the state-of-the-art and possible extensions. *IEEE Transactions
	on Knowledge and Data Engineering, 17*(6), 734–749.
- Burke, R. (2002). Hybrid recommender systems: Survey and experiments. *User Modeling
	and User-Adapted Interaction, 12*, 331–370.
- He, R., & McAuley, J. (2016). Ups and downs: Modeling the visual evolution of fashion
	trends with one-class collaborative filtering. In *Proceedings of the 25th
	International Conference on World Wide Web*.
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for
	recommender systems. *Computer, 42*(8), 30–37.
- McNee, S. M., Riedl, J., & Konstan, J. A. (2006). Being accurate is not enough: How
	accuracy metrics have hurt recommender systems. In *CHI '06 Extended Abstracts on
	Human Factors in Computing Systems*.
- Ramos, J. (2003). Using TF-IDF to determine word relevance in document queries. In
	*Proceedings of the First Instructional Conference on Machine Learning*.
- Voigt, P., & von dem Bussche, A. (2017). *The EU General Data Protection Regulation
	(GDPR): A practical guide*. Springer.

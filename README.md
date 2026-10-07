# Hybrid Recommender for E-Commerce

This repository contains the planning and implementation work for a capstone project on
hybrid recommendation systems. The project investigates whether an adaptive combination
of collaborative filtering and content-based recommendation can address the cold-start
problem more effectively than either approach used alone.

The intended outcome is a reproducible Python implementation, a small Streamlit
application through which users can inspect recommendations, and a deployable project
artifact. The work was developed incrementally across the capstone (DSC-580/590, Grand
Canyon University), and the progress log below records each stage as it was completed.

As of the latest update, the repo has the full pipeline, the analysis notebooks, unit
tests, and a small Streamlit app for looking at the recommendations. The final hybrid gets
NDCG@10 = 0.119 on the held-out test set, compared with 0.100 for SVD alone (see
[Main results](#main-results)).

## Project status and roadmap

This repository was developed across the full capstone lifecycle. The research scope,
dataset considerations, and evaluation strategy are documented in `notes/`. The planned
sequence, with where each stage is now, is:

1. **Planning:** establish the research question, review relevant literature, document
	the dataset, and define the evaluation strategy. Status: done (`notes/`).
2. **Implementation:** formalise preprocessing, build the collaborative and content-based
	models, and implement the adaptive hybrid recommender in `src/`. Status: done (`src/`,
	`notebooks/implementation.ipynb`).
3. **Evaluation:** compare the hybrid approach with appropriate baselines using ranking
	accuracy, coverage, diversity, and cold-start performance. Status: done, in
	`notebooks/model_insight_validation.ipynb` and
	`notebooks/advanced_analysis_refinement_final.ipynb`.
4. **Application:** integrate the final pipeline into the Streamlit interface in `app/`.
	Status: done (`app/app.py`, which uses `src/recommender.py`).
5. **Deployment:** package and deploy the application, document the runtime environment,
	and verify that the deployed artefact behaves consistently with the evaluated system.
	Status: done. The app is deployed on Streamlit Community Cloud (see
	[Deployed application](#deployed-application)), the package versions are pinned, and a
	test checks that the app's model still gives the reported numbers.

## Progress log

### Update 1: Implementation phase

The project has progressed from the planning stage into an initial implementation phase.
A reproducible synthetic dataset generator is now in place, and a first version of the
adaptive blending rule has been implemented. These components provide a working foundation
for local experimentation while the remainder of the pipeline remains under active development.

The synthetic dataset is generated in `src/generate_synthetic_data.py` and is designed to
approximate the statistical properties of a sparse e-commerce review dataset, including
high sparsity, a strong positive rating skew, and long-tail product popularity. The data
provenance and generation rationale are documented in `notes/data-origin.md`, and the
related design decisions are recorded in `notes/design-notes.md`.

The adaptive recommendation rule is implemented in `src/blending.py`. This component
adjusts the relative contribution of collaborative filtering and content-based filtering
according to user activity, allowing a stronger content-based signal for cold-start users
and a stronger collaborative signal for users with more interaction history. At this
stage, the implementation is intentionally lightweight and serves as a working prototype
for evaluation rather than a final production design.

The implementation work is now reflected in the repository through the synthetic data
pipeline, the hybrid weighting function, and the exploratory notebook in `notebooks/`.
This milestone focuses on making the design reproducible and testable locally, with the
next stage centred on refinement of the preprocessing and evaluation workflow. The
repository still retains earlier planning material for context, but the current codebase
should be read as the active implementation stage of the project.

The repository also includes preliminary project notes documenting the design rationale,
baseline strategy, and validation criteria. The current implementation remains incomplete,
and the preprocessing pipeline, final model training, and evaluation results will be added
as the capstone progresses.

### Update 2: Model insight and validation

The full pipeline (`src/pipeline.py`) now builds SVD, TF-IDF, fixed and adaptive hybrids
and a most-popular baseline on a time-based train/validation/test split. The validation
notebook (`notebooks/model_insight_validation.ipynb`) adds bootstrap confidence intervals,
paired significance tests, results by customer history, coverage and fairness checks, and
privacy controls (`src/privacy.py`).

### Update 3: Refinement and application

Six changes were tried one at a time and tuned on the validation set
(`notebooks/advanced_analysis_refinement_final.ipynb`). The resulting final model is
packaged in `src/recommender.py` and served by a Streamlit app (`app/app.py`). Package
versions are now pinned.

#### Main results

All numbers are on the held-out test set (time-based split), K = 10.

| Model | Precision@10 | Recall@10 | NDCG@10 | Catalog coverage |
| --- | --- | --- | --- | --- |
| Final hybrid | 0.037 | 0.163 | **0.119** | 46% |
| SVD only | 0.032 | 0.139 | 0.100 | 21% |
| Fixed 50/50 hybrid | 0.031 | 0.132 | 0.099 | 54% |
| Most popular | 0.022 | 0.105 | 0.072 | 3.5% |

- The final model beats both SVD and the starting 50/50 blend (paired Wilcoxon, p < .001)
  and was better on all five data seeds I tried.
- The original adaptive hybrid did **not** perform as intended. It gave customers with 3 or fewer
  interactions zero SVD weight, and for customers with 1-2 interactions it scored close to
  random. SVD still had a useful signal for them (mostly their favourite category), so
  turning it off made things worse. The final model keeps SVD for everyone.
- The biggest single improvement came from removing price bucket from the content
  features. That is probably a property of the synthetic data (price isn't really a
  preference signal in the generator), so it should not be assumed to hold on real data.
- There is still a trade-off: the final model covers less of the catalog than the 50/50
  blend.

#### Final model

- SVD on the raw rating matrix, 12 latent factors
- TF-IDF on product category, customer profile = mean similarity to products rated 4+
- Blend: 0.1 x SVD + 0.9 x content (alpha picked on validation)
- Popularity penalty before ranking, lambda = 0.0025 (largest value that kept validation
  NDCG within 1%)

The settings live in `src/recommender.py`, which is what the app uses.

## Research question

How does a hybrid recommender with an activity-dependent blending weight perform on a
sparse e-commerce interaction dataset, particularly for customers and products with
limited or no interaction history?

*Update:* the question was broadened from products only to customers and products. In the
implementation, the blending weight depends on the **customer's** history, and the
evaluation only covers new or low-history customers; see [Limitations](#limitations).

## Project rationale

Collaborative filtering can capture patterns in user behaviour, but it depends on
sufficient interaction history. Content-based methods can recommend newer products from
their metadata, although they do not learn the broader preferences represented in the
interaction matrix. This project combines the two approaches and varies their relative
contribution according to product activity:

- products with limited history receive greater weight from the content-based model;
- products with stronger interaction histories receive greater weight from the
	collaborative model.

*Update:* in the implementation, the adaptive weight was driven by the **customer's**
number of training interactions, not the product's. Refinement then showed that
switching SVD off for low-history customers hurt them, so the final model uses one fixed
blend weight for everyone (see [Update 3](#update-3-refinement-and-application)).

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

*Update:* how the plan turned out in the end:

- SVD was kept as planned (12 latent factors on the raw rating matrix).
- The content features ended up as **category only**. Price bucket was dropped during
  refinement, and average rating was not used in the final model.
- The adaptive blending rule was replaced by a fixed blend (0.1 x SVD + 0.9 x content).
- The diversity-aware re-ranking was implemented as a small popularity penalty
  (lambda = 0.0025).

### Data preparation

The dataset is shaped like an Amazon Electronics review dataset and is used for academic
experimentation rather than production deployment. Because the full dataset is not
practical to process in the current development
environment, the working sample would be constructed using activity-based stratification.
This retains users and products across activity levels instead of applying an
undifferentiated random cut. The sample also preserves the conditions relevant to the
cold-start analysis.

Two characteristics require particular care during modelling:

- 72.1% of ratings are four or five stars in the current generated data, indicating a
	strong positivity bias;
- the most active five percent of products account for approximately 42.4% of all
	interactions in the current generated data, indicating a substantial long-tail effect.

For this reason, raw star ratings will not be treated as an unqualified measure of
preference. Interaction weighting and diversity-aware analysis will be considered as
part of the evaluation design.

*Update:* the data is synthetic and generated to look like the Amazon Electronics review
data (see `notes/data-origin.md`). After filtering to customers and products with at
least 3 interactions, there are 66,321 interactions, 3,849 customers and 1,984 products.

## Repository structure

```text
recommender-project/
├── app/                    # Streamlit app (app.py)
├── data/
│   ├── processed/          # Generated, cleaned data (not committed)
│   └── raw/                # Generated source data (not committed)
├── images/                 # Diagrams used in the design documentation
├── notebooks/              # Implementation, validation and refinement notebooks
├── notes/                  # Project rationale, data notes, and methodology
├── reports/figures/        # Figures exported from the notebooks
├── research/               # Small one-off checks
├── src/                    # Pipeline, metrics, blending, privacy, final model
├── tests/                  # pytest tests
├── requirements.txt        # Pinned Python dependencies
└── README.md
```

The planning material is in [project-notes.md](notes/project-notes.md) and
[dataset-notes.md](notes/dataset-notes.md). These documents record the scope, exploratory
findings, methodological decisions, and ethical considerations that guided the
implementation. The analysis itself is in the three notebooks listed in the
[progress log](#progress-log).

## Reproducibility

The environment can be prepared with Python (I used 3.9.6; the pinned versions also
support up to 3.12) and a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*Update:* package versions are now pinned in `requirements.txt`. These are the versions I
used when I re-ran everything from a clean install for the final audit.

**Streamlit app** (from the repo root):

```bash
streamlit run app/app.py
```

The first load trains the models, which takes around 15-20 seconds. After that it is
cached. The app lets you pick a model, a customer (filtered by how much history they
have) and K, and shows their past ratings next to the recommendations, with a flag for
items they actually liked in the test period. There is also a tab with the test-set
metrics.

*Update:* the app was extended for the final submission with four tabs:

- **Recommendations:** each recommended product now has a short "why recommended"
  explanation based on how the selected model scores it (for example, the number of the
  customer's 4-5 star products in the same category), so the results are transparent.
- **Model performance:** test-set metrics for all models, with charts for ranking
  accuracy, the accuracy-coverage trade-off and NDCG@10 by customer history. The model
  selected in the sidebar is highlighted.
- **About:** the purpose of the project, the data, the models compared, the evaluation
  design, key findings and limitations.
- **Help & contact:** a glossary of terms, frequently asked questions and contact
  details.

### Deployed application

The app is deployed on Streamlit Community Cloud:

**URL:** [https://recommender-project.streamlit.app](https://recommender-project.streamlit.app/)

Access is protected by a simple password. The password is shared with the report
submission.

The deployed app uses the same code and data seed as the evaluation, and its "Model
performance" tab shows the same test-set results as the [Main results](#main-results)
table (final hybrid NDCG@10 = 0.1193).

When the app is run locally without a password configured, it opens directly without the
password screen.

**Tests:**

```bash
python -m pytest -q
```

`tests/test_recommender.py` also checks that the final NDCG@10 is still 0.1193, so if
something in the pipeline changes the test will catch it.

**Notebooks:** the notebooks generate the data themselves (seed 42), so they can be run
top to bottom without any downloads. The synthetic dataset can also be regenerated with
the standalone generator, from the project source folder:

```bash
cd src
python generate_synthetic_data.py --out-dir ../data/raw --seed 42
```

This produces the local raw data files used for the current working sample. The generated
files are not committed to the repository and can be replaced by changing the random seed
or by updating the generation parameters.

## Ethics and limitations

The data is synthetic, generated locally by `src/generate_synthetic_data.py`, and is used
for academic purposes only. It does not represent live customer data; its provenance and
local handling are documented in `notes/data-origin.md`. Any extension to identifiable customer data would require an
appropriate legal basis, retention policy, deletion process, and privacy review.

The project also treats popularity bias as a substantive design issue. A model that
reinforces existing exposure can make it harder for newer or less-established products to
be discovered. Accuracy will therefore be reported alongside coverage and diversity, with
the limitations of implicit behavioural data stated explicitly.

*Update:* no real customer data is used. The pipeline still has the controls it would
need for real data (`src/privacy.py`): data minimisation, keyed pseudonymisation of
customer IDs, file integrity hashes and encrypted exports, with tests in
`tests/test_privacy.py`. Coverage is reported next to accuracy in the
[main results](#main-results).

### Limitations

- **Synthetic data.** The results show the method works on data with these properties,
  not that it will work on real customers.
- **New products were not tested.** The test set only keeps products that appear in
  training, so the project only answers the new-customer part of the research question.
- **Activity filter.** The minimum of 3 interactions is applied to the whole timeline
  before the split. It should be computed on the training period only.
- **Ties.** Some models (TF-IDF especially) produce lots of tied scores, and the tie order
  is not fixed, so their 4th decimal can move a little between runs. The final model, SVD
  and the 50/50 blend reproduce exactly.
- **Offline only.** A 4-star rating in the test period is not the same as a click or a
  purchase. An A/B test would be needed to know the real effect.

## Selected references

- Abdollahpouri, H., Burke, R., & Mobasher, B. (2017). Controlling popularity bias in
  learning-to-rank recommendation. In *Proceedings of the 11th ACM Conference on
  Recommender Systems* (pp. 42–46).

- Adomavicius, G., & Tuzhilin, A. (2005). Toward the next generation of recommender
	systems: A survey of the state-of-the-art and possible extensions. *IEEE Transactions
	on Knowledge and Data Engineering, 17*(6), 734–749.
- Burke, R. (2002). Hybrid recommender systems: Survey and experiments. *User Modeling
	and User-Adapted Interaction, 12*, 331–370.
- Cremonesi, P., Koren, Y., & Turrin, R. (2010). Performance of recommender algorithms on
  top-N recommendation tasks. In *Proceedings of the 4th ACM Conference on Recommender
  Systems* (pp. 39–46).
- He, R., & McAuley, J. (2016). Ups and downs: Modeling the visual evolution of fashion
	trends with one-class collaborative filtering. In *Proceedings of the 25th
	International Conference on World Wide Web*.
- Herlocker, J. L., Konstan, J. A., Terveen, L. G., & Riedl, J. T. (2004). Evaluating
  collaborative filtering recommender systems. *ACM Transactions on Information Systems,
  22*(1), 5–53.
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for
	recommender systems. *Computer, 42*(8), 30–37.
- McNee, S. M., Riedl, J., & Konstan, J. A. (2006). Being accurate is not enough: How
	accuracy metrics have hurt recommender systems. In *CHI '06 Extended Abstracts on
	Human Factors in Computing Systems*.
- Ramos, J. (2003). Using TF-IDF to determine word relevance in document queries. In
	*Proceedings of the First Instructional Conference on Machine Learning*.
- Voigt, P., & von dem Bussche, A. (2017). *The EU General Data Protection Regulation
	(GDPR): A practical guide*. Springer.

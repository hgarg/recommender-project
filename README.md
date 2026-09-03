# capstone-recommender

Hemant Garg — DSC-580/590, Grand Canyon University

## where this actually stands

Putting this repo up now, during planning, instead of waiting until the project is basically done — that was feedback on my last submission (establish the repo early, commit as you go instead of dumping everything at the end).

Right now there's not much "real" in here yet. What's committed so far is the write-up work behind my planning phase report and this week's proposal/methodology refinement assignment — mostly notes, not code. The folders for data, notebooks, src, app, and tests are stubbed out so the structure exists, but I didn't want to fill them with placeholder starter code just to make the repo look more built-out than it is. That'll get filled in for real once I'm actually in the implementation milestone.

## the project, briefly

Hybrid recommender for an e-commerce-style dataset (Amazon Electronics-ish interactions). Combines SVD collaborative filtering with a TF-IDF content-based model. The part I'm most attached to is using a dynamic blend weight instead of a fixed one — new items lean on the TF-IDF side, items with more history lean on SVD — so cold start is handled on purpose instead of just being a known weakness I shrug at in the writeup.

End goal is something people can actually click through (Streamlit), not just a notebook that only makes sense to me.

## layout

- `notes/` — the actual content right now (see below)
- `data/` — empty except placeholders, raw/processed data isn't going in git (see .gitignore)
- `notebooks/`, `src/`, `app/`, `tests/` — empty, reserved for the implementation phase

## notes so far

- `notes/project-notes.md` — scope/objectives, needs analysis, ethics stuff, basically the working version of my proposal refinement
- `notes/dataset-notes.md` — numbers from my earlier EDA pass, kept here so I stop losing track of them between assignments

## eventually

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

nothing to actually run yet.

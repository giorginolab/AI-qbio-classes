# Supervised prediction, done honestly

Practical section (8 h) for **AI in Quantitative Biology**, MSc Quantitative Biology.

Four two-hour sessions building a QSAR regression model — predicting acute
aquatic toxicity from molecular descriptors — where the modelling stays simple
and the *evaluation* is the subject. Students leave able to audit a supervised
learning result, their own or someone else's.

## Where to put this

Drop the contents next to the `data/` folder that holds the two CSVs:

```
classes/
├── data/
│   ├── qsar_fish_toxicity.csv
│   └── qsar_aquatic_toxicity.csv
├── notebooks/
│   ├── 01_first_model.ipynb                 <- hand these to students
│   ├── 02_variance_and_leakage.ipynb
│   ├── 03_model_selection.ipynb
│   ├── 04_trust_and_domain.ipynb
│   └── *_solutions.ipynb                    <- keep these back
├── qsar_helpers.py
├── INSTRUCTOR_GUIDE.md
├── requirements.txt
└── README.md
```

`qsar_helpers.find_data_dir()` walks up from the notebook's directory looking
for `data/`, so the layout above works with no configuration. If the data lives
elsewhere, set `QSAR_DATA` to that folder.

## Setup

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv run jupyter lab
```

To run the solution notebooks headless — useful as a regression test after
editing the material, since any cell that raises aborts the run:

```bash
uv run python run_solutions.py                                   # all four
uv run python run_solutions.py notebooks/03_model_selection_solutions.ipynb
```

Executed copies, with all outputs and figures, land in `executed/`.

Everything runs on a laptop CPU. No GPU, no network access during the
sessions. Total compute for all four solution notebooks is a few minutes.

## Sessions

| # | Title | Core question |
|---|---|---|
| 1 | A first model, and the baseline it has to beat | Compared to what? |
| 2 | Where the honest number comes from | Averaged over what? |
| 3 | Choosing a model without fooling yourself | How many things did you try? |
| 4 | What the model knows, and where it stops working | On which molecules? |

Session 4 ends with the capstone: the same workflow applied from scratch to the
second dataset, handed in as a notebook plus a 250-word methods paragraph.

## The data

Two datasets from the UCI Machine Learning Repository, produced by the Milano
Chemometrics and QSAR Research Group (Cassotti, Ballabio, Todeschini, Consonni).

- **`qsar_fish_toxicity.csv`** — 908 molecules, 6 descriptors, 96 h LC50 toward
  *Pimephales promelas* (fathead minnow). Used in sessions 1-4.
- **`qsar_aquatic_toxicity.csv`** — 546 molecules, 8 descriptors, 48 h LC50
  toward *Daphnia magna*. Used for the collinearity exercise in session 4 and
  for the capstone.

Both are semicolon-separated with no header row; the column names are in
`qsar_helpers.py`. Targets are `-log10(mol/L)`, so **larger means more toxic**.

## Prerequisites

Python, NumPy and pandas at working level. No prior scikit-learn needed — the
API is introduced as it is used. No chemistry needed beyond the one-line
descriptor glosses in `qsar_helpers.DESCRIPTOR_NOTES`.

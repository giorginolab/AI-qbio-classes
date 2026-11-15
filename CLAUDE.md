# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Teaching material, not an application, for practical classes for *AI in Quantitative Biology*.
See `README.md` for the layout, data sources and how to run things.

Do not create worktrees unless necessary. Work in main.

## Layout

- `classes/` holds **the three classes actually taught** (150/90/150 min), a deck and a notebook each:
  `class1_basics` (fish-toxicity QSAR, from a line to a network), `class2_neural_networks_enzymes`
  (MLP + enzyme EC classes + the homology trap) and `class3_rnn_targeting` (GRU on targeting signals).
  It has its own `README.md` and `make_decks.py`. **Do not modify anything in `classes/`.**
- `extra_topics/` holds the **supplementary** notebooks, regrouped by macro-topic so that none of them repeats
  a class: `topic1_training_neural_networks` (PyTorch from scratch + the overfitting playground),
  `topic2_convolutional_networks` (CNN on N-termini, filter interpretation, CNN vs GRU, shuffling control,
  transformers), `topic3_learning_without_labels` (autoencoders + peptide generation),
  `topic4_transfer_scaling_evaluation` (ESM-2 vs descriptors, scaling, the Merck split, provenance) and
  `extra01`-`extra03` (scaling laws, laws vs flexible models, ESM-2 on enzymes). `extra01`/`extra02` are
  toxicity-free alternatives to class 1 / class 2.
- The `topic*` notebooks were cut from the older `class1`-`class3` and `t01`-`t03` notebooks (removed in
  the "regroup by macro-topic" commit; recover them from git history if needed). They cross-reference the
  classes by number and each other by file name.
- `data/` holds the datasets (committed). `data_prep/` holds the scripts that built them
  (`make_enzyme_dataset.py`, `make_targeting_dataset.py`, `make_scaling_datasets.py`,
  `make_esm2_embeddings.py`); they need internet and are instructor-only.
- `slides/` holds teaching-notes `.pptx` decks (new `.pptx` are git-ignored; use `git add -f`).
- `REFERENCES.md` lists the papers cited on the slides; `literature/` holds PDFs.
- `protein_bert/` (upstream code) and `env_proteinbert/` (pinned TensorFlow env) are only for the ProteinBERT demo.
- `carlos-course/` and `old/` are separate or earlier material.

## Conventions

- **The notebooks are the source.** There is no generator; edit the `.ipynb` directly.
- Solutions are inline (✏️ exercises, ❓ questions, answer in the next cell). Seeds are fixed (`RNG = 0`)
  and the numbers quoted in the answers are the ones the notebook prints: if you change a seed, split or
  model, re-run and update the text.
- Support-code cells are collapsed: first line `# @title ...` plus `jupyter.source_hidden` and
  `cellView: form` metadata.
- Notebooks find `data/` by walking up from the cwd; `$QBIO_DATA` overrides.
- Several exercises are **deliberate nulls** (e.g. scaler leakage, duplicate leakage, selection bias at
  n=908), each paired with a regime where the effect is large. Do not "fix" them to show a bigger effect.
- The committed datasets are the reference: re-running a `data_prep` script (Swiss-Prot changes every
  release) shifts the quoted numbers.
- The fish and aquatic CSVs are headerless and `;`-separated; column names are defined in the notebooks.
- Marimo: `data_prep/make_marimo.py` (notebook -> marimo, swapping ipywidgets sliders for marimo forms)
  is kept but **only run it when the user explicitly asks**, and do not create a `marimo/` directory
  otherwise. It reads notebooks from `extra_topics/` and writes to `extra_topics/marimo/`. Its playground table is
  **stale**: it still keys `t01`/`t02`/`t03`, which no longer exist (the playgrounds now live in
  `topic1` and in `classes/class3_rnn_targeting`).
- nbconvert script exports (`extra_topics/*.py`), `__pycache__/`, `.ipynb_checkpoints/`, `.venv/`, `.#*` and
  `data/proteinbert_model/` are git-ignored noise.

## Commands

```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
uv run jupyter lab

# regression test (no test suite or linter): execute a notebook headless; any raising cell aborts
uv run jupyter nbconvert --to notebook --execute --output-dir "$TMPDIR" extra_topics/<notebook>.ipynb

# ProteinBERT demo (TensorFlow, separate env)
uv sync --project env_proteinbert
```

Requires scikit-learn >= 1.4 (`root_mean_squared_error`); PyTorch (CPU) for `t03` and `class2`. Compute is
CPU-only; runtimes are seconds to a few minutes per notebook.

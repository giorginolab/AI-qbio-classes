# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Teaching material, not an application, for practical classes for *AI in Quantitative Biology*.

Do not create worktrees unless necessary. Work in main.

## New 3-session course (`course/`)

`course/` is a separate, newer set of three 90-min practicals (QSAR toxicity, enzyme EC classes, GRU on protein targeting signals), in the style of `carlos-course/`: one self-contained notebook per session, solutions inline. There the **notebooks are the source** (no generator). `course/marimo/*.py` are marimo versions built by `course/make_marimo.py` (it also swaps the ipywidgets sliders for marimo forms); re-run it after editing a notebook. nbconvert script exports are gitignored. Support-code cells are collapsed: first line `# @title ...` plus `jupyter.source_hidden` and `cellView: form` metadata. Datasets `data/enzymes.tsv` and `data/targeting.tsv` come from `course/data_prep/`. See `course/README.md` for the key numbers quoted in the answers. `s01_scaling_laws` and `s02_laws_vs_flexible_models` are toxicity-free alternatives to `t01`/`t02` (scaling laws on `data/anage.tsv` and `data/genomes.tsv`, built by `course/data_prep/make_scaling_datasets.py`); both sets are kept. Everything below describes the older material.

## Commands

```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
uv run jupyter lab

python build_notebooks.py                       # regenerate student + solution notebooks
uv run python run_solutions.py                  # execute all *_solutions.ipynb headless (regression test)
uv run python run_solutions.py notebooks/03_model_selection_solutions.ipynb   # one notebook
```

There is no test suite or linter; `run_solutions.py` is the test (any cell that raises aborts the run). Executed copies with outputs go to `executed/`.

## Architecture

- **`build_notebooks.py` is the single source of truth for all notebooks.** Each session is a list `S1`..`S4` of cells built with `md(...)` and `code(student, solution=None)`. `build()` emits two notebooks per session: the student one (scaffold) and `*_solutions.ipynb` (uses the `solution=` variant where given). Edit this file, never the `.ipynb` files directly, or the two versions drift and your edits are overwritten.
- **`qsar_helpers.py`** is imported by every notebook (`sys.path.insert(0, "..")` in the shared `HEADER` cell). It loads the two headerless, semicolon-separated CSVs in `data/` with column names defined in the module, and provides `xy`, `fold_error`, `molecule_groups` (GroupKFold groups from identical descriptor rows), and `score_table`. `find_data_dir()` checks `$QSAR_DATA`, then walks up from the cwd looking for `data/`.
- Datasets: `qsar_fish_toxicity.csv` (908 × 6 descriptors, sessions 1-4) and `qsar_aquatic_toxicity.csv` (546 × 8, session 4 collinearity + capstone). Target `LC50` is `-log10(mol/L)`; larger = more toxic.

## Conventions worth preserving

- Seeds are fixed (`RNG = 0`); the instructor guide quotes numbers to three decimals, so changing seeds, splits, or models invalidates the guide and the expected values in solution markdown.
- Several exercises are **deliberate nulls** (scaler leakage, aggregate duplicate leakage, selection bias at n=908), each paired with a regime where the effect is large. Do not "fix" them to show a bigger effect.
- Requires scikit-learn ≥ 1.4 (`root_mean_squared_error`). Compute is CPU-only and runs in a few minutes total.
- `__pycache__/`, `.ipynb_checkpoints/` and `.#CLAUDE.md` (an editor lock file) are noise.

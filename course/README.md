# AI in Quantitative Biology - three practicals (3 x 90 min)

Self-contained notebooks, solutions included (exercises ✏️ and questions ❓ are followed by their
solution). Each `.py` file is a `jupyter nbconvert --to script` export of the notebook with the same name.
Everything runs on a laptop CPU in about 2 minutes per notebook.

| | notebook | biology | data mining | ML |
|---|---|---|---|---|
| 1 | `t01_toxicity_qsar` | toxicity of 908 chemicals to fish (QSAR); homework on *Daphnia* | distributions, correlations, PCA | linear regression, kNN, random forest, cross-validation, permutation importance |
| 2 | `t02_neural_networks_enzymes` | A: same fish data. B: EC class of 2047 enzymes from sequence | class/length/organism tables, amino-acid enrichment heatmap, PCA | MLP (overfitting, regularisation); logistic regression, RF, MLP classifiers; confusion matrix; **family-grouped CV vs random CV** |
| 3 | `t03_rnn_protein_targeting` | signal peptide / mitochondrial transit peptide / none, from the first 50 residues (mini-TargetP) | positional profiles of hydropathy and charge | composition baseline, one-hot encoding, **GRU in PyTorch**, shuffled-order control, per-residue predictions, cross-species test |

## Key numbers (fixed seeds, `RNG = 0`)

* **t01** (5-fold CV R²): MLOGP alone 0.41, linear 0.56, kNN 0.62 (unscaled 0.57), RF 0.60.
  *Daphnia*: linear 0.46, RF 0.51, kNN 0.48 scaled vs 0.24 unscaled.
* **t02** MLP 0.58 vs RF 0.60 (an MLP does not beat RF on a small table); a wide MLP reaches train
  R² 0.93 / test 0.47, and with `alpha=1` the test R² is 0.56.
  Enzymes (chance 0.17): LR 0.29, RF 0.40, MLP 0.29 with random CV; 0.27 / 0.31 / 0.27 with protein
  families held out. Composition is a weak signal, and homology inflates the scores.
* **t03** composition baseline 0.80 (50 aa) / 0.87 (best window, 25 aa); GRU 0.94 (5k parameters,
  ~5 s); GRU on shuffled residues 0.80 -> the gain is residue order. Non-human -> human 0.89.

Changing seeds, splits or models shifts these values and the answers written in the notebooks.

## Running

```bash
uv venv --python 3.12 && uv pip install -r course/requirements.txt
uv run jupyter lab
```

The notebooks look for `data/` in the current folder and its parents, then in `$QBIO_DATA`. On
Colab, put `data/` in `MyDrive/AI_QBio/data`.

## Data (`../data/`)

* `qsar_fish_toxicity.csv`, `qsar_aquatic_toxicity.csv`: UCI QSAR datasets (Cassotti et al. 2015),
  headerless, `;`-separated. `LC50` is `-log10(mol/L)` (larger = more toxic).
* `enzymes.tsv`: 2047 reviewed UniProt/Swiss-Prot enzymes from human, *E. coli* K-12, yeast,
  *Arabidopsis* and *B. subtilis*, EC classes 1-6 (~350 each), at most 8 per protein family.
  Built by `data_prep/make_enzyme_dataset.py`.
* `targeting.tsv`: 3600 N-termini (50 aa) from the same organisms, 1200 each SP / mTP / other.
  Built by `data_prep/make_targeting_dataset.py`.

Both datasets were downloaded from the UniProt REST API on 2026-11-10. Swiss-Prot changes at each
release, so re-running the scripts gives a slightly different set. The committed TSVs are the
reference for the numbers above.

## Note: why the RNN does not predict EC classes

We first tried a bidirectional GRU on full enzyme sequences for the EC task. It needed ~45 s per
epoch on CPU and reached only ~0.29 accuracy after 12 epochs, which is too slow and too weak for a
90-minute class. Short N-terminal targeting signals are a task where sequence order clearly
matters and training takes seconds, so t03 uses that instead.

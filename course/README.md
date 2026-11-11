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

## Alternative sessions 1 and 2: scaling laws (`s01`, `s02`)

Two newer notebooks that can replace `t01` and `t02` (which are kept unchanged). They contain no
toxicity material and are built around one contrast: a two-parameter law against flexible models.
They run in under a minute and need no PyTorch. `t03` follows them as session 3.

| | notebook | biology | ML |
|---|---|---|---|
| 1 | `s01_scaling_laws` | metabolic rate and lifespan against body mass (AnAge); homework on other traits | log-log axes, least squares, bootstrap interval, residuals, a categorical variable, Simpson's paradox |
| 2 | `s02_laws_vs_flexible_models` | same data; genes against genome size in prokaryotes and eukaryotes | linear, kNN, RF, MLP; cross-validation; extrapolation; **order-grouped CV vs random CV**; dirty data |

Key numbers (`RNG = 0`):

* **s01** mammal metabolic rate: exponent 0.709 (bootstrap 95% 0.688 - 0.728, excluding both 2/3
  and 3/4), R² 0.95; by size 0.60 / 0.71 / 0.82. Birds 0.655 (0.635 - 0.674). Mammal longevity:
  exponent 0.159, R² 0.50; human 4.5x and Brandt's bat 6.4x the predicted lifespan. All classes
  pooled R² 0.82; with one intercept per class 0.96 (reptiles at 6% of a mammal). "Careless"
  dataset slope 1.11.
* **s02** 5-fold CV R²: linear 0.949, kNN 0.947, RF 0.928, MLP 0.953. Trained below 1 kg, R² above:
  0.79 / -1.06 / -1.04 / 0.83; elephant (2336 W measured) 415 / 3 / 3 / 402 W, and 16 - 402 W over
  five MLP seeds. Longevity from four traits, random CV vs orders held out: linear 0.78 / 0.59,
  kNN 0.79 / 0.50, RF 0.80 / 0.48; bats held out are predicted 20% too short. Prokaryotes: genes ∝
  size^0.964, R² 0.98, 942 genes/Mb; that law predicts 2.3 M human genes, the forest 11 000
  (annotated 57 816). Eukaryotes: exponent 0.28, R² 0.60.

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

* `anage.tsv`: 3663 animal species from AnAge build 15 (Tacutu et al. 2018): taxonomy, adult mass,
  maximum longevity, basal metabolic rate (627 species) and a few reproductive traits.
* `genomes.tsv`: genome size and number of annotated genes for 13 751 prokaryotes (complete
  genomes, no organism names) and 4685 eukaryotes, one row per species, from the NCBI genome
  reports. A few absurd eukaryote gene counts are left in on purpose (exercise 4.1 of `s02`).
  Both built by `data_prep/make_scaling_datasets.py`, downloaded on 2026-11-11.

The enzyme and targeting datasets were downloaded from the UniProt REST API on 2026-11-10. Swiss-Prot changes at each
release, so re-running the scripts gives a slightly different set. The committed TSVs are the
reference for the numbers above.

## Note: why the RNN does not predict EC classes

We first tried a bidirectional GRU on full enzyme sequences for the EC task. It needed ~45 s per
epoch on CPU and reached only ~0.29 accuracy after 12 epochs, which is too slow and too weak for a
90-minute class. Short N-terminal targeting signals are a task where sequence order clearly
matters and training takes seconds, so t03 uses that instead.

# AI in Quantitative Biology - practicals

`course/` has six notebooks. t01–t03 are the original three 90-minute sessions. extra01 and extra02 are
toxicity-free replacements for t01 and t02 (t03 stays as session 3), and extra03 is an optional extra
practical (homework). Each notebook has its exercise (✏️) and question (❓) solutions inline, and all of them run
on a laptop CPU.

## Main track

**t01 – Mining a toxicity dataset** (fish LC50 of 908 chemicals, 6 molecular descriptors)
- **Exploring the data:** distributions, correlations with toxicity, and PCA.
- **Linear regression:** a one-descriptor model first, then the loss landscape and gradient descent
  drawn by hand, to show what "fitting" means.
- **Comparing models:** kNN (with and without scaling) and random forest, compared with 5-fold
  cross-validation. Part 5 then covers:
  - how much a single train/test split varies;
  - kNN underfitting versus overfitting;
  - learning curves;
  - a polynomial overfitting playground with sliders.
- **Interpretation:** permutation importance.
- **Homework:** the *Daphnia* (water flea) dataset.
- **Key numbers (5-fold CV R²):** MLOGP alone 0.41, linear 0.56, kNN 0.62 (0.57 unscaled), RF 0.60.

**t02 – Neural networks, and classifying enzymes**
- **Part A, MLPs on the fish data:**
  - The MLP (0.58) doesn't beat the random forest (0.60) on a small table.
  - What function each model learns, and how they extrapolate.
  - The effect of network width, overfitting epoch by epoch, and L2 regularisation: a wide MLP
    reaches train 0.93 / test 0.47, and `alpha=1` brings test back to 0.56.
  - A slider playground.
- **Part B, predicting the EC class of 2047 enzymes from amino-acid composition:**
  - Enrichment heatmap and PCA, then logistic regression, RF and MLP classifiers with confusion
    matrices.
  - **The homology trap:** with random CV, accuracy is LR 0.29, RF 0.40, MLP 0.29 (chance 0.17).
    With protein families held out it drops to 0.27 / 0.31 / 0.27, because related sequences
    inflate the scores.

**t03 – A recurrent network reads protein N-termini** (a mini-TargetP: signal peptide /
mitochondrial transit peptide / other, from the first 50 residues)
- **Data mining along the sequence:** hydropathy and charge profiles by position.
- **Composition baseline:** 0.80 with all 50 residues, 0.87 with the best 25-residue window.
- **The GRU:** one-hot encoding, then a GRU in PyTorch (5k parameters, about 5 s to train)
  reaching **0.94**. Also covered:
  - loss versus accuracy;
  - a playground for making the GRU overfit;
  - a control with shuffled residues, which falls back to 0.80, so the gain comes from residue
    order.
- **Interpretation:** per-residue predictions as the network reads, PCA of the hidden states,
  gradient × input saliency.
- **Extra:** a cross-species test (trained on non-human, tested on human: 0.89).

## Alternative sessions 1–2: scaling laws (AnAge and genome data)

Both are built around one contrast: a two-parameter law against flexible models.

**extra01 – Scaling laws**
- **Fitting the law:** metabolic rate against body mass in 422 mammals, using log-log axes and
  least squares. The exponent is 0.709, and its bootstrap interval rules out both 2/3 and 3/4.
- **Residuals:** they show the slope changing with body size.
- **Lifespan:** the residuals are the biology. Humans live 4.5× and Brandt's bat 6.4× longer than
  the law predicts.
- **The pooling trap:** pooling all classes gives one line with R² 0.82; one intercept per class
  gives 0.96 (Simpson's paradox).
- **Homework:** other traits.

**extra02 – What a flexible model cannot do**
- **Within the data**, linear, kNN, RF and MLP are tied (CV R² about 0.95).
- **Extrapolation** (trained below 1 kg): the law and the MLP predict the elephant at 415 / 402 W
  (measured 2336), kNN and RF at 3 W. Five MLP seeds range from 16 to 402 W.
- **Species are not independent:** holding out whole taxonomic orders cuts R² sharply
  (RF 0.80 → 0.48), and held-out bats are predicted 20% too short-lived.
- **A law that breaks:** genes against genome size fits prokaryotes almost exactly (R² 0.98), but
  that law predicts 2.3 million human genes. Also covers cleaning dirty data.

## Optional session 4

**extra03 – What a protein language model knows about enzymes** (about 60 min, scikit-learn only,
with precomputed ESM-2 embeddings)
- **Same classifier, different input:** the t02 enzyme task with the same logistic regression.
  Composition gives 0.29 / 0.27 (random / families held out). ESM-2 embeddings give
  0.65–0.77 / 0.53–0.61.
- **Controls:** shuffled labels (0.15) and a nearest-neighbour check.
- **Scaling with model size:** across 8M, 35M and 150M parameters, accuracy gains level off after
  35M.

## Threads running through the course

- **Leakage and grouped validation** come back three times: homology in t02 and extra03, taxonomic
  orders in extra02, and species in t03.
- **Overfitting** gets a hands-on playground in t01, t02 and t03.
- **Interpretation** runs from permutation importance to residuals to RNN saliency.

## More reference numbers

All numbers use fixed seeds (`RNG = 0`). Changing seeds, splits or models shifts these values and
the answers written in the notebooks.

* **t01** *Daphnia* homework: linear 0.46, RF 0.51, kNN 0.48 scaled vs 0.24 unscaled.
* **extra01** exponent 0.709, bootstrap 95% 0.688 - 0.728, R² 0.95; by size 0.60 / 0.71 / 0.82.
  Birds 0.655 (0.635 - 0.674). Mammal longevity: exponent 0.159, R² 0.50. Reptiles at 6% of a
  mammal (one intercept per class). "Careless" dataset slope 1.11.
* **extra02** 5-fold CV R²: linear 0.949, kNN 0.947, RF 0.928, MLP 0.953. Trained below 1 kg, R² above:
  0.79 / -1.06 / -1.04 / 0.83. Longevity from four traits, random CV vs orders held out: linear
  0.78 / 0.59, kNN 0.79 / 0.50, RF 0.80 / 0.48. Prokaryotes: genes ∝ size^0.964, 942 genes/Mb; the
  forest predicts 11 000 human genes (annotated 57 816). Eukaryotes: exponent 0.28, R² 0.60.
* **extra03** accuracy, random split / families held out: ESM-2 8M 0.65 / 0.53; 35M 0.76 / 0.61; 150M
  0.77 / 0.61. 1-NN on the 8M embeddings 0.70 / 0.54. Embeddings compressed to 100 columns by PCA,
  families held out: 0.53, 0.57, 0.54. The embeddings (`data/enzymes_esm2.npz`) are built by
  `data_prep/make_esm2_embeddings.py`, which needs PyTorch and `transformers` and uses a GPU if
  there is one.

## Running

```bash
uv venv --python 3.12 && uv pip install -r course/requirements.txt
uv run jupyter lab
```

All data are in `data/` in this repository. The notebooks use `$QBIO_DATA` if it is set,
otherwise they look for `data/` in the current folder and its parents, so they work from any
folder of a clone, locally or on Colab.

Support code that students need not read (setup, figure-only cells, playgrounds and their sliders)
is collapsed: such cells start with `# @title ...` and have `jupyter.source_hidden` and
`cellView: form` set, so JupyterLab, VS Code and Colab show them folded.

The `.ipynb` notebooks are the source. `make_marimo.py` can build marimo versions on request (`uv run python course/make_marimo.py`); none are kept in the repo. Script exports (`jupyter nbconvert --to script`) are not kept in git.

## Data (`../data/`)

* `qsar_fish_toxicity.csv`, `qsar_aquatic_toxicity.csv`: UCI QSAR datasets (fish: Cassotti et al. 2015;
  *Daphnia*: Cassotti et al. 2014), headerless, `;`-separated. `LC50` is `-log10(mol/L)` (larger = more toxic).
* `enzymes.tsv`: 2047 reviewed UniProt/Swiss-Prot enzymes from human, *E. coli* K-12, yeast,
  *Arabidopsis* and *B. subtilis*, EC classes 1-6 (~350 each), at most 8 per protein family.
  Built by `data_prep/make_enzyme_dataset.py`.
* `targeting.tsv`: 3600 N-termini (50 aa) from the same organisms, 1200 each SP / mTP / other.
  Built by `data_prep/make_targeting_dataset.py`.

* `anage.tsv`: 3663 animal species from AnAge build 15 (Tacutu et al. 2018): taxonomy, adult mass,
  maximum longevity, basal metabolic rate (627 species) and a few reproductive traits.
* `genomes.tsv`: genome size and number of annotated genes for 13 751 prokaryotes (complete
  genomes, no organism names) and 4685 eukaryotes, one row per species, from the NCBI genome
  reports. A few absurd eukaryote gene counts are left in on purpose (exercise 4.1 of `extra02`).
  Both built by `data_prep/make_scaling_datasets.py`, downloaded on 2026-11-11.

* `enzymes_esm2.npz`: mean-pooled ESM-2 embeddings (float16) of the enzymes in `enzymes.tsv`,
  same row order, one array per model size.

The enzyme and targeting datasets were downloaded from the UniProt REST API on 2026-11-10. Swiss-Prot changes at each
release, so re-running the scripts gives a slightly different set. The committed TSVs are the
reference for the numbers above.

## Note: why the RNN does not predict EC classes

We first tried a bidirectional GRU on full enzyme sequences for the EC task. It needed ~45 s per
epoch on CPU and reached only ~0.29 accuracy after 12 epochs, which is too slow and too weak for a
90-minute class. Short N-terminal targeting signals are a task where sequence order clearly
matters and training takes seconds, so t03 uses that instead.

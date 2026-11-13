# AI in Quantitative Biology — the lecture deck, split into three classes

This directory splits `slides/ai-qbio-unimi_2026_r1.pdf` (79 slides) into **three self-contained
classes**, and gives each one a hands-on notebook that puts the claims of those slides on real data.

The split follows the deck's own structure — it already announces its parts on slides 2, 3, 27 and 47.
Each split PDF carries the title slide, then its own block.

| class | slides | PDF | notebook |
|---|---|---|---|
| 1 — Basics of artificial neural networks | 1–26 | `slides/class1_basics_of_neural_networks.pdf` | `class1_from_regression_to_networks.ipynb` |
| 2 — Advanced architectures | 27–46 | `slides/class2_advanced_architectures.pdf` | `class2_architectures.ipynb` |
| 3 — Applications, and what to distrust | 47–79 | `slides/class3_applications_and_limits.pdf` | `class3_applications.ipynb` |

Every notebook is self-contained, fixed-seed (`RNG = 0`), CPU-only, and **ships its solutions inline**:
exercises are marked ✏️ and open questions ❓, with the answer in the following cell. Numbers quoted in
the answers are the numbers the notebook actually prints — if you change a seed, a split or a model, the
answers go stale.

## Running them

```bash
uv venv --python 3.12 && uv pip install -r ../course/requirements.txt
uv run jupyter lab          # from the repository root
```

Marimo versions are in `marimo/` (built from the notebooks by `uv run python course/make_marimo.py`; re-run it after
editing a notebook): `uv run marimo edit lectures/marimo/class1_from_regression_to_networks.py`.

The notebooks find `../data/` by walking up from the working directory; set `$QBIO_DATA` to override
(useful on Colab).

## Class 1 — From a regression line to a neural network  (~20 s to run)

*Slides 1–26: induction, regression, the artificial neuron, loss and gradient descent, overfitting,
featurization.* Data: 908 chemicals, 6 molecular descriptors, fathead-minnow LC50.

- Induction from data: which descriptor tracks toxicity (MLOGP, r = 0.65).
- Regression as a box (slide 9): one descriptor CV R² 0.41, six descriptors 0.56, and what the
  coefficient +0.39 means to a toxicologist.
- One artificial neuron in four lines of numpy — the identity neuron reproduces `LinearRegression`
  exactly, and a single ReLU buys nothing.
- The loss landscape as a contour plot with gradient-descent paths; it **diverges at lr = 0.2**, and the
  narrow valley explains why.
- Width scan: 1 unit = linear (0.557), 2 units collapse to −0.019 (dead ReLUs: an *optimization* failure,
  not a capacity one), 20 units best (0.598), 100 units worse (0.571).
- Slide 21's figure reproduced from scratch: train MSE falls to 0.231 while test MSE turns around at
  epoch 218 and climbs from 0.885 to 1.109 — followed by the critical reading the slide invites, that
  stopping on the test set makes that score biased, so you need three splits.
- 200 random 75/25 splits of the same data give R² between 0.41 and 0.69 — why one split is not a result.
- Featurization controls: random features and shuffled labels both give R² ≈ 0, calibrating what
  "learned nothing" looks like.
- Homework: the *Daphnia magna* dataset (546 × 8; linear CV R² ≈ 0.46).

## Class 2 — Advanced architectures  (~55 s to run)

*Slides 27–46: deep networks, CNNs, RNNs, autoencoders, and a first look at transformers and protein
language models.* Data: 3600 protein N-termini (50 residues) labelled signal peptide / mitochondrial
transit peptide / other.

- **Featurization** (slide 25): one-hot encoding shown as an image, why integer coding of residues is
  wrong, and a hand-crafted baseline — amino-acid composition + logistic regression, **0.802**.
- **PyTorch in ten minutes** (an interlude, slide 26): tensors and the float32/int64 rule, autograd
  checked against the derivative we did by hand in class 1, `nn.Linear` shown to *be* the 63-parameter
  logistic regression, and the three-line training loop — including what silently goes wrong if you
  forget `zero_grad()`.
- **Dense vs convolutional** (slides 29–32): a dense net on the flattened one-hot (32 131 parameters,
  0.857) against a 1-D CNN (**2 947** parameters, **0.912**) — ten times smaller and better, because
  convolution + global max-pooling *is* a learned, translation-invariant featurization. Filter-width
  sweep: 3 → 0.788, 9 → 0.912, 21 → 0.931, saturating at the length of the hydrophobic h-region.
- **Opening the black box** — the payoff slide 31 promises: the four filters most correlated with
  Kyte-Doolittle hydropathy (r up to +0.93) are exactly the four that vote "signal peptide", their
  top-activating 9-mers are `LCWVLLCLL`, `FLLLLAFLL`, `CLLLLLLLA`, and one peaks at residue 8 in SPs
  (activation 2.41) versus 0.33 elsewhere. The mTP filters are the Arg-rich ones (`MLRRLLRSS`). Nobody
  told the network about hydropathy or charge.
- **GRU head to head** (slide 33): same data and loop, 5 283 parameters, **0.944** — order beats motifs.
- **Control**: shuffling residues within each sequence costs the GRU 0.149 and the CNN 0.220, dropping
  the CNN *below* the bag-of-counts baseline. An inductive bias is a bet that costs you when it is wrong.
- **Autoencoder** (slide 34): a 2-D bottleneck trained only to reconstruct, labels never used, supports
  15-NN accuracy **0.761** against **0.650** for PCA. The same trick on enzyme composition fails (0.232
  for 6 EC classes) — the explicit hand-off to class 3's pretrained embeddings.
- Transformers are a text-only teaser here (slides 36–46); class 3 does the experiment.

## Class 3 — Applications, and what to distrust  (~25 s to run)

*Slides 47–79: drug-discovery applications, transfer learning and scaling, the Merck Kaggle challenge,
ethics and data provenance.* Data: 2047 enzymes (EC class, 1102 families) with precomputed ESM-2
embeddings, plus the toxicity tables.

- **Hand-crafted vs pretrained representation**, framed on deephERG's "185 MOE descriptors + 100
  Mol2vec embeddings" (slide 50): same logistic regression, amino-acid composition **0.291** vs ESM-2 8M
  **0.646** (chance 0.167). Concatenating both gives 0.644 — the hand-crafted features add *nothing*.
- **Two axes of scaling** (slide 44): someone else's GPU-months (8M/35M/150M → 0.646 / 0.754 / 0.773)
  against your own pipetting (ESM-2 with **50** labels, 0.352, already beats composition with **1600**,
  0.303). A PCA control shows part of the model-size trend is embedding width, not knowledge.
- **The Merck lesson** (slides 67–69): under a random split 61.8% of test proteins have a family-mate in
  training; holding families out costs the embeddings 0.646 → 0.530 but composition only 0.291 → 0.270 —
  the better model loses more. Then the molecular analogue of Merck's time split: sorting by a descriptor
  and splitting on it drops a random forest from R² 0.604 to as low as 0.044, and **reverses** the
  forest-vs-linear ranking on one descriptor. Trees cannot extrapolate: the forest's largest prediction
  is 6.28 where measured test values reach 9.35.
- **Where do the data come from?** (slides 59, 72): organism × EC crosstab, family-size distribution,
  descriptor ranges, and an exercise to write the limitations paragraph such a model deserves.
- **Optional**: a 17k-parameter character-level GRU generating signal peptides (slide 54), judged
  honestly — a detector calls 73% of them SP, and max identity to the training set shows no memorisation.

## Relation to `course/`

`course/` is a separate, longer practical set built around the same datasets (QSAR toxicity, enzyme EC
classes, a GRU on protein targeting signals), with marimo versions and data-preparation scripts. The
notebooks here are **keyed to the lecture slides** and are shorter; where they overlap, the `course/`
notebook is the fuller treatment and is cross-referenced from the relevant section.

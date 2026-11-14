# Three classes of 90 minutes (proposal)

A cut of the repository down to three 90-minute classes, each alternating blocks of slides with blocks
of a notebook. Nothing outside this folder was changed: `course/`, `slides/` and `old/` are as they were.

| Class | Deck | Notebook | Built from | Slides / notebook / buffer (min) |
|---|---|---|---|---|
| 1. Basics of neural networks | `Class 1.pptx` (32 slides, 5 hidden) | `class1_basics.ipynb` (42 cells) | `course/class1` | 39 / 49 / 2 |
| 2. Flexible models, honest scores, the homology trap | `Class 2.pptx` (17 slides, 2 hidden) | `class2_models_and_enzymes.ipynb` (63 cells) | `t01` + `t02` | 31 / 55 / 4 |
| 3. A recurrent network reads protein N-termini | `Class 3.pptx` (42 slides, 5 hidden) | `class3_rnn_targeting.ipynb` (45 cells) | `t03` | 45 / 39 / 6 |

Each notebook opens with its minute-by-minute timetable (which slides, then which notebook part). The
timetables are estimates.

## Class 1

Starts from `slides/0 Class 1.pptx`. The deck is unchanged except for one move: "An optimization problem"
and "Backpropagation" (were 27-28) now come before the three overfitting slides (were 24-26), so that the
slides follow the notebook: loss, gradient descent, network, then overfitting. The notebook is
`course/class1_from_regression_to_networks.ipynb` with its content intact and its slide references
re-keyed to this deck (the original cited the numbering of an older 79-page PDF).

## Class 2

Notebook, three parts:

- **A (from `t01`)**: one train/test split, 5-fold cross-validation, kNN and random forest against linear
  regression, 200 random splits, the random-forest scatter, learning curves, permutation importance.
  Left out because class 1 already covers them on the same data: meeting the data, correlations, the
  single-descriptor fit and the loss landscape. Also left out: PCA, standardised coefficients, the kNN
  exercises, the polynomial playground and the Daphnia homework (which is the homework of class 1).
- **B (from `t02` Part A)**: the MLP, the function each model learned along MLOGP, network size and
  regularisation. Left out: overfitting epoch by epoch (class 1 Part 6 shows the same curve) and the
  MLP playground.
- **C (from `t02` Part B)**: the enzymes, amino-acid composition, three classifiers and the confusion
  matrix, the homology trap. Left out: Exercise B1.1 and the PCA map of compositions.

Deck: title, deephERG (a toxicity model built on descriptors), the Part II divider and deep networks,
featurization, evaluation of binary classifiers, the Merck challenge, convolutional networks. The
"Evaluation of (binary) classifiers" slide is hidden in your decks; it is **shown** here because Part C
is where classification and the confusion matrix first appear.

## Class 3

Notebook: `t03` with Parts 1-7 kept (data, profiles along the sequence, composition baseline, one-hot
encoding, the GRU, the shuffling control, watching the network read, saliency). Left out: loss versus
accuracy, the GRU playground, Exercise 7.1 (misclassified proteins) and the PCA of the hidden states.

Deck: title, recurrent networks, autoencoders and generative models, transformers up to ProteinBERT,
transfer learning and scaling, Part III applications (without deephERG, used in class 2), final words,
take-home messages, resources. The Merck slides are in class 2. Four closing slides that duplicated
earlier ones (a second "Final words", "Take-home messages", "Resources (self-study)" and "End") were
left out.

## What is not in the three classes

Everything stays in `course/`.

| Material | Note |
|---|---|
| `class2_architectures` (CNN, GRU, autoencoder on targeting signals) | Class 2 now shows convolutional networks on slides only. |
| `class3_applications` (ESM-2 embeddings, scaling, the Merck split on molecules) | Class 3 covers transformers and transfer learning on slides only. |
| `extra01`, `extra02` (scaling laws) | A separate thread, not connected to the deck. |
| `extra03_enzymes_esm2` | The natural follow-up to class 2 Part C: the same enzymes with a pretrained representation. |
| ProteinBERT demo | Class 3 slides 17-18 point to it. |

## Status and caveats

- The notebooks were **not executed**. Every cell kept its stored output from the notebook it came from.
  Classes 2 and 3 were checked statically for undefined names after the cuts (none left), but the printed
  numbers are unverified until each notebook is run once:
  `uv run jupyter nbconvert --to notebook --execute --output-dir "$TMPDIR" classes90/<notebook>.ipynb`.
- Three code cells differ from their originals, only by added definitions that the cut cells used to
  provide: `X`, `y` and two imports at the start of class 2 Part A, and `y_te`, `colours` in the saliency
  cell of class 3.
- Class 2 merges two notebooks. Part B reloads the fish data and redoes the same split as Part A, as it
  did when it was a separate session. In the 200-splits figure of Part A the red line is still labelled
  "Part 4" (the label is inside a stored image); the text says so.
- Hidden slides were otherwise kept hidden. In class 1 these include linear separability and the
  evaluation of binary classifiers.
- The decks are `.pptx`, which the repository ignores, so they are not in git. `make_decks.py` rebuilds
  them from `slides/0 Class 1.pptx` and `old/ai-qbio-unimi_2026_r1.pptx`; run it from the repository root:
  `uv run --no-project --with python-pptx python classes90/make_decks.py`.
  `Class 1.pptx` is 5 MB against 19 MB for the original because unused media were not carried over.
- Classes 2 and 3 open with the title slide only ("Hands-on 2", "Hands-on 3"); the course outline
  slide is in class 1.

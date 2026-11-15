# Three classes: 150, 90 and 150 minutes (proposal)

A cut of the repository down to three classes, each alternating blocks of slides with blocks of a
notebook. Nothing outside this folder was changed: `course/`, `slides/` and `old/` are as they were.
(The folder name dates from when all three were 90 minutes.)

| Class | Length | Deck | Notebook | Built from | Slides / notebook / break / buffer (min) |
|---|---|---|---|---|---|
| 1. From a regression line to a neural network | 150 | `Class 1.pptx` (32 slides, 5 hidden) | `class1_basics.ipynb` (84 cells) | `course/class1` + `t01` | 43 / 93 / 10 / 4 |
| 2. Neural networks, and classifying enzymes | 90 | `Class 2.pptx` (17 slides, 2 hidden) | `class2_neural_networks_enzymes.ipynb` (46 cells) | `t02` | 32 / 50 / 0 / 8 |
| 3. A recurrent network reads protein N-termini | 150 | `Class 3.pptx` (42 slides, 5 hidden) | `class3_rnn_targeting.ipynb` (58 cells) | `t03` | 41 / 92 / 10 / 7 |

Each notebook opens with its minute-by-minute timetable (which slides, then which notebook part). The
slide numbers quoted in a notebook are those of the deck of the same class. The timetables are estimates.

## Class 1 (150 min)

Notebook: `course/class1_from_regression_to_networks.ipynb`, with the parts of `t01` it did not already
cover placed where they belong:

- **Part 1** gains the histograms, a warm-up exercise and the PCA map: look at the data before modelling.
- **Parts 2-5** are unchanged (regression as a box, one neuron, gradient descent, width scan).
- **Part 6** (overfitting) is followed by the polynomial playground, right after the overfitting slides.
  Its own "200 random splits" cell was dropped: Part 9 does the same experiment in more detail.
  The break comes here.
- **Part 7** (featurization controls) is unchanged.
- **Parts 8-10** are new, from `t01`: one split with a baseline and linear fits; cross-validation, kNN
  and random forest, the kNN exercises, learning curves; permutation importance.

Not taken from `t01` because Parts 1, 2 and 4 already do it on the same data: meeting the table,
correlations, and the loss landscape.

Deck: `slides/0 Class 1.pptx` with two moves and two text corrections.

- "An optimization problem" and "Backpropagation" come before the three overfitting slides, so the
  slides follow the notebook: loss, gradient descent, network, then overfitting.
- "Interpretability" (the second one, on black boxes) moves after "Featurization", so that it
  introduces Part 10.
- The Backpropagation slide said the weights are adjusted to decrease the error "on the test set";
  it now says "on the training set". "loss funtion" is corrected to "loss function".

The slide "Training and stopping" still says that training stops when the loss on the test set is
minimal: the notebook asks the students to criticise exactly that sentence, so it was left alone.

## Class 2 (90 min)

Notebook: `t02` with both parts (A: a neural network on the toxicity data; B: enzyme classes from
amino-acid composition, three classifiers, the homology trap). Left out: overfitting epoch by epoch
(class 1 Part 6 shows the same curve) and the MLP playground.

Deck: title, deephERG (a published toxicity model built on descriptors), the Part II divider and deep
networks, featurization, evaluation of binary classifiers, the Merck challenge, convolutional networks.
"Evaluation of (binary) classifiers" is hidden in your decks; it is **shown** here because Part B is
where classification and the confusion matrix first appear.

## Class 3 (150 min)

Notebook: `t03` complete, with a break after Part 5.

Deck: title, recurrent networks, autoencoders and generative models, transformers up to ProteinBERT,
transfer learning and scaling, Part III applications (without deephERG, used in class 2), final words,
take-home messages, resources. The Merck slides are in class 2. Four closing slides that duplicated
earlier ones (a second "Final words", "Take-home messages", "Resources (self-study)" and "End") were
left out. One typo corrected ("performace").

## Teaching notes

Each class also has a teaching-notes deck (`Class N · Teaching notes.pptx`), built by `make_decks.py`
from the per-practical decks in `slides/`. They explain the ideas behind each part of the notebook and
carry the expected numbers in the speaker notes.

- **Class 1** (29 slides, from `slides/1 Fish toxicity QSAR`): reordered to follow `class1_basics`.
  Part 1 is the look at the data, then a new slide covering Parts 2-7, which `Class 1.pptx` teaches.
  After that come the overfitting playground (end of Part 6) and Parts 8-10. The loss-landscape slide
  of t01 was dropped because Part 4 draws its own landscape with different numbers (divergence at
  lr = 0.2, not 0.15). Exercise numbers in the notes follow the class 1 notebook (Warm-up, 4, 6, 7).
- **Class 2** (25 slides, from `slides/2 Neural networks and enzymes`): the deck without the two
  slides on early stopping and the MLP playground, and without the two "to try next" items about them.
- **Class 3** (30 slides, from `slides/3 Recurrent networks and protein targeting`): complete. The
  notes of the Part 5 playground slide mark the break.

In all three decks the small headline on the cover says "Hands On/N" instead of "Practical N"; class 1 is
retitled "Fish toxicity: from a line to a network, and how to trust it". The notes no longer name
the t0x notebooks or "session 4": they point to the class notebooks, the lecture slides and `extra06`.

## What is not in the three classes

Everything stays in `course/`.

| Material | Note |
|---|---|
| `class2_architectures` (CNN, GRU, autoencoder on targeting signals) | Class 2 shows convolutional networks on slides only. |
| `class3_applications` (ESM-2 embeddings, scaling, the Merck split on molecules) | Class 3 covers transformers and transfer learning on slides only. |
| `extra01`, `extra02` (scaling laws) | A separate thread, not connected to the deck. |
| `extra06_enzymes_esm2` | The natural follow-up to class 2 Part B: the same enzymes with a pretrained representation. |
| ProteinBERT demo | Class 3 slides 17-18 point to it. |

## Status and caveats

- The notebooks were **not executed**. Every cell kept its stored output from the notebook it came from.
  All three were checked statically (every name is defined before the cell that uses it), but the printed
  numbers are unverified until each notebook is run once:
  `uv run jupyter nbconvert --to notebook --execute --output-dir "$TMPDIR" classes/<notebook>.ipynb`.
- Class 1 merges two notebooks, so the cells from `t01` will run under the class 1 setup cell (which sets
  pandas display precision to 3 and silences warnings): expect small formatting differences in their
  tables on the first run. One code cell differs from its original: the first cell of Part 8 redefines
  `X` and `y` as a table, because Parts 2-7 use plain arrays.
- In the 200-splits figure of Part 9 the red line is labelled "Part 4" (the label is inside a stored
  image and will stay until the cell's code is edited); the text says so.
- Hidden slides were otherwise kept hidden. In class 1 these include linear separability and the
  evaluation of binary classifiers.
- The decks are `.pptx`, which the repository ignores, so they are not in git. `make_decks.py` only builds
  the three teaching-notes decks, from the per-practical decks in `slides/` (also untracked); run it from the
  repository root: `uv run --no-project --with python-pptx python classes/make_decks.py`. The lecture decks
  (`Class N.pptx`) are no longer built by it.
- Classes 2 and 3 open with the title slide only ("Hands-on 2", "Hands-on 3"); the course outline
  slide is in class 1.

# Three classes of 90 minutes (proposal)

A cut of the repository down to three 90-minute classes, each alternating blocks of slides with blocks
of a notebook. Nothing outside this folder was changed: `course/`, `slides/` and `old/` are as they were.

| Class | Deck | Notebook | Slides / notebook / buffer (min) |
|---|---|---|---|
| 1. Basics of neural networks | `Class 1.pptx` (32 slides, 5 hidden) | `class1_basics.ipynb` (42 cells) | 39 / 49 / 2 |
| 2. Architectures | `Class 2.pptx` (23 slides, 4 hidden) | `class2_architectures.ipynb` (52 cells) | 35 / 55 / 0 |
| 3. Applications and what to distrust | `Class 3.pptx` (37 slides, 3 hidden) | `class3_applications.ipynb` (36 cells) | 37 / 48 / 5 |

Each notebook opens with its minute-by-minute timetable (which slides, then which notebook part). The
slide numbers quoted in a notebook are those of the deck of the same class.

## What was done

**Class 1** starts from `slides/0 Class 1.pptx`. The deck is unchanged except for one move: "An optimization
problem" and "Backpropagation" (were 27-28) now come before the three overfitting slides (were 24-26), so
that the slides follow the notebook: loss, gradient descent, network, then overfitting. The notebook is
`course/class1_from_regression_to_networks.ipynb` with its content intact and the slide references
re-keyed to this deck (the original cited the numbering of an older 79-page PDF).

**Class 2** deck: title, the featurization and software slides from Part I as a recap, then Part II up to
ProteinBERT (slides 33-52 of `old/ai-qbio-unimi_2026_r1.pptx`). Notebook cut from 69 to 52 cells:

- the PyTorch interlude went from 14 cells (tensors, autograd, `nn.Module`, training loop, the
  `zero_grad` exercise) to 3: the three-line training loop on logistic regression, and nothing else;
- Part 3 lost the filter weight heatmaps and the position-of-firing exercise; it keeps the
  hydropathy table and the top-activating 9-mers;
- Part 6 lost the enzyme autoencoder exercise.

**Class 3** deck: title, transfer learning and scaling (the end of Part II, where the notebook starts),
Part III, the Merck slides moved up from the appendix to just before the split experiment, the
"where do the data come from" slide, then final words, take-home and resources. Four closing slides that
duplicated earlier ones (a second "Final words", "Take-home messages", "Resources (self-study)" and "End")
were left out. Notebook cut from 45 to 36 cells: the optional Part 5 (a GRU generating signal peptides)
and the embedding-width control of Part 2 are gone.

## What is left out of the three classes

Everything stays in `course/`; none of it is needed to follow the three classes.

| Material | Suggested use |
|---|---|
| `t01_toxicity_qsar` | Self-study after class 1 (same fish data: PCA, cross-validation, random forests). Its Daphnia homework is already the homework of class 1. |
| `t02_neural_networks_enzymes` | Self-study after class 1 or 2; its homology trap is class 3's Part 3 in longer form. |
| `t03_rnn_protein_targeting` | Self-study after class 2 (the GRU of Part 4 at full length). |
| `extra01`, `extra02` (scaling laws) | A separate thread, not connected to the deck. Project material. |
| `extra03_enzymes_esm2` | Largely overlaps class 3 Parts 1-3. Redundant for the classes. |
| `class1`-`class3` in `course/` | The longer originals of the three notebooks here. |
| ProteinBERT demo | Class 3 slides 5-6 point to it; too slow to run live. |

## Status and caveats

- The notebooks were **not re-executed**. Every cell kept its stored output from the original notebook,
  and the cuts only remove cells, so the printed numbers are the original ones. Class 2 is the one place
  where a removed cell preceded kept ones that train networks; each of those sets its own seed, so the
  numbers should be unchanged, but that is unverified until the notebook is run once.
- The timetables are estimates. Class 2 has no buffer: the notebook names what to skip (the filter-width
  sweep, or Part 6).
- Hidden slides were kept hidden. In class 1 these include linear separability and the evaluation of
  binary classifiers.
- The decks are `.pptx`, which the repository ignores, so they are not in git. `make_decks.py` rebuilds
  them from `slides/0 Class 1.pptx` and `old/ai-qbio-unimi_2026_r1.pptx` (run it from the repository
  root: `uv run --no-project --with python-pptx python classes90/make_decks.py`). `Class 1.pptx` is 5 MB
  against 19 MB for the original because unused media were not carried over.
- Classes 2 and 3 open with the title slide only ("Hands-on 2", "Hands-on 3"); the course outline
  slide is in class 1.

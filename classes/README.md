# Three classes: 150, 90 and 150 minutes

Each class is a deck (`Class N.pptx`) plus a notebook; slides and notebook parts alternate. Each notebook
opens with its own timetable.

## Class 1 (150 min): from a regression line to a neural network
`class1_basics.ipynb`, fish-toxicity QSAR

- Look at the data: histograms, correlations, PCA map
- Regression as a box, one neuron, gradient descent, width scan
- Overfitting and the polynomial playground
- Featurization controls
- Baseline and linear fits, cross-validation, kNN and random forest, learning curves
- Permutation importance

## Class 2 (90 min): neural networks, and classifying enzymes
`class2_neural_networks_enzymes.ipynb`

- A neural network (MLP) on the toxicity data
- Enzyme EC classes from amino-acid composition, three classifiers
- The homology trap

## Class 3 (150 min): a recurrent network reads protein N-termini
`class3_rnn_targeting.ipynb`

- GRU on targeting signals (signal peptide / mitochondrial transit peptide / other)
- Recurrent-network playground

## Teaching notes

`Class N · Teaching notes.pptx`, built by `make_decks.py`, explain the ideas behind each notebook part.
The `.pptx` files are not in git; `make_decks.py` builds only these notes decks, not the lecture decks.

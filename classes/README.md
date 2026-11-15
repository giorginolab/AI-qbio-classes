# Three classes: 150, 90 and 150 minutes

Each class has its own folder with the lecture deck (`<topic>_slides.pptx`), the teaching notes
(`<topic>_teaching_notes.pptx`) and the notebook (`<topic>.ipynb`), where the topic is the folder name without its
number; slides and notebook parts alternate. Each notebook opens with its own timetable. The notebooks find `data/` by walking up from their folder.

## Class 1 (150 min): from a regression line to a neural network
`1_basics/`, fish-toxicity QSAR

- Look at the data: histograms, correlations, PCA map
- Regression as a box, one neuron, gradient descent, width scan
- Overfitting and the polynomial playground
- Featurization controls
- Baseline and linear fits, cross-validation, kNN and random forest, learning curves
- Permutation importance

## Class 2 (90 min): neural networks, and classifying enzymes
`2_neural_networks_enzymes/`

- A neural network (MLP) on the toxicity data
- Enzyme EC classes from amino-acid composition, three classifiers
- The homology trap

## Class 3 (150 min): a recurrent network reads protein N-termini
`3_rnn_targeting/`

- GRU on targeting signals (signal peptide / mitochondrial transit peptide / other)
- Recurrent-network playground


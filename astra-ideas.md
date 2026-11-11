# Proposals for AI in Quantitative Biology

These proposals are based on a review of the three practicals in `course/` and their dataset preparation scripts. The target audience is a master's in quantitative biology.

**The strongest improvement would be to make experimental reasoning the organising principle:** students should leave able to decide what a model's success means biologically, and design a control that could overturn their interpretation.

The existing material is a good basis: real datasets, CPU-friendly models, meaningful baselines, family-held-out evaluation, and a shuffled-sequence experiment. Those should be preserved.

## 1. Reduce the compulsory material and give each session one scientific question

The current notebooks are ambitious for 90 minutes, particularly session 2, which combines several overfitting demonstrations with a second biological problem. Session 3 adds hidden-state trajectories and saliency on top of learning PyTorch and GRUs.

Suggested organisation of the compulsory work:

| Session | Central question | Essential student result | Move to optional material |
|---|---|---|---|
| Toxicity | How much predictive information do these descriptors contain? | A baseline comparison, an error estimate, and a biological interpretation | Some PCA detail and the full gradient-descent exploration |
| Enzymes | Does the model generalise beyond familiar protein families? | Random versus family-held-out results, with a justified claim | Most architecture sweeps; retain one overfitting demonstration |
| Targeting | Which aspects of a sequence carry targeting information? | Comparison of representations plus one controlled perturbation | Hidden-state PCA and the full saliency analysis |

For a quantitative biology master's, interpreting one well-designed experiment is more valuable than executing several additional models.

## 2. Make students predict outcomes before running code

The notebooks already include discussion questions, but solutions immediately follow them. Use a repeated sequence:

1. Individually predict the outcome and give a biological reason.
2. Compare predictions with a neighbour.
3. Run the experiment.
4. Explain any discrepancy and propose one further control.

For example: “Will shuffling within the first ten residues have the same effect as shuffling the entire N-terminus?” Students must specify what each manipulation preserves.

Keep solutions collapsed or release them after discussion. Gradually reduce scaffolding: a worked example in session 1, incomplete experimental code in session 2, and a choice of controls in session 3.

Active-learning evidence supports this general direction, although the large Freeman et al. meta-analysis concerns undergraduate STEM rather than this particular master's setting. [Freeman et al., 2014](https://pubmed.ncbi.nlm.nih.gov/24821756/)

## 3. Make evaluation practice consistent with the course's own message

There is a particularly useful teaching opportunity here:

- Session 3 chooses the composition window by looking at test accuracy, then repeatedly inspects test results in the GRU playground.
- Session 2 correctly explains that early stopping requires validation data, but its demonstrations still centre on test-error minima.

Introduce **training / validation / final test** explicitly, and use validation curves throughout the interactive exercises. Students can select the window, regularisation and stopping epoch, then evaluate once on a reserved test set.

Also sharpen the question behind each split: random splitting estimates performance on another sample from the represented population; family-held-out splitting targets unfamiliar families. The latter is appropriate for that claim, rather than universally the “honest” split.

A short assessment could follow DOME's categories—data, optimisation, model and evaluation—with students writing one sentence under each. [Walsh et al., 2021](https://doi.org/10.1038/s41592-021-01205-4)

## 4. Turn session 3 into a small investigation of “sequence grammar”

This would be the first choice for greater originality. The current full-shuffling experiment is strong, but it destroys several things simultaneously: absolute position, local motifs and longer-range organisation. It supports the importance of sequence organisation; it does not establish which organisation matters or that recurrence is necessary.

Suggested comparisons:

| Representation or control | Question |
|---|---|
| Whole-window amino-acid composition | Is overall composition sufficient? |
| Composition in five consecutive ten-residue blocks | Is coarse position sufficient? |
| Logistic regression on flattened one-hot inputs | Are position-specific residue preferences sufficient? |
| GRU | Does learning interactions improve prediction? |
| Shuffle within blocks versus reorder intact blocks | Does local order or placement matter more? |

Students could choose one comparison rather than run everything. A simple baseline matching the GRU would be a valuable finding.

Use one figure from the TargetP 2.0 paper as the starting point: ask students to translate a biological interpretation into a testable prediction for their smaller model. Its published data also include similarity-based partitions, offering a reference for improving the practical's evaluation. [Almagro Armenteros et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/6769257/), [TargetP data and partitions](https://services.healthtech.dtu.dk/services/TargetP-2.0/3-Data.php)

## 5. Add a short “what else could explain the score?” investigation

The targeting dataset offers a concrete example. In the committed data, predicting each organism's most frequent class gives **40.9% accuracy in-sample**, against 33.3% for the balanced three-class task. This is a descriptive shortcut check, not a validated classifier result. It shows that organism and label are associated.

Students could:

- Evaluate an organism-only baseline properly.
- Repeat a comparison within eukaryotes.
- Compare random and species-held-out evaluation.
- Discuss why holding out a species does not necessarily remove homologous proteins from training.

The dataset builder removes identical 50-residue sequences, but does not establish homology-separated partitions. It also selects particularly clean cytoplasmic/nuclear negatives. That makes a useful teaching dataset, while limiting claims about arbitrary proteins.

This would connect the enzyme and targeting sessions through the same scientific problem: **what population does the reported score describe?**

## 6. Replace one interpretability plot with an experiment that challenges the interpretation

The “watching the RNN read” visualisation is engaging, but the classifier is trained on the final hidden state. Applying its output layer to earlier states does not automatically give calibrated predictions for truncated proteins.

Likewise, gradient × input is a local sensitivity measure, rather than a direct measurement of a discrete amino-acid substitution.

An effective exercise would be:

> Pick a residue highlighted by saliency. Predict the effect of three substitutions, measure their effects on the model, and compare with substitutions at low-saliency positions.

A stronger optional extension compares saliency before and after randomising model weights. Adebayo et al. provide the methodological motivation for checking whether an explanation actually depends on learned parameters. [Adebayo et al., 2018](https://papers.nips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html)

The final question should be: “What experiment on a real protein would test this model-derived hypothesis?” That brings the practical back to biology.

## 7. Add one modern extension using precomputed protein embeddings

For homework or an optional fourth practical, provide frozen protein-language-model embeddings for the enzyme dataset. Students fit the same simple classifier to composition and embeddings, using identical family-held-out partitions.

This isolates the contribution of the representation and avoids GPU requirements. Ask whether the benefit survives family holdout, and distinguish supervised label leakage from possible exposure to sequences during pretraining. Rives et al. provide a suitable foundational reading on learning biological information from unlabelled protein sequences. [Rives et al., 2021](https://doi.org/10.1073/pnas.2016239118)

## Assessment and priorities

Assess the course through a **one-page scientific report**: one biological question, one justified split, one baseline, one control, one figure and one limitation. Award marks for the validity of the inference, including well-supported negative results.

Priorities would be the workload reduction, consistent validation practice, and the sequence-grammar investigation; the embeddings exercise can remain optional.

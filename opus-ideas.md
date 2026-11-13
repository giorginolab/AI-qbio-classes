# Suggestions for the three-session course (`course/`)

Review of `t01_toxicity_qsar`, `t02_neural_networks_enzymes`, `t03_rnn_protein_targeting`
against a master's-in-quantitative-biology audience. Proposals only — no notebook was edited.

Written 2026-11-11. Pruned 2026-11-13: items already in the notebooks were removed (the
protein-language-model session, now `extra03`; the Cassotti, applicability-domain and
SignalP/TargetP references).

## Overall assessment

The three notebooks are already strong for this audience: real data, honest baselines, a
deliberate negative result (enzyme composition is weak), the homology-split trap, and a
shuffled-residue control. The main gaps are (a) a few places where the biology is thinner
than the ML, (b) statistical practice that stops just short of what a student will need in
their thesis, and (c) a representation thread that is implicit and would be worth making
explicit.

---

## 1. Measured: is there homology leakage in t03? — No

I suspected the t03 random split had the same leakage that Part B of t02 teaches about
(near-identical Ig/TCR N-termini, one proteome dominating each class). **I measured it and
it is not there.** GRU (`hidden=32`, 15 epochs), 5-fold, same architecture as the notebook:

| split | GRU accuracy |
|---|---|
| random (the notebook's) | **0.946** ± 0.004 |
| clusters ≥ 50 % identity held out | 0.944 ± 0.016 |
| clusters ≥ 35 % identity held out | 0.948 ± 0.009 |
| name-families held out (first 2 words of protein name) | 0.940 ± 0.006 |

At most −0.006, inside fold-to-fold noise. The composition+LR baseline is likewise flat
(0.872 → 0.872–0.875).

**Why.** Clustering the 50-mers by ungapped identity barely groups anything: at 50 % identity
only 190 of 3600 sequences sit in a cluster larger than one, and the largest cluster has 11
members. `make_targeting_dataset.py` already drops identical 50-mers, and N-terminal signals
have no conserved motif — the premise of the whole session — so homologues do *not* have
near-identical N-termini. The Ig/TCR block is only 53 sequences, all SP: too small to matter.

**The lesson this buys.** Leakage inflates a score only when the label is predictable from
family identity. In t02 the EC class essentially *is* a family property, so family leakage is
worth ≈ 0.09. In t03 the targeting class is a physico-chemical grammar that unrelated proteins
share and close homologues can differ in, so knowing a relative tells the model almost nothing.

**Proposal.** Do not change the dataset and do not present grouped splitting in t03 as a fix.
Run it as a **negative control**, in the style of the course's existing deliberate nulls:
students apply session 2's lesson, measure, find nothing, and are then asked *why* a precaution
that mattered enormously last week does not matter here. That is a stronger exercise than a
positive result would have been.

**The species effect is real, if mild.** Leave-one-organism-out, trained on the other four:

| held-out organism | n | accuracy |
|---|---|---|
| *B. subtilis* | 110 | 0.955 |
| *Arabidopsis thaliana* | 1263 | 0.949 |
| *E. coli* | 162 | 0.944 |
| *S. cerevisiae* | 539 | 0.924 |
| *Homo sapiens* | 1526 | **0.907** |

Human is the only meaningful drop (≈ −0.04 vs the random split), consistent with the 0.89 the
README quotes for non-human → human. This is an argument for promoting the cross-species check
out of "if you have time" into a required ~10-minute part: it is the control that actually shows
something, and it rules out the "the model learned proteome composition" objection that the
organism imbalance (mTP mostly yeast/*Arabidopsis*, "other" mostly human) otherwise leaves open.

### A genuine one in t01

217 of the 908 fish rows share a descriptor vector with another row. This is already
acknowledged obliquely (the *k* = 1 training-error puzzle) and the older `qsar_helpers.py`
even had `molecule_groups` for it, but in `course/` it has become an answer rather than an
exercise. Consider asking students to find the duplicates and re-run the CV with `GroupKFold`:
two lines, and it previews t02's family split in session 1.

---

## 2. Didactics

**Make uncertainty a habit.** t01 does this beautifully once (the 200-seed split distribution)
and then drops it — every later number in all three notebooks is a point estimate. The single
most transferable skill here is "never report a score without a spread". Suggest: repeated
stratified CV reported as mean ± sd throughout, and in t02 a *paired* comparison across the same
folds, asking "is RF better than the MLP, or is the difference inside the fold-to-fold noise?"
The answer is no, and that is worth having felt.

**Add calibration to t02/t03.** Quantitative biologists use classifier outputs downstream as
probabilities. A reliability diagram for the GRU — already shown to become over-confident via
the test loss — costs ten lines and makes the loss-vs-accuracy discussion land much harder.

**Class-imbalance realism.** Everything is balanced by construction (~350/class, 1200/class);
real targeting and EC prediction are not. A short appendix subsampling mTP to 5 %, showing
accuracy stay high while recall collapses, motivates MCC / balanced accuracy and heads off the
mistake students most often make in their own projects.

**Predict before running.** Several cells would be sharper if students wrote a guess first
(will the MLP beat RF? will family-splitting cost RF or logistic regression more?). The
inline-solution format is otherwise a risk: with answers directly below, many will read rather
than attempt. Collapsible `<details>` blocks render folded in Jupyter and cost nothing.

**Time budget.** t02 looks overfull: Part A (MLP, size sweep, epoch-by-epoch, playground) plus
Part B (composition, PCA, three classifiers, confusion matrix, group CV, relatives analysis) in
90 minutes is ambitious. Move the size sweep or the playground to homework and protect Part B,
which carries the unique message.

---

## 3. Originality

The QSAR → MLP → RNN arc is sound and common. Two directions would make it more distinctive.

**(a) Let students compute the descriptors in t01.** The six UCI descriptors are a black box,
and `NdsCH`/`NdssC` mean nothing to a biologist. With RDKit (pip-installable, CPU, seconds)
students can compute logP, TPSA and a Morgan fingerprint from SMILES themselves. That turns
"features are given" into "features are a modelling choice", and allows a 6-descriptor model
versus a 2048-bit fingerprint on the same 908 molecules — the representation axis the whole
course is secretly about.

**(b) Name the representation thread out loud.** Session 1 = fixed engineered descriptors;
session 2 = bag-of-residues (order discarded); session 3 = order-aware encoding; `extra03` =
learned pretrained representation. Each step is a strictly richer representation of the same
kind of object, and the accuracy gains track it. A one-slide spine at the start of each notebook
would make three practicals feel like one course — and that framing is what distinguishes a
quantitative-biology course from a generic ML course: the lesson is how you encode biology, not
which estimator wins.

---

## 4. Literature worth citing

Short and assignable. Checked against Crossref (DOIs) and JMLR on 2026-11-13.

- **Walsh I, Fishman D, Garcia-Gasulla D, et al. (2021).** DOME: recommendations for supervised
  machine learning validation in biology. *Nat Methods* 18:1122–1127.
  doi:[10.1038/s41592-021-01205-4](https://doi.org/10.1038/s41592-021-01205-4)
  — a checklist for reporting ML in biology. Ideal as a homework frame: fill the DOME items for
  your own notebook.
- **Kapoor S, Narayanan A (2023).** Leakage and the reproducibility crisis in machine-learning-based
  science. *Patterns* 4:100804.
  doi:[10.1016/j.patter.2023.100804](https://doi.org/10.1016/j.patter.2023.100804)
  — canonical reference for t02 Part B; gives a taxonomy and the vocabulary.
- **Lin Z, Akin H, Rao R, et al. (2023).** Evolutionary-scale prediction of atomic-level protein
  structure with a language model. *Science* 379:1123–1130.
  doi:[10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)
  — ESM-2, the model used in `extra03`, which does not cite it yet.
- **Yu T, Cui H, Li JC, et al. (2023).** Enzyme function prediction using contrastive learning.
  *Science* 379:1358–1363. doi:[10.1126/science.adf2465](https://doi.org/10.1126/science.adf2465)
  — CLEAN, a recent EC-number predictor: context for why t02's negative result is a *correct*
  negative (composition alone genuinely cannot do this).
- For the statistics habit, on comparing classifiers:
  - **Dietterich TG (1998).** Approximate statistical tests for comparing supervised
    classification learning algorithms. *Neural Comput* 10:1895–1923.
    doi:[10.1162/089976698300017197](https://doi.org/10.1162/089976698300017197)
  - **Demšar J (2006).** Statistical comparisons of classifiers over multiple data sets.
    *J Mach Learn Res* 7:1–30. No DOI; <https://jmlr.org/papers/v7/demsar06a.html>

---

## 5. Smaller points

- t01's gradient-descent loss landscape is excellent. A standardised-input version side by side
  (the answer already describes it) would turn a verbal claim into a figure.
- t03's saliency section honestly flags the position-50 artefact. Letting students fix it with
  `bidirectional=True` is a one-argument change with a visible payoff.
- The "deliberate null" convention from the old `CLAUDE.md` is a real asset; `course/` has fewer.
  One per session, explicitly labelled "we expected an effect and there is none", keeps it —
  and §1 above is a ready-made one for t03.
- Consider a shared `qbio_helpers.py`: the `find_data_dir`/loader block is copy-pasted three
  times, and students who edit one copy get confused.

---

## Reproducing the §1 numbers

Identity clustering via position-wise ungapped identity on the fixed-length 50-mers →
single-linkage connected components → `StratifiedGroupKFold`, against a `StratifiedKFold`
baseline, with the notebook's own `TargetingRNN`; plus a leave-one-organism-out loop.
The two throwaway scripts (~60 lines each) were job-scoped and are not committed — ask if they
should be recovered or turned into notebook cells.

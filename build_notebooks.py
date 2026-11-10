"""Build the student and solution notebooks for the 8-hour ML practical.

Each session is a list of cells. A code cell may carry a `solution=` variant:
the student notebook gets the scaffold, the solutions notebook gets the
worked code. Single source, so the two can never drift apart.

    python build_notebooks.py          # writes notebooks/*.ipynb
"""

from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "notebooks"

# --------------------------------------------------------------------------- #
# cell helpers
# --------------------------------------------------------------------------- #


def md(text):
    return ("md", text.strip("\n"), None)


def code(student, solution=None):
    return ("code", student.strip("\n"), solution.strip("\n") if solution else None)


HEADER = """
import sys, warnings
sys.path.insert(0, "..")          # so that `import qsar_helpers` works
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qsar_helpers import (load_fish, load_aquatic, xy, fold_error,
                          molecule_groups, score_table,
                          DESCRIPTOR_NOTES, TARGET_LABEL)

RNG = 0                            # one seed for the whole notebook
plt.rcParams.update({"figure.figsize": (6, 4), "figure.dpi": 110,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})
print("pandas", pd.__version__)
import sklearn; print("scikit-learn", sklearn.__version__)
"""


def build(cells, stem, title):
    for solutions in (False, True):
        nb = nbf.v4.new_notebook()
        nb.cells.append(
            nbf.v4.new_markdown_cell(
                f"# {title}\n\n"
                + ("**Solutions.** " if solutions else "**Student notebook.** ")
                + (
                    "Worked code and the numbers you should expect."
                    if solutions
                    else "Cells marked **EXERCISE** are for you to fill in. "
                    "Everything else can be run as it stands."
                )
            )
        )
        for kind, student, solution in cells:
            if kind == "md":
                nb.cells.append(nbf.v4.new_markdown_cell(student))
            else:
                src = solution if (solutions and solution) else student
                nb.cells.append(nbf.v4.new_code_cell(src))
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
        nb.metadata["language_info"] = {"name": "python"}
        name = f"{stem}_solutions.ipynb" if solutions else f"{stem}.ipynb"
        OUT.mkdir(parents=True, exist_ok=True)
        nbf.write(nb, OUT / name)
        print("wrote", name, f"({len(nb.cells)} cells)")


# --------------------------------------------------------------------------- #
# SESSION 1
# --------------------------------------------------------------------------- #

S1 = [
    md("""
## What you will do today

You will build a model that predicts how toxic a small molecule is to a fish,
from numbers computed from its chemical structure alone. This is a *QSAR* model
(quantitative structure-activity relationship), and it is the kind of model that
regulators such as ECHA accept, under REACH, as partial evidence in place of an
animal test. So the question "how well does it really work?" is not academic.

Over the four sessions the modelling gets barely more sophisticated than
today's. What changes is how carefully we measure. By the end you should be
unable to look at a reported R<sup>2</sup> without asking three questions:
compared to what, averaged over what, and on which molecules.

**This session (2 h)**

1. Understand the data and the target, in units a toxicologist would recognise
2. Establish the baseline that any model must beat
3. Fit the first model, and see why a single train/test split is not an answer
4. Replace it with cross-validation and get the first number worth quoting
"""),
    code(HEADER),
    md("""
## 1. The data

`qsar_fish_toxicity.csv`: 908 molecules, six molecular descriptors, and the
measured 96-hour LC50 toward *Pimephales promelas* (the fathead minnow) —
the concentration that kills half the exposed fish.

The target is reported as **-log10(mol/L)**. Watch the sign: a *high* value
means a *low* lethal concentration, i.e. a **more toxic** molecule. One unit on
this scale is a factor of ten in concentration.
"""),
    code("""
fish = load_fish()
X, y = xy(fish)
print(fish.shape)
fish.head()
"""),
    code("""
for name in X.columns:
    print(f"{name:>10s}  {DESCRIPTOR_NOTES[name]}")
"""),
    md("""
Three of these describe the molecule's shape and electron distribution
(`CIC0`, `SM1_Dz(Z)`, `GATS1i`), two are counts of specific carbon fragments
(`NdsCH`, `NdssC`), and `MLOGP` is lipophilicity — how strongly the compound
prefers octanol over water.

### EXERCISE 1.1 — look at the target before you model it

Plot the distribution of `LC50` and report its mean, standard deviation, min
and max. Then answer in a sentence: across this dataset, what is the ratio
between the lethal concentration of the most and the least toxic molecule?
"""),
    code(
        """
# TODO: histogram of y, plus mean / sd / min / max
# TODO: convert the min-max range in log units into a concentration ratio
...
""",
        """
fig, ax = plt.subplots()
ax.hist(y, bins=30, color="#4C72B0", edgecolor="white")
ax.set_xlabel(TARGET_LABEL); ax.set_ylabel("molecules")
ax.set_title("Target distribution, fathead minnow LC50")
fig.tight_layout()

print(f"mean {y.mean():.3f}   sd {y.std():.3f}   min {y.min():.3f}   max {y.max():.3f}")
span = y.max() - y.min()
print(f"range = {span:.2f} log units  ->  a factor of {fold_error(span):,.0f} in concentration")
""",
    ),
    md("""
The target is close to symmetric (skew 0.25) and spans about 9.6 log units:
the least toxic compound needs roughly four billion times the concentration of
the most toxic one. That spread is why R<sup>2</sup> on this dataset can look
flattering — there is a lot of variance available to explain.

### EXERCISE 1.2 — which descriptors carry signal?

Make one scatter plot per descriptor against `LC50`, and compute the Pearson
correlation of each descriptor with the target. Which descriptor is the
strongest single predictor?
"""),
    code(
        """
# TODO: 2 x 3 grid of scatter plots, descriptor on x, LC50 on y
# TODO: print correlations with the target, sorted by absolute value
...
""",
        """
fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharey=True)
for ax, name in zip(axes.ravel(), X.columns):
    ax.scatter(X[name], y, s=6, alpha=0.4, color="#4C72B0")
    ax.set_xlabel(name)
for ax in axes[:, 0]:
    ax.set_ylabel(TARGET_LABEL)
fig.suptitle("Each descriptor against the target")
fig.tight_layout()

corr = fish.corr(numeric_only=True)["LC50"].drop("LC50")
print(corr.reindex(corr.abs().sort_values(ascending=False).index).round(3).to_string())
""",
    ),
    md("""
`MLOGP` leads with r = 0.65. That is not a coincidence of this dataset: for most
industrial organic chemicals, acute fish toxicity is dominated by **non-polar
narcosis** — the compound partitions into cell membranes and disrupts them, and
how much partitions in is largely a function of lipophilicity. A one-descriptor
model built on log P is the classical "baseline toxicity" line in
ecotoxicology, and it is the thing our six-descriptor model has to improve on.

Keep this in mind for session 4: a model whose predictions are driven by log P
is behaving the way the chemistry says it should. A model whose predictions are
driven by `NdssC` would need an explanation.

## 2. The baseline comes first

Before fitting anything, decide what "doing nothing" scores. The dumbest
possible predictor ignores the molecule and always predicts the mean LC50 of
the training set.
"""),
    code("""
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, cross_validate

cv5 = KFold(n_splits=5, shuffle=True, random_state=RNG)
dummy = cross_validate(DummyRegressor(strategy="mean"), X, y, cv=cv5,
                       scoring=("r2", "neg_root_mean_squared_error"))
print(f"dummy  R2   = {dummy['test_r2'].mean():+.3f}")
print(f"dummy  RMSE = {-dummy['test_neg_root_mean_squared_error'].mean():.3f} log units")
"""),
    md("""
### EXERCISE 1.3 — two things to explain

1. The dummy's R<sup>2</sup> comes out slightly **negative**, -0.016.
   R<sup>2</sup> is often described as "the fraction of variance explained",
   which cannot be negative. Why is it negative here?
2. Its RMSE is about 1.45 log units. What does that mean as a fold-error in
   concentration? Use `fold_error`.

*Hint for (1): which mean is used for the prediction, and which mean does the
R<sup>2</sup> formula compare against, when the score is computed on a held-out
fold?*
"""),
    code(
        """
# TODO: compute the dummy's fold-error and write your answer to (1) as a comment
...
""",
        """
rmse_dummy = -dummy["test_neg_root_mean_squared_error"].mean()
print(f"dummy RMSE {rmse_dummy:.3f} log units -> wrong by a factor of {fold_error(rmse_dummy):.0f}")

# (1) R2 on a held-out fold is 1 - SSE/SST where SST uses the *test* fold's own
#     mean, while the prediction is the *training* fold's mean. Those two means
#     differ by sampling noise, so the dummy is always very slightly worse than
#     the best possible constant for that fold, and R2 lands just below zero.
#     R2 = 0 is therefore the score of an *oracle* constant predictor; a
#     legitimately fitted dummy scores marginally less.
""",
    ),
    md("""
A factor of 29 in concentration, which is useless — as it should be. Every
R<sup>2</sup> you report from now on is a claim about distance from this line.

## 3. One model, two scores, and a trap

Now fit an actual model. Split the data 75/25, fit a linear regression on the
training part, and score it on both parts.
"""),
    code("""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
lin = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr)

print(f"train R2 = {r2_score(y_tr, lin.predict(X_tr)):.3f}")
print(f" test R2 = {r2_score(y_te, lin.predict(X_te)):.3f}")
"""),
    md("""
### EXERCISE 1.4 — the number that is not an estimate

Re-run the split-fit-score sequence for `random_state` = 0, 1, 2, ..., 9.
Collect the ten test-set R<sup>2</sup> values. Report the smallest and the
largest.

Then answer: if a colleague shows you a single test-set R<sup>2</sup> for a
model trained on 908 molecules, what have they actually told you?
"""),
    code(
        """
# TODO: loop over 10 seeds, collect test R2 for the linear model, report min/max
...
""",
        """
scores = []
for seed in range(10):
    a, b, c, d = train_test_split(X, y, test_size=0.25, random_state=seed)
    m = make_pipeline(StandardScaler(), LinearRegression()).fit(a, c)
    scores.append(r2_score(d, m.predict(b)))
scores = np.array(scores)
print(np.round(scores, 3))
print(f"min {scores.min():.3f}   max {scores.max():.3f}   "
      f"spread {np.ptp(scores):.3f}")

# They have told you the score of one model on one arbitrary quarter of the
# data. The spread across seeds here is of the same order as the difference
# between competing model families (session 3), so a single split cannot
# distinguish "better model" from "luckier split".
""",
    ),
    md("""
Across ten seeds the test R<sup>2</sup> runs from 0.442 to 0.618 — a range of
0.18, for the *same model on the same data*. Session 2 is about that spread.

## 4. Cross-validation: the first number worth quoting

Instead of one split, use five, so that every molecule is predicted exactly once
by a model that never saw it.
"""),
    code("""
lin_cv = cross_validate(make_pipeline(StandardScaler(), LinearRegression()),
                        X, y, cv=cv5,
                        scoring=("r2", "neg_root_mean_squared_error"))
r2 = lin_cv["test_r2"]
rmse = -lin_cv["test_neg_root_mean_squared_error"]
print(f"linear   R2   = {r2.mean():.3f} +- {r2.std():.3f}  (sd across 5 folds)")
print(f"linear   RMSE = {rmse.mean():.3f} log units -> factor {fold_error(rmse.mean()):.1f}")
print(f"per-fold R2: {np.round(r2, 3)}")
"""),
    md("""
### EXERCISE 1.5 — is this model fit for purpose?

The model is clearly far better than the dummy. But "better than nothing" is not
the standard a regulatory dossier is held to.

1. Convert the cross-validated RMSE into a fold-error in concentration.
2. Suppose a screening decision needs the predicted LC50 to be right within a
   factor of 3. What RMSE in log units does that correspond to? Does this model
   meet it?
3. Fit a model using `MLOGP` **alone** and cross-validate it. How much of the
   six-descriptor model's performance is just lipophilicity?
"""),
    code(
        """
# TODO: (1) fold error, (2) the RMSE target for a factor-of-3 tolerance,
#       (3) cross-validate a one-descriptor model on MLOGP only
...
""",
        """
print(f"(1) RMSE {rmse.mean():.3f} log units -> factor {fold_error(rmse.mean()):.1f}")
print(f"(2) a factor-of-3 tolerance is log10(3) = {np.log10(3):.3f} log units;"
      f" this model is {rmse.mean()/np.log10(3):.1f}x worse than that")

from sklearn.model_selection import cross_val_score
only = cross_val_score(make_pipeline(StandardScaler(), LinearRegression()),
                       X[["MLOGP"]], y, cv=cv5, scoring="r2")
print(f"(3) MLOGP alone: R2 = {only.mean():.3f}   "
      f"all six: R2 = {r2.mean():.3f}")
""",
    ),
    md("""
## Where we are

| predictor | CV R<sup>2</sup> | CV RMSE (log units) | fold-error |
|---|---|---|---|
| mean of the training set | -0.016 | 1.456 | 29x |
| `MLOGP` alone | 0.408 | | |
| all six descriptors, linear | 0.557 ± 0.043 | 0.960 | 9.1x |

Three things to carry into session 2.

- A score is meaningless without the baseline next to it.
- A score from one train/test split moved over a range of 0.18 in
  R<sup>2</sup> across ten seeds, for reasons that have nothing to do with the
  model.
- Our best model so far is wrong by a factor of nine in concentration. Keep
  that in view when the R<sup>2</sup> starts climbing.
"""),
]

# --------------------------------------------------------------------------- #
# SESSION 2
# --------------------------------------------------------------------------- #

S2 = [
    md("""
## Where the honest number comes from

Session 1 ended with two numbers for the same model: 0.56 from cross-validation,
and anything between about 0.50 and 0.61 depending on which split you happened
to take. Today we quantify that variability, and then look at four ways of
accidentally inflating the number — only two of which actually bite on this
dataset. Measuring which ones bite, rather than assuming, is the point.

**This session (2 h)**

1. The split lottery: how much is a single test-set score worth?
2. Pipelines, and why they are not a stylistic preference
3. Leakage you can measure: preprocessing, feature selection, duplicate molecules
4. A checklist you can apply to someone else's paper
"""),
    code(HEADER),
    code("""
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (KFold, GroupKFold, RepeatedKFold,
                                     train_test_split, cross_val_score,
                                     cross_val_predict)
from sklearn.metrics import r2_score

fish = load_fish()
X, y = xy(fish)
cv5 = KFold(n_splits=5, shuffle=True, random_state=RNG)
"""),
    md("""
## 1. The split lottery

### EXERCISE 2.1

Fit a random forest on a 75/25 split and score it on the held-out quarter.
Do this for 200 different values of `random_state` in the **split** (keep the
forest's own seed fixed, so that only the split changes). Plot the 200 scores as
a histogram and report mean, sd, min and max.

Then: find the seed that gives the highest score, and write down the sentence
you could put in an abstract if you only reported that one.

*Runtime is about a minute. Use `n_estimators=200`.*
"""),
    code(
        """
# TODO: loop over 200 split seeds, collect test R2, plot the histogram
...
""",
        """
scores = []
for seed in range(200):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed)
    rf = RandomForestRegressor(n_estimators=200, random_state=RNG).fit(X_tr, y_tr)
    scores.append(r2_score(y_te, rf.predict(X_te)))
scores = np.array(scores)

fig, ax = plt.subplots()
ax.hist(scores, bins=25, color="#4C72B0", edgecolor="white")
ax.axvline(scores.mean(), color="k", lw=1.5, label=f"mean {scores.mean():.3f}")
ax.axvline(scores.max(), color="#C44E52", lw=1.5, ls="--", label=f"max {scores.max():.3f}")
ax.set_xlabel("test-set $R^2$"); ax.set_ylabel("splits")
ax.set_title("Same model, same data, 200 different 75/25 splits")
ax.legend(); fig.tight_layout()

print(f"mean {scores.mean():.3f}  sd {scores.std():.3f}  "
      f"min {scores.min():.3f}  max {scores.max():.3f}")
print(f"5th-95th percentile: {np.quantile(scores, 0.05):.3f} - {np.quantile(scores, 0.95):.3f}")
print(f"best seed: {scores.argmax()}")

# "A random forest on six molecular descriptors predicted acute fish toxicity
#  with R2 = 0.73 on a held-out test set." - true, reproducible, and worthless.
""",
    ),
    md("""
The honest centre is about 0.61, and the range runs from roughly 0.46 to 0.73.
Nobody needs to fabricate anything to publish the 0.73: they just need to try a
few splits and stop looking when the number is pleasing. This is the single most
common way that published performance figures end up irreproducible.

The fix is not a better seed. It is to report an estimator whose own variability
you have measured — which is what repeated cross-validation gives you.

### EXERCISE 2.2 — how many repeats are enough?

`RepeatedKFold(n_splits=5, n_repeats=r)` averages r independent 5-fold runs.
Compute the mean R<sup>2</sup> of the random forest for r = 1, 2, 3, 5, 10 and
plot it against r. At what point does the estimate stop moving by more than
0.01?
"""),
    code(
        """
# TODO: mean CV R2 for increasing numbers of repeats; plot mean vs n_repeats
...
""",
        """
rows = []
for r in (1, 2, 3, 5, 10):
    cv = RepeatedKFold(n_splits=5, n_repeats=r, random_state=RNG)
    s = cross_val_score(RandomForestRegressor(n_estimators=200, random_state=RNG),
                        X, y, cv=cv, scoring="r2")
    rows.append((r, s.mean(), s.std()))
    print(f"n_repeats={r:>2d}  R2 = {s.mean():.3f} +- {s.std():.3f}  ({len(s)} folds)")

rows = np.array(rows)
fig, ax = plt.subplots()
ax.errorbar(rows[:, 0], rows[:, 1], yerr=rows[:, 2], marker="o", capsize=3, color="#4C72B0")
ax.set_xlabel("n_repeats"); ax.set_ylabel("mean CV $R^2$")
ax.set_title("The estimate settles; the fold-to-fold sd does not shrink")
fig.tight_layout()

# The mean is stable to within ~0.01 from about 3 repeats (0.603, 0.602,
# 0.605, 0.609, 0.613). The sd behaves differently: it *rises* from 0.027 at
# one repeat to about 0.05 and then settles. With only 5 folds the sd is
# itself estimated from 5 numbers and is badly underestimated; more repeats
# give a better estimate of a fold-to-fold variability that is a property of
# the data and does not shrink however long you compute.
""",
    ),
    md("""
Two distinct quantities get confused here. The **mean** is your performance
estimate, and more repeats make it more precise. The **sd across folds** is a
property of the data and model, and it does not shrink with more repeats — it is
what tells you whether a 0.02 difference between two models means anything.

## 2. Pipelines

A `Pipeline` bundles preprocessing with the model so that `fit` is only ever
called on training data. Before treating that as bureaucracy, check whether
scaling matters at all.

### EXERCISE 2.3

For k-nearest-neighbours (k=5), ridge regression (alpha=10) and a random forest,
compare 5-fold CV R<sup>2</sup> with and without `StandardScaler`. Explain the
pattern you see from how each algorithm works.
"""),
    code(
        """
# TODO: for each of kNN / Ridge / RF, CV score raw vs wrapped in StandardScaler
...
""",
        """
for name, est in [("kNN(k=5)", KNeighborsRegressor(n_neighbors=5)),
                  ("Ridge(10)", Ridge(alpha=10)),
                  ("RandomForest", RandomForestRegressor(n_estimators=200, random_state=RNG))]:
    raw = cross_val_score(est, X, y, cv=cv5, scoring="r2").mean()
    scaled = cross_val_score(make_pipeline(StandardScaler(), est), X, y, cv=cv5, scoring="r2").mean()
    print(f"{name:>13s}  unscaled {raw:.3f}   scaled {scaled:.3f}   diff {scaled-raw:+.3f}")

# kNN improves by ~0.045: it measures Euclidean distance, so MLOGP (sd 1.43)
# would otherwise dominate NdsCH (sd 0.61) purely through its units.
# Ridge is unchanged in R2 (the fit is equivariant up to the penalty, which is
# weak here). Trees split one feature at a time on thresholds, so any monotone
# rescaling leaves them identical.
""",
    ),
    md("""
So scaling is a modelling decision for distance-based methods and a no-op for
trees. The reason to put it in a `Pipeline` is different, and it is the subject
of the rest of the session.

## 3. Leakage, measured

"Leakage" means information from the evaluation data reaching the model. We will
look at three mechanisms, and **measure** each one on this dataset rather than
assuming it matters.

### 3a. Scaling fitted on everything

The natural-looking mistake: scale the whole matrix, then cross-validate. Each
fold's model has then seen the mean and sd of the test fold.

### EXERCISE 2.4

Measure the size of this effect for kNN on this dataset: scale `X` once up
front and cross-validate, versus scaling inside a pipeline. Report the gap.
"""),
    code(
        """
# TODO: scale-then-CV vs pipeline, for kNN(k=5)
...
""",
        """
X_pre = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
wrong = cross_val_score(KNeighborsRegressor(5), X_pre, y, cv=cv5, scoring="r2").mean()
right = cross_val_score(make_pipeline(StandardScaler(), KNeighborsRegressor(5)),
                        X, y, cv=cv5, scoring="r2").mean()
print(f"scaled before CV : {wrong:.4f}")
print(f"scaled inside CV : {right:.4f}")
print(f"optimism         : {wrong-right:+.4f}")
""",
    ),
    md("""
About +0.004 — nothing, and far below the ±0.05 fold-to-fold noise. A mean and a
standard deviation computed from 908 molecules barely change when you remove
180 of them.

This is worth dwelling on, because the lesson is not the one students usually
take away. The correct conclusion is **not** "scaler leakage is harmless, skip
the pipeline". It is:

- the *magnitude* of a leak depends on how much the leaked statistic depends on
  the individual rows — which is why the same mistake is catastrophic with 30
  samples and invisible with 900;
- you cannot know the magnitude without measuring, and you cannot measure it in
  someone else's paper;
- so the practice has to be defended by argument, not by effect size.

The next mechanism is the same mistake applied to a statistic that *does* depend
strongly on individual rows.

### 3b. Feature selection fitted on everything

Real QSAR descriptor sets have thousands of columns, not six, and are routinely
filtered down before modelling. To put ourselves in that regime, we will take a
small subset of molecules and give ourselves a large matrix of **pure noise**
descriptors — columns generated from a random number generator, which by
construction contain no information about toxicity whatsoever.

Any performance we obtain from these columns is, definitionally, an artefact.

### EXERCISE 2.5

With 60 molecules and 2000 noise descriptors:

1. Select the 10 noise columns most correlated with `y` using `SelectKBest` on
   the **whole** matrix, then cross-validate a ridge regression on those 10.
2. Do it properly: put `SelectKBest` and `Ridge` in a `Pipeline` and
   cross-validate that, so selection happens inside each fold.

Report both R<sup>2</sup> values. Remember that the honest answer is "no better
than the dummy".
"""),
    code(
        """
rng = np.random.default_rng(RNG)
idx = rng.choice(len(X), 60, replace=False)
y_sub = y[idx]
noise = pd.DataFrame(rng.normal(size=(60, 2000)),
                     columns=[f"noise_{i:04d}" for i in range(2000)])
print(noise.shape, "pure noise descriptors for", len(y_sub), "molecules")

# TODO: (1) SelectKBest(k=10) on all of `noise`, then CV a Ridge on the selection
# TODO: (2) the same selection inside a Pipeline, cross-validated
...
""",
        """
rng = np.random.default_rng(RNG)
idx = rng.choice(len(X), 60, replace=False)
y_sub = y[idx]
noise = pd.DataFrame(rng.normal(size=(60, 2000)),
                     columns=[f"noise_{i:04d}" for i in range(2000)])
print(noise.shape, "pure noise descriptors for", len(y_sub), "molecules")

from sklearn.feature_selection import SelectKBest, f_regression
cv_sub = KFold(5, shuffle=True, random_state=RNG)

# (1) the mistake: choose the columns using every molecule, then "validate"
sel = SelectKBest(f_regression, k=10).fit(noise, y_sub)
chosen = noise.loc[:, sel.get_support()]
wrong = cross_val_score(make_pipeline(StandardScaler(), Ridge(1.0)),
                        chosen, y_sub, cv=cv_sub, scoring="r2").mean()

# (2) selection is part of the model, so it belongs inside the fold
pipe = Pipeline([("select", SelectKBest(f_regression, k=10)),
                 ("scale", StandardScaler()),
                 ("model", Ridge(1.0))])
right = cross_val_score(pipe, noise, y_sub, cv=cv_sub, scoring="r2").mean()

print(f"select on all data, then CV : R2 = {wrong:+.3f}")
print(f"select inside each fold     : R2 = {right:+.3f}")
""",
    ),
    md("""
Selecting first gives R<sup>2</sup> ≈ **+0.39 from random numbers**. Done
correctly the same procedure scores about **-0.18**, i.e. worse than predicting
the mean, which is the truth.

The mechanism: with 2000 candidates and 60 molecules, some columns correlate
with `y` by chance alone, and the strongest of them do so *on these particular
60 molecules*. Choosing them using all 60 molecules, including the ones you then
treat as held out, hands the model a fingerprint of the test rows. Nothing about
the subsequent cross-validation can undo that — the fold split happens after the
information has already moved.

This is not a toy failure mode. It is the standard explanation for biomarker
panels that do not replicate: gene filtering, probe selection, or
differential-expression screening done once on the full cohort, followed by
cross-validation of the classifier only.

### EXERCISE 2.6 — vary the regime

Repeat 2.5 for a few combinations of sample size and descriptor count:
(n=60, p=500), (n=60, p=2000), (n=120, p=2000), (n=300, p=2000). The artefact
is itself a random quantity, so **average five independent draws** of molecules
and noise per combination and report the mean. How does the size of the artefact
depend on n, and on p?
"""),
    code(
        """
# TODO: loop over (n, p) combinations, 5 draws each, tabulate wrong vs right
...
""",
        """
from sklearn.feature_selection import SelectKBest, f_regression

def noise_experiment(n_sub, p, seed):
    rg = np.random.default_rng(seed)
    ii = rg.choice(len(X), n_sub, replace=False)
    ys = y[ii]
    nz = pd.DataFrame(rg.normal(size=(n_sub, p)), columns=[f"n{i}" for i in range(p)])
    cvk = KFold(5, shuffle=True, random_state=RNG)
    s = SelectKBest(f_regression, k=10).fit(nz, ys)
    w = cross_val_score(make_pipeline(StandardScaler(), Ridge(1.0)),
                        nz.loc[:, s.get_support()], ys, cv=cvk, scoring="r2").mean()
    r = cross_val_score(Pipeline([("s", SelectKBest(f_regression, k=10)),
                                  ("sc", StandardScaler()), ("m", Ridge(1.0))]),
                        nz, ys, cv=cvk, scoring="r2").mean()
    return w, r

rows = []
for n_sub, p in [(60, 500), (60, 2000), (120, 2000), (300, 2000)]:
    res = np.array([noise_experiment(n_sub, p, seed) for seed in range(5)])
    rows.append(dict(n=n_sub, p=p,
                     select_then_CV=round(res[:, 0].mean(), 3),
                     sd=round(res[:, 0].std(), 3),
                     select_inside_CV=round(res[:, 1].mean(), 3)))
print(pd.DataFrame(rows).to_string(index=False))

# The artefact grows with p (more candidate columns, so a stronger best-of-p
# chance correlation) and shrinks with n (a chance correlation over more
# molecules is less specific to the rows you are about to score). Note the sd
# across draws: a single draw is not enough to order these regimes, which is
# the same lesson as exercise 2.1 applied to the artefact itself.
""",
    ),
    md("""
### 3c. The same molecule on both sides of the split

A mechanism specific to biological and chemical data: **rows that are not
independent**. Here, different entries can carry identical descriptor values —
either genuine duplicates, or distinct molecules that these six descriptors
cannot tell apart. If one lands in training and its twin in test, the model is
being asked a question it has already been given the answer to.

### EXERCISE 2.7

1. Use `molecule_groups(X)` to label rows with identical descriptor vectors.
   How many rows have at least one twin?
2. Compare 5-fold `KFold` with 5-fold `GroupKFold` (passing those labels as
   `groups=`) for kNN with k=1, k=5 and a random forest. How large is the gap?
3. The aggregate gap will surprise you. Use `cross_val_predict` with k=1 under
   both schemes and compute R<sup>2</sup> **separately** on the twinned rows and
   on the rows with no twin. Now how large is it?
"""),
    code(
        """
groups = molecule_groups(X)
has_twin = pd.Series(groups).duplicated(keep=False).to_numpy()
print(f"{len(X)} rows, {len(set(groups))} distinct descriptor vectors, "
      f"{has_twin.sum()} rows with at least one twin")

# TODO: (2) KFold vs GroupKFold for kNN(1), kNN(5), RF
# TODO: (3) cross_val_predict with k=1 under both schemes; score the twinned
#           and the singleton rows separately
...
""",
        """
groups = molecule_groups(X)
has_twin = pd.Series(groups).duplicated(keep=False).to_numpy()
print(f"{len(X)} rows, {len(set(groups))} distinct descriptor vectors, "
      f"{has_twin.sum()} rows with at least one twin")

print("\\n(2) aggregate comparison")
for name, est in [("kNN(k=1)", make_pipeline(StandardScaler(), KNeighborsRegressor(1))),
                  ("kNN(k=5)", make_pipeline(StandardScaler(), KNeighborsRegressor(5))),
                  ("RandomForest", RandomForestRegressor(n_estimators=200, random_state=RNG))]:
    a = cross_val_score(est, X, y, cv=cv5, scoring="r2").mean()
    b = cross_val_score(est, X, y, cv=GroupKFold(5), groups=groups, scoring="r2").mean()
    print(f"  {name:>13s}  random {a:.3f}   grouped {b:.3f}   gap {a-b:+.3f}")

print("\\n(3) k=1, scored separately on the affected and unaffected rows")
k1 = make_pipeline(StandardScaler(), KNeighborsRegressor(1))
p_rand = cross_val_predict(k1, X, y, cv=cv5)
p_grp = cross_val_predict(k1, X, y, cv=GroupKFold(5), groups=groups)
for label, mask in [("rows with a twin", has_twin), ("rows without", ~has_twin)]:
    print(f"  {label:>17s} (n={mask.sum():3d}):  random {r2_score(y[mask], p_rand[mask]):.3f}"
          f"   grouped {r2_score(y[mask], p_grp[mask]):.3f}")
""",
    ),
    md("""
The aggregate gap is about +0.004 for k=1 — apparently nothing. But restricted to
the 217 rows that actually have a twin, k=1 scores **0.65 with a random split and
0.52 with a grouped split**. On the 691 rows with no twin there is no gap at all.

Both facts are true and they teach different things:

- the leak is real, large, and exactly where the mechanism predicts it;
- averaged over a dataset where 76% of rows are unaffected, and with a model that
  does not rely purely on memorising single neighbours, it vanishes into the
  noise.

So "we checked and it made no difference to the headline number" is not evidence
that a dependence structure is harmless. It may only mean the affected fraction
is small. When your replicates *are* the majority of your data — multiple cell
lines per patient, several images per animal, homologous sequences across a
protein family, repeat measurements of the same compound — the aggregate effect
is the 0.13, not the 0.004.

And note which direction the honest choice goes: `GroupKFold` here gives a
*lower* score. Correct methodology generally costs you performance on paper.

## Session 2 checklist

Things you can now check in any supervised-learning paper or thesis chapter.

1. Is a baseline reported? Dummy, or a single-descriptor model?
2. Is the headline score a single split, or an average with a spread?
3. Is every data-dependent step — scaling, imputation, feature selection,
   dimensionality reduction, outlier removal — inside the resampling loop?
4. Are the rows independent? If not, is the split grouped accordingly?
5. Would the authors have reported a different number if the first split had
   been unlucky? Can you tell?

### EXERCISE 2.8 — homework, 20 minutes

Take the leakage mechanism from 3b and write out, in five sentences, how the
equivalent mistake would look in a study you might actually run: 40 patient
samples, 20000 transcripts, a two-group outcome, and a feature-filtering step.
State precisely where in the pipeline the test-set information enters.
"""),
]

# --------------------------------------------------------------------------- #
# SESSION 3
# --------------------------------------------------------------------------- #

S3 = [
    md("""
## Choosing a model without fooling yourself

You now have a trustworthy way to score one model. Today you have to choose
between many, and tune each of them — which creates a new way to be wrong, one
that no amount of careful cross-validation inside a single model can fix.

**This session (2 h)**

1. A comparison table with error bars, and a paired comparison that respects folds
2. Hyperparameter search, and how much of a leaderboard is noise
3. Nested cross-validation: scoring the *procedure*, not the winner
4. Learning curves: should you measure more molecules, or better descriptors?
"""),
    code(HEADER),
    code("""
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (KFold, RepeatedKFold, GridSearchCV,
                                     cross_val_score, cross_validate, learning_curve)
from sklearn.base import clone
from sklearn.metrics import r2_score

fish = load_fish()
X, y = xy(fish)
cv_rep = RepeatedKFold(n_splits=5, n_repeats=5, random_state=RNG)   # 25 folds
"""),
    md("""
## 1. The comparison table

### EXERCISE 3.1

Cross-validate six models with `cv_rep` (5-fold, 5 repeats = 25 folds):
dummy, linear regression, ridge (alpha=1), kNN (k=5), a random forest
(500 trees) and `HistGradientBoostingRegressor`. Scale where it matters.

Produce a table of mean R<sup>2</sup> ± sd and mean RMSE, and a horizontal bar
chart with error bars. Then answer: which models are distinguishable from each
other, given the spreads?

*Runtime about two minutes.*
"""),
    code(
        """
models = {
    "Dummy (mean)": DummyRegressor(strategy="mean"),
    "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge (a=1)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "kNN (k=5)": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
    "RandomForest": RandomForestRegressor(n_estimators=500, random_state=RNG),
    "HistGB": HistGradientBoostingRegressor(random_state=RNG),
}

# TODO: cross-validate each model with cv_rep, keep the per-fold R2 arrays in a
#       dict called `folds`, build the table, and plot mean +- sd
...
""",
        """
models = {
    "Dummy (mean)": DummyRegressor(strategy="mean"),
    "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge (a=1)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "kNN (k=5)": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
    "RandomForest": RandomForestRegressor(n_estimators=500, random_state=RNG),
    "HistGB": HistGradientBoostingRegressor(random_state=RNG),
}

folds, rows = {}, []
for name, est in models.items():
    res = cross_validate(est, X, y, cv=cv_rep,
                         scoring=("r2", "neg_root_mean_squared_error"))
    folds[name] = res["test_r2"]
    rows.append(dict(model=name,
                     R2=res["test_r2"].mean(), R2_sd=res["test_r2"].std(),
                     RMSE=-res["test_neg_root_mean_squared_error"].mean()))
table = pd.DataFrame(rows).sort_values("R2", ascending=False)
print(table.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 3.2))
t = table.iloc[::-1]
ax.barh(t.model, t.R2, xerr=t.R2_sd, color="#4C72B0", capsize=3)
ax.set_xlabel("cross-validated $R^2$  (mean $\\\\pm$ sd over 25 folds)")
ax.set_title("Six models, same 25 folds")
fig.tight_layout()
""",
    ),
    md("""
Dummy at -0.01, the two linear models tied at 0.56, and the three non-linear
models bunched between 0.60 and 0.61 — with fold-to-fold spreads of 0.05 to
0.06, i.e. larger than the gaps between them.

Two readings of that table are both correct and worth stating explicitly. The
step from "no model" to "linear model" is large and unambiguous. The step from
"linear" to "random forest" is about 0.05 in R<sup>2</sup>, which is the same
size as the error bars — so the bar chart alone cannot establish it.

### EXERCISE 3.2 — compare models on the folds they share

The fold-to-fold sd includes variation caused by the *folds* (some quarters of
this dataset are simply harder), and that variation is common to all models.
Because every model saw the same 25 folds, you can subtract it out: compute the
**per-fold difference** between two models instead of comparing their means.

1. Compute `folds["RandomForest"] - folds["Ridge (a=1)"]` and report its mean
   and sd. In what fraction of the 25 folds does the forest win?
2. Plot the paired fold scores (ridge on x, forest on y, with the diagonal) so
   the pairing is visible.
3. A paired t-test on these 25 differences would give a very small p-value.
   Explain why that p-value should not be reported.
"""),
    code(
        """
# TODO: paired per-fold differences, win fraction, paired scatter plot
...
""",
        """
d = folds["RandomForest"] - folds["Ridge (a=1)"]
print(f"RF - Ridge, per fold: mean {d.mean():+.3f}   sd {d.std():.3f}")
print(f"forest wins in {100*(d>0).mean():.0f}% of the 25 folds")

fig, ax = plt.subplots(figsize=(4.2, 4.2))
lo, hi = 0.40, 0.75
ax.plot([lo, hi], [lo, hi], color="k", lw=1, ls="--")
ax.scatter(folds["Ridge (a=1)"], folds["RandomForest"], s=22, alpha=0.75, color="#4C72B0")
ax.set_xlabel("Ridge $R^2$"); ax.set_ylabel("RandomForest $R^2$")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_title("Same fold, two models")
fig.tight_layout()

# (3) The 25 "observations" are not independent: the 5 folds within a repeat
#     share molecules by construction, and across repeats the same 908
#     molecules are reused. A paired t-test assumes independent draws, so its
#     standard error is far too small and the p-value is meaningless -- it can
#     be driven arbitrarily low simply by raising n_repeats, without collecting
#     a single new molecule. Report the effect size and the win fraction, and
#     be explicit that the comparison is on one dataset.
""",
    ),
    md("""
The forest is ahead by 0.049 ± 0.041 per fold and wins 84% of folds: a real but
modest improvement, consistently in the same direction. That sentence is
defensible. "The random forest substantially outperformed linear models" is not.

## 2. Hyperparameter search

### EXERCISE 3.3

Use `GridSearchCV` to tune k for kNN over `[1, 2, 3, 5, 8, 12, 20, 30, 50]`.
Plot mean CV R<sup>2</sup> against k with its error bars, and report the best k.
"""),
    code(
        """
# TODO: GridSearchCV over n_neighbors; plot the validation curve with error bars
...
""",
        """
ks = [1, 2, 3, 5, 8, 12, 20, 30, 50]
gs = GridSearchCV(make_pipeline(StandardScaler(), KNeighborsRegressor()),
                  {"kneighborsregressor__n_neighbors": ks},
                  scoring="r2", cv=KFold(5, shuffle=True, random_state=RNG)).fit(X, y)

m = gs.cv_results_["mean_test_score"]; s = gs.cv_results_["std_test_score"]
fig, ax = plt.subplots()
ax.errorbar(ks, m, yerr=s, marker="o", capsize=3, color="#4C72B0")
ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels(ks)
ax.set_xlabel("k (neighbours)"); ax.set_ylabel("CV $R^2$")
ax.set_title("Validation curve: too flexible on the left, too rigid on the right")
fig.tight_layout()

print(f"best k = {gs.best_params_['kneighborsregressor__n_neighbors']}, "
      f"CV R2 = {gs.best_score_:.3f}")
print(pd.DataFrame({"k": ks, "R2": m.round(3), "sd": s.round(3)}).to_string(index=False))
""",
    ),
    md("""
k=1 is badly overfit (R<sup>2</sup> ≈ 0.41: a single nearest neighbour is mostly
measurement noise), the curve peaks broadly around k=8-12, and large k drifts
back toward the dummy. The peak is flat — which is the point of the next
exercise.

### EXERCISE 3.4 — how much of a leaderboard is noise?

Build a list of model configurations: kNN for each k above, ridge for
alpha in `[0.01, 0.1, 1, 10, 100, 1000]`, random forests crossing
`max_depth` in `[2, 3, 5, 8, None]` with `max_features` in `[0.3, 0.6, 1.0]`,
and `HistGradientBoostingRegressor` crossing `learning_rate` in
`[0.03, 0.1, 0.3]` with `max_leaf_nodes` in `[15, 31, 63]`. That is 39
configurations.

Score every one with the **same** 5-fold split, sort them, and print the top
eight. Then compare the gap between 1st and 8th place with the ±0.045 sd of a
single split you measured in session 2.

*Runtime two to three minutes.*
"""),
    code(
        """
# TODO: build the 39-configuration dict, score all with one shared KFold,
#       print the top 8, compare the 1st-8th gap to the single-split sd
...
""",
        """
cands = {}
for k in [1, 2, 3, 5, 8, 12, 20, 30, 50]:
    cands[f"kNN{k}"] = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k))
for a in [0.01, 0.1, 1, 10, 100, 1000]:
    cands[f"ridge{a}"] = make_pipeline(StandardScaler(), Ridge(alpha=a))
for depth in [2, 3, 5, 8, None]:
    for mf in [0.3, 0.6, 1.0]:
        cands[f"rf_d{depth}_mf{mf}"] = RandomForestRegressor(
            n_estimators=200, max_depth=depth, max_features=mf, random_state=RNG)
for lr in [0.03, 0.1, 0.3]:
    for leaves in [15, 31, 63]:
        cands[f"hgb_lr{lr}_l{leaves}"] = HistGradientBoostingRegressor(
            learning_rate=lr, max_leaf_nodes=leaves, random_state=RNG)
print(len(cands), "configurations")

cv_shared = KFold(5, shuffle=True, random_state=RNG)
board = pd.Series({n: cross_val_score(m, X, y, cv=cv_shared, scoring="r2").mean()
                   for n, m in cands.items()}).sort_values(ascending=False)
print(board.head(8).round(3).to_string())
print(f"\\n1st - 8th = {board.iloc[0]-board.iloc[7]:.3f}")
print("single-split sd measured in session 2 = 0.045")
""",
    ),
    md("""
The top eight configurations span 0.023 in R<sup>2</sup> — half the sd of a single
train/test split, and they come from three different model families. The
identity of the "winner" is therefore mostly a property of this particular fold
assignment, not of the models. Re-run it with a different `random_state` in
`cv_shared` and a different configuration will be on top.

Two practical consequences. Reporting *only* the winner's score overstates what
you know. And the effort of squeezing out the last 0.02 is better spent on
better descriptors or more molecules — which is exactly what the learning curve
at the end of this session will tell you.

## 3. Nested cross-validation

The score above is contaminated in a way session 2 did not cover. We used the
same folds to *choose* the configuration and to *report* its performance. The
maximum of 39 noisy numbers is biased upward — the winner is partly whichever
configuration best fits that fold assignment's quirks.

Nested CV measures the whole procedure "try all 39, keep the best" instead:

- an **outer** split sets aside a test fold;
- within the remaining data, an **inner** CV picks the configuration;
- the winner is refit on the outer training part and scored once on the outer
  test fold;
- repeat for every outer fold.

### EXERCISE 3.5

Implement nested CV over the 39 configurations (5 outer folds, 5 inner folds)
and compare it with the flat "best score on the leaderboard" number.

*This fits 39 x 5 x 5 models and takes several minutes — start it, then read on.*
"""),
    code(
        """
# TODO: nested CV. For each outer fold: inner-CV all 39 configs on the outer
#       training part, clone the winner, refit, score on the outer test fold.
...
""",
        """
outer = KFold(5, shuffle=True, random_state=7)
inner = KFold(5, shuffle=True, random_state=1)
nested, picked = [], []
for tr, te in outer.split(X):
    scores = {n: cross_val_score(m, X.iloc[tr], y[tr], cv=inner, scoring="r2").mean()
              for n, m in cands.items()}
    win = max(scores, key=scores.get)
    picked.append(win)
    best = clone(cands[win]).fit(X.iloc[tr], y[tr])
    nested.append(r2_score(y[te], best.predict(X.iloc[te])))
nested = np.array(nested)

print("configuration chosen in each outer fold:", picked)
print(f"nested CV R2            = {nested.mean():.3f} +- {nested.std():.3f}")
print(f"flat leaderboard winner = {board.iloc[0]:.3f}")
print(f"selection optimism      = {board.iloc[0]-nested.mean():+.3f}")
""",
    ),
    md("""
At n=908 the measured optimism is about **-0.01** — not detectable, and in fact
the nested estimate (0.645 ± 0.048) comes out slightly *above* the flat winner
(0.631). With a fold-to-fold sd of 0.048 that difference is noise; the honest
reading is that selection bias here is below what this experiment can resolve.

Worth noting what the outer folds chose: four of the five picked
`rf(max_depth=None, max_features=0.3)` and the fifth picked the same forest with
`max_features=0.6`. So the *family* is selected stably even though the
leaderboard positions within it are not — a useful distinction when someone
claims their architecture search found something.

### EXERCISE 3.6 — where selection bias does bite

Selection bias grows with the number of configurations tried and shrinks with
sample size. Repeat 3.5 on random subsamples of **120 molecules**, three times
with different subsamples, and report the mean optimism.

*Reduce the forests to `n_estimators=100` to keep this under a few minutes.*
"""),
    code(
        """
# TODO: three subsamples of 120 molecules; for each, flat best CV vs nested CV
...
""",
        """
small = {n: (clone(m).set_params(n_estimators=100)
             if isinstance(m, RandomForestRegressor) else m)
         for n, m in cands.items()}

rg = np.random.default_rng(1)
gaps = []
for rep in range(3):
    ii = rg.choice(len(X), 120, replace=False)
    Xs, ys = X.iloc[ii], y[ii]
    flat = max(cross_val_score(m, Xs, ys, cv=KFold(5, shuffle=True, random_state=rep),
                               scoring="r2").mean() for m in small.values())
    nest = []
    for tr, te in KFold(5, shuffle=True, random_state=100 + rep).split(Xs):
        sc = {n: cross_val_score(m, Xs.iloc[tr], ys[tr],
                                 cv=KFold(5, shuffle=True, random_state=1), scoring="r2").mean()
              for n, m in small.items()}
        w = clone(small[max(sc, key=sc.get)]).fit(Xs.iloc[tr], ys[tr])
        nest.append(r2_score(ys[te], w.predict(Xs.iloc[te])))
    gaps.append(flat - np.mean(nest))
    print(f"rep {rep}: flat best {flat:.3f} | nested {np.mean(nest):.3f} | optimism {gaps[-1]:+.3f}")
print(f"\\nmean optimism at n=120: {np.mean(gaps):+.3f}   (at n=908 it was about -0.01)")
""",
    ),
    md("""
At 120 molecules the same procedure is optimistic by **+0.062 on average**
(+0.126, +0.030, +0.029 across the three subsamples). Same code, same 39
configurations — only the sample size changed. Note also how much the three
subsamples differ: at this n, the optimism is itself a noisy quantity, and a
single estimate of it would not be worth much either.

The general rule: the inflation scales with the number of things you tried
divided by the amount of data you tried them on. Reported performance from a
small cohort after a large architecture or hyperparameter search should be read
as an upper bound, unless the paper says it nested the selection.

## 4. Learning curves: what to spend money on

A practical question with a real budget behind it. The model is wrong by a
factor of about eight in concentration. To improve it, should the lab measure
more molecules, or invest in better descriptors?

### EXERCISE 3.7

Use `learning_curve` for the random forest and for ridge regression, with
training sizes from 10% to 100%. Plot training and cross-validated R<sup>2</sup>
against training-set size for both. Then answer:

1. Which model is limited by variance (overfitting), and which by bias?
2. Is the forest's cross-validated score still rising at n=726?
3. What would you advise the lab to do?
"""),
    code(
        """
# TODO: learning curves for RF and Ridge; plot train vs CV score against n
...
""",
        """
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
for ax, (name, est) in zip(axes, [
        ("RandomForest", RandomForestRegressor(n_estimators=200, random_state=RNG)),
        ("Ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0)))]):
    sizes, tr_s, te_s = learning_curve(est, X, y, train_sizes=np.linspace(0.1, 1.0, 8),
                                       cv=KFold(5, shuffle=True, random_state=RNG),
                                       scoring="r2")
    ax.plot(sizes, tr_s.mean(1), marker="o", color="#C44E52", label="training score")
    ax.plot(sizes, te_s.mean(1), marker="s", color="#4C72B0", label="cross-validated")
    ax.fill_between(sizes, te_s.mean(1) - te_s.std(1), te_s.mean(1) + te_s.std(1),
                    alpha=0.2, color="#4C72B0")
    ax.set_title(name); ax.set_xlabel("training molecules")
    print(f"{name}: train {np.round(tr_s.mean(1), 3)}")
    print(f"{name}: cv    {np.round(te_s.mean(1), 3)}")
axes[0].set_ylabel("$R^2$"); axes[0].legend(loc="center right")
fig.tight_layout()
""",
    ),
    md("""
The two curves say opposite things.

The **forest** sits at 0.94 on its training data and 0.60 in cross-validation.
That 0.34 gap is variance: the model is fitting molecule-specific noise it
cannot generalise. Its cross-validated score is still climbing at the full
dataset (0.52 at n=72 to 0.60 at n=726, with no plateau), so more molecules
would still help this model.

**Ridge** scores 0.58 on training data and 0.56 in cross-validation. There is
essentially no gap, and the curve flattens by about n=400. More molecules will
not help it at all: it is bias-limited, and the ceiling is set by what a linear
function of these six descriptors can express.

So the honest advice depends on which model you intend to deploy — and for the
flexible one, *both* more molecules and better descriptors would pay. The
descriptor question is the one session 4 opens up: six numbers cannot distinguish
molecules that differ in ways that matter for toxicity, which is also why 217 of
our rows have an identical twin.

## Session 3 checklist

1. Report every candidate's score, not only the winner's.
2. Compare models on shared folds, as paired differences; report effect size and
   win fraction, not a p-value from reused folds.
3. If you selected among configurations, either nest the selection or label the
   number an upper bound.
4. State the number of configurations you tried and the sample size, so a reader
   can judge the selection bias themselves.
5. Use a learning curve before asking for more data; it may not be the
   constraint.
"""),
]

# --------------------------------------------------------------------------- #
# SESSION 4
# --------------------------------------------------------------------------- #

S4 = [
    md("""
## What the model knows, and where it stops working

A model with a defensible cross-validated score can still be useless in
deployment, and the reason is almost never the score. Today: whether the signal
is real at all, what the model is actually using, and the question that matters
most for a QSAR model in practice — **which molecules is this prediction valid
for?**

**This session (2 h)**

1. y-scrambling: an empirical null for your pipeline
2. Residuals: where the model fails, and why these descriptors have a floor
3. Feature importance, and why two methods disagree
4. The applicability domain: what happens outside the training chemistry
5. Capstone, to hand in
"""),
    code(HEADER),
    code("""
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (KFold, RepeatedKFold, train_test_split,
                                     cross_val_score, cross_val_predict)
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, root_mean_squared_error

fish = load_fish()
X, y = xy(fish)
cv5 = KFold(5, shuffle=True, random_state=RNG)
"""),
    md("""
## 1. Is there any signal? (y-scrambling)

The standard sanity check in QSAR, and a good habit everywhere: shuffle the
target, leaving the descriptor matrix untouched, and run the **entire** pipeline.
Any structure the procedure finds in shuffled data is structure it invented.

### EXERCISE 4.1

Cross-validate the random forest on the real target. Then build a null
distribution from 30 shuffles of `y`, plot it with the real score marked, and
report where the real score falls relative to the null.

*Runtime about two minutes.*
"""),
    code(
        """
# TODO: real CV R2; then 30 shuffles of y, same CV, collect the null; plot both
...
""",
        """
rf = RandomForestRegressor(n_estimators=200, random_state=RNG)
real = cross_val_score(rf, X, y, cv=cv5, scoring="r2").mean()

rng = np.random.default_rng(RNG)
null = np.array([cross_val_score(rf, X, rng.permutation(y), cv=cv5, scoring="r2").mean()
                 for _ in range(30)])

fig, ax = plt.subplots()
ax.hist(null, bins=12, color="#999999", edgecolor="white", label="y scrambled (n=30)")
ax.axvline(real, color="#C44E52", lw=2, label=f"real target: {real:.3f}")
ax.set_xlabel("cross-validated $R^2$"); ax.set_ylabel("runs")
ax.set_title("y-scrambling null for the whole pipeline")
ax.legend(); fig.tight_layout()

print(f"real      R2 = {real:+.3f}")
print(f"scrambled R2 = {null.mean():+.3f} +- {null.std():.3f}   (max {null.max():+.3f})")
""",
    ),
    md("""
The null sits at **-0.19 ± 0.04** and never gets above -0.09, while the real
score is +0.60. The signal is unambiguous.

Note where the null is centred: not at zero, but clearly *below* it. A flexible
model fitted to noise is actively worse than predicting the mean on held-out
data. That is why "my R<sup>2</sup> is positive, so there must be signal" is not
an argument — the comparison you need is against this distribution, and when
leakage is present the null moves up with the real score (try 4.1 again with the
feature-selection mistake from session 2 if you want to see it).

## 2. Residuals

### EXERCISE 4.2

Use `cross_val_predict` to get an out-of-fold prediction for every molecule.
Then:

1. Plot predicted against observed, with the 1:1 line.
2. Plot the residual against `MLOGP`. Is the error uniform across lipophilicity?
3. List the five molecules with the largest absolute residual, with their
   descriptor values. Any pattern?
4. Among the rows that share an identical descriptor vector with another row
   (session 2), compute the spread in measured LC50. What does this imply about
   the best RMSE any model on these six descriptors could achieve?
"""),
    code(
        """
# TODO: out-of-fold predictions; parity plot; residual vs MLOGP;
#       worst five molecules; LC50 spread within identical descriptor vectors
...
""",
        """
pred = cross_val_predict(RandomForestRegressor(n_estimators=200, random_state=RNG),
                         X, y, cv=cv5)
resid = y - pred

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
axes[0].scatter(y, pred, s=8, alpha=0.4, color="#4C72B0")
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=1)
axes[0].set_xlabel("observed " + TARGET_LABEL); axes[0].set_ylabel("predicted (out of fold)")
axes[0].set_title(f"parity plot, $R^2$ = {r2_score(y, pred):.3f}")
axes[1].scatter(X.MLOGP, resid, s=8, alpha=0.4, color="#4C72B0")
axes[1].axhline(0, color="k", lw=1)
axes[1].set_xlabel("MLOGP"); axes[1].set_ylabel("residual (observed - predicted)")
axes[1].set_title("residual against lipophilicity")
fig.tight_layout()

print("RMSE by MLOGP tertile:")
tert = pd.qcut(X.MLOGP, 3, labels=["low", "mid", "high"])
for lab, g in pd.Series(resid).groupby(tert, observed=True):
    print(f"  {lab:>4s}  n={len(g):3d}  RMSE={np.sqrt((g**2).mean()):.3f}")

print("\\nfive largest absolute residuals:")
worst = np.argsort(-np.abs(resid))[:5]
print(pd.concat([X.iloc[worst].round(3),
                 pd.DataFrame({"LC50": y[worst].round(2),
                               "pred": pred[worst].round(2),
                               "resid": resid[worst].round(2)}, index=X.index[worst])],
                axis=1).to_string())

groups = molecule_groups(X)
gdf = pd.DataFrame({"g": groups, "y": y})
rep = gdf.groupby("g")["y"].agg(["size", "std", "min", "max"])
rep = rep[rep["size"] > 1]
print(f"\\n{len(rep)} descriptor vectors occur more than once "
      f"({int(rep['size'].sum())} rows)")
print(f"mean within-group sd of LC50 : {rep['std'].mean():.3f} log units")
print(f"largest within-group range   : {(rep['max']-rep['min']).max():.3f} log units")
print(f"model RMSE overall           : {root_mean_squared_error(y, pred):.3f} log units")
""",
    ),
    md("""
Two things come out of this.

The residuals are not uniform, but not in the direction most people guess. RMSE
by `MLOGP` tertile is **1.03 (low) / 0.72 (mid) / 0.95 (high)** — the model does
*worst* on the least lipophilic third. And the five largest residuals are nearly
all large **under-predictions**: compounds measured as highly toxic (LC50 of 8.1,
9.6, 6.4, 7.8 on the -log10 scale) whose `MLOGP` is low or moderate, predicted
near 3-5.

That is a recognisable pattern in ecotoxicology rather than a modelling
artefact. Baseline narcosis sets a floor on toxicity as a function of
lipophilicity; compounds that act by a *specific* mechanism — binding a receptor
or inhibiting an enzyme — are far more toxic than their log P predicts, which
the field calls **excess toxicity**. A model whose dominant descriptor is `MLOGP`
reproduces the baseline and necessarily misses those compounds, and six bulk
descriptors carry nothing that would flag them. Worth asking the room: in which
direction does this model fail, and is that the safe direction for a screening
tool?

More fundamentally: rows with **identical descriptor vectors** differ in measured
LC50 by 0.32 log units on average, and by up to 3.1 log units in the worst case.
No model built on these six descriptors can distinguish those molecules — they
are literally the same input. That within-group spread is an irreducible floor
on achievable RMSE, and our model's RMSE of 0.91 is well above it, so the
descriptors are not yet the binding constraint. But it does bound what any
amount of modelling on this representation could deliver, which is the kind of
statement worth making before anyone proposes a neural network.

## 3. Feature importance, and two methods that disagree

### EXERCISE 4.3

Fit a random forest on 75% of the data. Compute

- the built-in `feature_importances_` (mean impurity decrease, computed on the
  **training** data), and
- `permutation_importance` on the **held-out** 25%, with 30 repeats.

Plot them side by side. Where do they disagree, and which would you trust for
the claim "descriptor X matters for toxicity"?
"""),
    code(
        """
# TODO: fit RF on a 75/25 split; impurity importance vs permutation importance
#       on the held-out part; plot both
...
""",
        """
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=RNG)
rf = RandomForestRegressor(n_estimators=500, random_state=RNG).fit(X_tr, y_tr)
pi = permutation_importance(rf, X_te, y_te, n_repeats=30, random_state=RNG, scoring="r2")

imp = pd.DataFrame({"feature": X.columns,
                    "impurity (train)": rf.feature_importances_,
                    "permutation (test)": pi.importances_mean,
                    "perm sd": pi.importances_std}).sort_values("permutation (test)")
print(imp.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(6.4, 3.4))
pos = np.arange(len(imp))
ax.barh(pos - 0.2, imp["impurity (train)"], height=0.38, label="impurity (training)", color="#CCB974")
ax.barh(pos + 0.2, imp["permutation (test)"], height=0.38, xerr=imp["perm sd"],
        label="permutation (held out)", color="#4C72B0", capsize=2)
ax.set_yticks(pos); ax.set_yticklabels(imp.feature)
ax.set_xlabel("importance (different units)")
ax.legend(); fig.tight_layout()
""",
    ),
    md("""
`MLOGP` dominates both rankings, which is what the narcosis mechanism predicts.
But the two methods disagree on the rest: impurity importance gives `NdsCH`
0.03 while permuting it on held-out data costs 0.09 of R<sup>2</sup>, and the
ordering of the middle features changes.

They measure different things. Impurity importance is computed on the training
data and is biased toward features with many distinct values — a feature with
500 unique values offers far more candidate split points than a 0-4 count, so it
accumulates impurity decrease even when it is partly fitting noise. Permutation
importance on held-out data asks a question you actually care about: how much
predictive performance is lost if this column is destroyed?

### EXERCISE 4.4 — correlated descriptors break both

Now switch to the aquatic dataset (`load_aquatic()`), where `TPSA(Tot)` and
`SAacc` are correlated at r = 0.86 — both are measures of polar surface.

1. Compute both importances there. Look at `TPSA(Tot)`, `SAacc` and `GATS1p`.
2. Drop `SAacc` entirely, refit, and re-score. What happens to the model's
   performance, and what happens to `TPSA(Tot)`'s importance?
3. Write one sentence stating what a permutation importance of ~0 means when the
   feature has a correlated partner in the matrix.
"""),
    code(
        """
aqua = load_aquatic()
Xa, ya = xy(aqua)
print("corr(TPSA, SAacc) =", round(Xa[["TPSA(Tot)", "SAacc"]].corr().iloc[0, 1], 3))

# TODO: (1) impurity + permutation importance on the aquatic data
# TODO: (2) drop SAacc, refit, compare CV R2 and TPSA importance
...
""",
        """
aqua = load_aquatic()
Xa, ya = xy(aqua)
print("corr(TPSA, SAacc) =", round(Xa[["TPSA(Tot)", "SAacc"]].corr().iloc[0, 1], 3))

Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(Xa, ya, test_size=0.25, random_state=RNG)
rfa = RandomForestRegressor(n_estimators=500, random_state=RNG).fit(Xa_tr, ya_tr)
pia = permutation_importance(rfa, Xa_te, ya_te, n_repeats=30, random_state=RNG, scoring="r2")
print("\\n(1) full descriptor set")
print(pd.DataFrame({"feature": Xa.columns,
                    "impurity": rfa.feature_importances_,
                    "permutation": pia.importances_mean}).round(3).to_string(index=False))

print("\\n(2) after dropping SAacc")
Xd = Xa.drop(columns="SAacc")
cv_a = KFold(5, shuffle=True, random_state=RNG)
full_cv = cross_val_score(RandomForestRegressor(n_estimators=500, random_state=RNG),
                          Xa, ya, cv=cv_a, scoring="r2").mean()
drop_cv = cross_val_score(RandomForestRegressor(n_estimators=500, random_state=RNG),
                          Xd, ya, cv=cv_a, scoring="r2").mean()
rfd = RandomForestRegressor(n_estimators=500, random_state=RNG).fit(
    Xd.loc[Xa_tr.index], ya_tr)
pid = permutation_importance(rfd, Xd.loc[Xa_te.index], ya_te, n_repeats=30,
                             random_state=RNG, scoring="r2")
print(f"CV R2 with all 8 descriptors : {full_cv:.3f}")
print(f"CV R2 without SAacc          : {drop_cv:.3f}")
print(pd.DataFrame({"feature": Xd.columns,
                    "impurity": rfd.feature_importances_,
                    "permutation": pid.importances_mean}).round(3).to_string(index=False))

# (3) A permutation importance near zero means "this model does not need this
#     column, given the others it has" -- not "this property is irrelevant to
#     the biology". With a correlated partner present, the information survives
#     the permutation and the importance is shared out between them, in a split
#     that depends on the fitting noise rather than on the chemistry.
""",
    ),
    md("""
Dropping a descriptor correlated at 0.86 with another leaves cross-validated
performance essentially unchanged — 0.512 with all eight descriptors, 0.501
without `SAacc`, a difference well inside the ±0.08 fold-to-fold spread — while
the importances rearrange. `H-050` goes from 0.029 to 0.089, and `TPSA(Tot)`
from 0.217 to 0.190: the model redistributes what it leans on, and the
redistribution does not even go to the dropped descriptor's closest partner.

So feature importance is a statement about **this fitted model on this
descriptor set**, not about mechanism. If the question is "does polar surface
area matter for *Daphnia* toxicity", importance rankings are not the instrument
that answers it.

## 4. The applicability domain

Cross-validation answers: how well does the model predict molecules *drawn from
the same pool* as the training molecules? That is rarely the deployment question.
In practice the model is used on a new chemical series — compounds more
lipophilic, larger, or differently substituted than anything in the training set.

### EXERCISE 4.5

Simulate that honestly on both datasets. Take the top quartile of `MLOGP` as the
test set and train on the remaining 75%. Compare with a random split of the
same size.

1. Do this for the fish dataset, reporting R<sup>2</sup> and RMSE for both splits.
2. Repeat for the aquatic dataset.
3. Compare the mean LC50 of the training and test portions. What systematic
   error do you expect, and in which direction?
"""),
    code(
        """
# TODO: for each dataset, hold out the top MLOGP quartile, train on the rest,
#       and compare with random splits of the same size
...
""",
        """
def extrapolation_test(Xd, yd, feature="MLOGP", q=0.75, label=""):
    thr = np.quantile(Xd[feature], q)
    hi = (Xd[feature] > thr).to_numpy()
    rf = RandomForestRegressor(n_estimators=500, random_state=RNG).fit(Xd[~hi], yd[~hi])
    p = rf.predict(Xd[hi])
    rnd = []
    for s in range(10):
        tr, te = train_test_split(np.arange(len(Xd)), test_size=float(hi.mean()), random_state=s)
        m = RandomForestRegressor(n_estimators=500, random_state=RNG).fit(Xd.iloc[tr], yd[tr])
        rnd.append(r2_score(yd[te], m.predict(Xd.iloc[te])))
    print(f"{label}: hold out {feature} > {thr:.2f}  (n_test = {hi.sum()})")
    print(f"   extrapolation : R2 = {r2_score(yd[hi], p):+.3f}   RMSE = {root_mean_squared_error(yd[hi], p):.3f}")
    print(f"   random split  : R2 = {np.mean(rnd):+.3f}")
    print(f"   mean LC50 train {yd[~hi].mean():.2f} -> test {yd[hi].mean():.2f};"
          f"  mean prediction {p.mean():.2f}")
    return hi, p

hi_f, pred_f = extrapolation_test(X, y, label="fish")
print()
hi_a, pred_a = extrapolation_test(Xa, ya, label="aquatic")

fig, ax = plt.subplots(figsize=(4.6, 4.4))
ax.scatter(y[hi_f], pred_f, s=14, alpha=0.6, color="#C44E52")
ax.plot([2, 10], [2, 10], "k--", lw=1)
ax.set_xlabel("observed LC50 (high-MLOGP quartile)")
ax.set_ylabel("predicted from low-MLOGP training set")
ax.set_title("Predictions outside the training domain")
fig.tight_layout()
""",
    ),
    md("""
This is the largest effect in the whole course. On the fish data,
cross-validated R<sup>2</sup> of 0.61 becomes **0.04** when the test molecules are
simply more lipophilic than the training molecules. On the aquatic data it
becomes **-0.66** — markedly worse than predicting the training mean.

The mechanism is visible in the parity plot: the forest cannot predict outside
the range of target values it was trained on, because every prediction is an
average of training leaves. Mean LC50 rises from 3.62 to 5.41 across the split,
and the predictions stay near the training range, so the model is systematically
**under-predicting toxicity** — exactly the direction that matters for a safety
assessment.

Note also that the choice of splitting descriptor matters: holding out the top
quartile of `CIC0` instead gives R<sup>2</sup> = 0.28, not 0.04. "Out of domain"
is not one thing.

### EXERCISE 4.6 — an applicability-domain indicator

A model used in practice should refuse to answer, or flag its answer, for
molecules unlike its training set. Build the simplest such indicator.

1. Fit `NearestNeighbors` on the **scaled** training descriptors. For each test
   molecule compute the mean distance to its 5 nearest training molecules.
2. Split the test molecules into quartiles of that distance and report the RMSE
   in each quartile.
3. Verify the indicator also flags the high-`MLOGP` molecules from 4.5.
"""),
    code(
        """
# TODO: kNN-distance to the training set as a domain indicator;
#       RMSE by distance quartile
...
""",
        """
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=RNG)
sc = StandardScaler().fit(X_tr)
nn = NearestNeighbors(n_neighbors=5).fit(sc.transform(X_tr))
dist = nn.kneighbors(sc.transform(X_te))[0].mean(axis=1)

rf = RandomForestRegressor(n_estimators=500, random_state=RNG).fit(X_tr, y_tr)
err = np.abs(y_te - rf.predict(X_te))

q = pd.qcut(dist, 4, labels=["nearest", "near", "far", "farthest"])
print("test molecules binned by mean distance to 5 nearest training molecules")
for lab, g in pd.DataFrame({"err": err, "d": dist}).groupby(q, observed=True):
    print(f"  {lab:>9s}  n={len(g):3d}  mean distance {g.d.mean():.2f}  "
          f"RMSE {np.sqrt((g.err**2).mean()):.3f}")

sc_all = StandardScaler().fit(X[~hi_f])
nn_all = NearestNeighbors(n_neighbors=5).fit(sc_all.transform(X[~hi_f]))
d_out = nn_all.kneighbors(sc_all.transform(X[hi_f]))[0].mean(axis=1)
print(f"\\nmean distance, random held-out molecules      : {dist.mean():.2f}")
print(f"mean distance, high-MLOGP held-out molecules  : {d_out.mean():.2f}")

fig, ax = plt.subplots()
ax.scatter(dist, err, s=12, alpha=0.5, color="#4C72B0")
ax.set_xlabel("mean distance to 5 nearest training molecules (scaled space)")
ax.set_ylabel("absolute error (log units)")
ax.set_title("Error grows with distance from the training chemistry")
fig.tight_layout()
""",
    ),
    md("""
A crude indicator, and it works: error rises across the distance quartiles, and
the high-`MLOGP` molecules sit far outside the distances seen in a random
held-out sample. A deployed QSAR model should report this alongside every
prediction; the OECD validation principles require a defined applicability
domain for exactly this reason.

## 5. Capstone — to hand in

You have a second dataset you have barely used: `load_aquatic()`, 546 molecules,
eight descriptors, LC50 toward *Daphnia magna*. Treat it as a new project and
apply everything from the four sessions.

**Deliverable: one notebook plus a methods paragraph of at most 250 words.**

Required content.

1. **Baseline.** The dummy's R<sup>2</sup> and RMSE, and the RMSE as a
   fold-error in concentration.
2. **A tuned model, selected honestly.** Choose among at least three model
   families with at least three hyperparameter settings each. Report either a
   nested-CV estimate or a flat estimate explicitly labelled as an upper bound,
   and state how many configurations you tried.
3. **A spread, not a point.** Repeated cross-validation, mean ± sd, and a
   sentence on whether your chosen model is distinguishable from the runner-up.
4. **Leakage audit.** State for each data-dependent step where it sits relative
   to the resampling loop. Report the number of rows sharing an identical
   descriptor vector and what you did about it.
5. **A y-scrambling null** for your final pipeline.
6. **An applicability domain.** One extrapolation experiment of your choice, one
   domain indicator, and a statement of which molecules your model should not be
   used for.
7. **One honest negative result.** Something you expected to matter that did
   not, with the number that shows it.

The methods paragraph should be written so that a reader could reproduce your
estimate and would not be surprised by your model's performance in deployment.

Marking weights the audit and the applicability domain as heavily as the
performance. A submission reporting R<sup>2</sup> = 0.50 with a correct audit
scores above one reporting 0.60 without.
"""),
    code(
        """
# Capstone scaffold. Everything below is yours to replace.
aqua = load_aquatic()
Xa, ya = xy(aqua)
print(aqua.shape)
print(f"{len(set(molecule_groups(Xa)))} distinct descriptor vectors in {len(Xa)} rows")
aqua.describe().T.round(2)
""",
        """
# Reference figures for the capstone (instructor copy).
# 5-fold x 5-repeat CV on the aquatic dataset:
#     dummy          R2 = -0.008 +- 0.011     RMSE 1.663  (factor 46)
#     ridge(alpha=1) R2 =  0.458 +- 0.058     RMSE 1.216  (factor 16)
#     kNN(k=5)       R2 =  0.487 +- 0.082
#     HistGB         R2 =  0.509 +- 0.094
#     RF(500)        R2 =  0.533 +- 0.083     RMSE 1.125  (factor 13)
# Extrapolation, top MLOGP quartile held out: R2 = -0.657 (random split 0.489)
# Only 16 of the 546 rows share a descriptor vector with another row here
# (against 217 of 908 in the fish data), so a grouped split makes no measurable
# difference: the measured gap is +0.000 for a random forest and +0.016 for
# kNN(k=1). That is a legitimate answer to requirement 7, provided the student
# reports the count and the measured gap.
aqua = load_aquatic()
Xa, ya = xy(aqua)
print(aqua.shape)
print(f"{len(set(molecule_groups(Xa)))} distinct descriptor vectors in {len(Xa)} rows")
aqua.describe().T.round(2)
""",
    ),
    md("""
## Reporting checklist

What a defensible supervised-learning result contains. Keep this; it applies
well beyond QSAR.

1. The baseline, in the same units and on the same folds.
2. The performance metric in units someone in the field can interpret — a
   fold-error, not only an R<sup>2</sup>.
3. An estimate with a spread, from repeated resampling, not a single split.
4. The resampling scheme, including the grouping variable if the rows are not
   independent.
5. Every data-dependent step located relative to the resampling loop.
6. The number of model configurations tried, and whether selection was nested.
7. An empirical null for the whole pipeline.
8. The applicability domain, and the performance drop outside it.
9. At least one thing you checked that turned out not to matter.
"""),
]


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    build(S1, "01_first_model", "Session 1 - A first model, and the baseline it has to beat")
    build(S2, "02_variance_and_leakage", "Session 2 - Where the honest number comes from")
    build(S3, "03_model_selection", "Session 3 - Choosing a model without fooling yourself")
    build(S4, "04_trust_and_domain", "Session 4 - What the model knows, and where it stops working")

#!/usr/bin/env python
# coding: utf-8

# # Session 2 - Neural networks, and classifying enzymes
# 
# **AI in Quantitative Biology - practical 2 of 3 (90 min)**
# 
# Two parts:
# 
# | Part | Topic | Time |
# |---|---|---|
# | A | A small neural network on the fish toxicity data (regression) | 35 min |
# | B | Can we tell an enzyme's class from its amino-acid sequence? (classification) | 50 min |
# 
# Part B introduces a trap that is everywhere in biological machine learning: **related sequences
# leaking between training and test set**.

# ## Setup
# 
# Everything runs on a laptop CPU in a few minutes; no GPU needed. The notebook expects the
# `data/` folder of the course (it looks for it in the current folder and the folders above).
# On Google Colab, upload the `data/` folder to your Drive in `MyDrive/AI_QBio/data` and run the
# cell below: it mounts the Drive and finds it there.

# In[ ]:


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RNG = 0                                         # fixed seed: everyone gets the same numbers
pd.set_option("display.precision", 3)

import os
from pathlib import Path

def find_data_dir():
    """$QBIO_DATA if set, else a `data/` folder here or above, else Google Drive on Colab."""
    if os.environ.get("QBIO_DATA"):
        return Path(os.environ["QBIO_DATA"])
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "data" / "qsar_fish_toxicity.csv").exists():
            return p / "data"
    if "google.colab" in sys.modules:
        from google.colab import drive
        drive.mount("/content/drive")
        return Path("/content/drive/MyDrive/AI_QBio/data")
    raise FileNotFoundError("Cannot find the data/ folder: set QBIO_DATA to its path")

DATA_DIR = find_data_dir()
print("data from", DATA_DIR)

FISH_COLUMNS = ["CIC0", "SM1_Dz", "GATS1i", "NdsCH", "NdssC", "MLOGP", "LC50"]

def load_fish():
    """Fathead minnow 96 h LC50 for 908 chemicals (UCI QSAR fish toxicity)."""
    return pd.read_csv(DATA_DIR / "qsar_fish_toxicity.csv", sep=";", header=None, names=FISH_COLUMNS)

def load_enzymes():
    """2047 Swiss-Prot enzymes labelled by top-level EC class."""
    return pd.read_csv(DATA_DIR / "enzymes.tsv", sep="\t")

AA = "ACDEFGHIKLMNPQRSTVWY"                     # the 20 standard amino acids

def aa_composition(seq):
    """Fraction of each of the 20 amino acids in `seq` (a vector of length 20)."""
    return np.array([seq.count(a) for a in AA]) / len(seq)

def plot_confusion(y_true, y_pred, labels, title=None):
    """Confusion matrix as a heatmap: rows = true class, normalised per row."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(len(labels) + 2, len(labels) + 1))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    if title:
        ax.set_title(title)


# # Part A - A neural network for toxicity
# 
# A **multi-layer perceptron (MLP)** is a stack of layers. Each *neuron* computes a weighted sum of its
# inputs and passes it through a non-linear *activation* (here ReLU: $\max(0, z)$):
# 
# $$ h_j = \mathrm{ReLU}\Big(\sum_i w_{ji} x_i + b_j\Big), \qquad \hat y = \sum_j v_j h_j + c $$
# 
# The weights $w, v$ and biases $b, c$ start random and are adjusted step by step (gradient descent)
# to reduce the squared error on the training data. With enough hidden neurons an MLP can
# approximate any smooth function - which is both its strength and its danger.
# 
# ```
#  6 descriptors  ->  [32 hidden neurons, ReLU]  ->  1 output (LC50)
# ```

# In[ ]:


from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

fish = load_fish()
X, y = fish.drop(columns="LC50"), fish["LC50"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RNG)

mlp = make_pipeline(
    StandardScaler(),                     # neural nets also want scaled inputs
    MLPRegressor(hidden_layer_sizes=(32,), max_iter=3000, random_state=RNG),
)
mlp.fit(X_train, y_train)
print(f"MLP test R2 = {r2_score(y_test, mlp.predict(X_test)):.3f}")


# The network was trained iteratively. `loss_curve_` stores the training loss after each pass over
# the data (*epoch*).

# In[ ]:


net = mlp[-1]
plt.plot(net.loss_curve_)
plt.yscale("log")
plt.xlabel("epoch"); plt.ylabel("training loss (squared error / 2)")
plt.title(f"{net.n_iter_} epochs");


# ✏️ **Exercise A.1.** Compare the MLP with the random forest of session 1 using the same 5-fold
# cross-validation (`cv` below). Which one wins?

# In[ ]:


cv = KFold(n_splits=5, shuffle=True, random_state=RNG)
rf = RandomForestRegressor(n_estimators=300, random_state=RNG, n_jobs=-1)

r2_mlp = cross_val_score(mlp, X, y, cv=cv, scoring="r2")
r2_rf = cross_val_score(rf, X, y, cv=cv, scoring="r2")
print(f"MLP R2 = {r2_mlp.mean():.3f}   RF R2 = {r2_rf.mean():.3f}")


# ### How big should the network be?
# 
# We train networks with two hidden layers of increasing width and compare the R² on the
# **training** set with the R² on the **test** set.

# In[ ]:


sizes = [1, 2, 4, 8, 32, 128, 512]

def size_sweep(alpha):
    """Train/test R2 for 2-hidden-layer MLPs of each width; alpha = L2 penalty on the weights."""
    rows = []
    for h in sizes:
        m = make_pipeline(StandardScaler(),
                          MLPRegressor(hidden_layer_sizes=(h, h), alpha=alpha,
                                       max_iter=2000, random_state=RNG))
        m.fit(X_train, y_train)
        rows.append((h, r2_score(y_train, m.predict(X_train)), r2_score(y_test, m.predict(X_test))))
    return pd.DataFrame(rows, columns=["width", "train R2", "test R2"]).set_index("width")

sweep = size_sweep(alpha=1e-4)          # almost no regularisation
sweep.plot(logx=True, marker="o")
plt.ylabel("R2"); plt.title("alpha = 0.0001");
sweep


# ❓ What happens to the gap between training and test R² as the network grows? What is this called?
# 
# ✏️ **Exercise A.2.** `alpha` adds a penalty on large weights (*L2 regularisation*, "weight decay").
# Run `size_sweep(alpha=1.0)` and compare with the plot above.

# In[ ]:


sweep_reg = size_sweep(alpha=1.0)
ax = sweep.plot(logx=True, marker="o", style="--", alpha=0.5)
sweep_reg.plot(logx=True, marker="o", ax=ax)
plt.legend(["train, alpha=1e-4", "test, alpha=1e-4", "train, alpha=1", "test, alpha=1"])
plt.ylabel("R2");
sweep_reg


# **Answer.** Without regularisation, the bigger networks fit the training set better and better
# (train R² up to ≈ 0.93) while the test R² does not improve: this is **overfitting** - the network
# memorises the 681 training chemicals, noise included. With `alpha=1.0` the training R² stays
# lower (≈ 0.8), the gap shrinks, and the large networks generalise best (test R² ≈ 0.55).
# On a small table of 6 descriptors, an MLP does **not** beat a random forest (≈ 0.58 vs ≈ 0.60
# in cross-validation). Neural networks shine with large data and with raw, structured inputs such as
# sequences and images - which is where we go next.

# # Part B - Can we tell an enzyme's class from its sequence?
# 
# Enzymes are classified by the reaction they catalyse into **EC (Enzyme Commission) classes**:
# 
# | EC | class | reaction |
# |---|---|---|
# | 1 | Oxidoreductases | transfer of electrons (redox) |
# | 2 | Transferases | transfer of a group (methyl, phosphate, ...) |
# | 3 | Hydrolases | cleavage by water |
# | 4 | Lyases | cleavage without water or oxidation |
# | 5 | Isomerases | rearrangement within a molecule |
# | 6 | Ligases | joining two molecules, using ATP |
# 
# Our dataset contains **2047 enzymes** from UniProt/Swiss-Prot (manually reviewed), from five model
# organisms (human, *E. coli*, baker's yeast, *Arabidopsis*, *B. subtilis*), about 350 per class.
# Each enzyme also has its **protein family** (groups of evolutionarily related proteins).
# 
# ## B1 - Meet the data

# In[ ]:


enz = load_enzymes()
print(enz.shape)
enz.head()


# In[ ]:


print(enz["ec_name"].value_counts(), "\n")
pd.crosstab(enz["organism"], enz["ec_name"])


# In[ ]:


classes = sorted(enz["ec_name"].unique())
enz.boxplot(column="length", by="ec_name", figsize=(8, 4), rot=30)
plt.suptitle(""); plt.title("sequence length by class"); plt.ylabel("residues");


# ✏️ **Exercise B1.1.** Print one example sequence and its protein name. How many proteins come from
# the largest protein family, and what is that family? *Hint:* `value_counts()` on `family`.

# In[ ]:


row = enz.iloc[0]
print(row["protein_name"], "|", row["organism"], "| EC", row["ec"])
print(row["sequence"])
print()
print(enz["family"].value_counts().head(5))


# ## B2 - Amino-acid composition
# 
# The simplest way to turn a sequence of any length into a fixed-size vector: the **fraction of
# each of the 20 amino acids**. This forgets the order completely.

# In[ ]:


comp = pd.DataFrame(np.vstack(enz["sequence"].map(aa_composition)), columns=list(AA))
comp.head()


# Is any amino acid enriched in a class? We compare each class's mean composition with the overall
# mean, on a $\log_2$ scale (+1 = twice as frequent).

# In[ ]:


class_mean = comp.groupby(enz["ec_name"]).mean()
enrichment = np.log2(class_mean / comp.mean())

plt.figure(figsize=(10, 3.5))
plt.imshow(enrichment, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
plt.colorbar(label="log2 enrichment")
plt.xticks(range(20), AA); plt.yticks(range(len(classes)), enrichment.index);


# In[ ]:


from sklearn.decomposition import PCA

Z = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(comp))
plt.figure(figsize=(6, 5))
for c in classes:
    sel = (enz["ec_name"] == c).values
    plt.scatter(Z[sel, 0], Z[sel, 1], s=6, alpha=0.5, label=c)
plt.legend(markerscale=3, fontsize=8); plt.xlabel("PC1"); plt.ylabel("PC2");


# ❓ How large are the differences in composition between classes? Do the classes separate in the
# PCA map? What do you expect for a classifier?
# 
# **Answer.** The enrichments are tiny (within ±0.25 on a log2 scale, i.e. < 20%) and the classes
# overlap completely in the PCA map. Composition carries only a faint signal about the catalysed
# reaction: expect a classifier to do better than chance, but not much. (Some enrichments have a
# story: Cys is enriched among transferases because ~15% of them transfer ubiquitin (EC 2.3.2.x,
# e.g. RING-type E3 ligases), whose zinc-binding domains contain twice the average Cys.)

# ## B3 - Classifiers
# 
# Features: the 20 composition values plus log(length). We measure **accuracy** (fraction of
# correct predictions) with stratified 5-fold cross-validation (each fold keeps the class
# proportions). With 6 balanced classes, **random guessing gives about 1/6 = 0.17**.

# In[ ]:


from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

Xe = comp.assign(log_length=np.log(enz["length"]))
ye = enz["ec_name"]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)

classifiers = {
    "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "random forest": RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1),
    "MLP (64)": make_pipeline(StandardScaler(),
                              MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, early_stopping=True,
                                            max_iter=500, random_state=RNG)),
}
acc_random = {}
for name, clf in classifiers.items():
    acc_random[name] = cross_val_score(clf, Xe, ye, cv=skf, scoring="accuracy").mean()
    print(f"{name:20s} accuracy = {acc_random[name]:.3f}")


# Accuracy hides *which* classes get confused. A **confusion matrix** shows, for each true class (row),
# how the predictions are distributed (columns). `cross_val_predict` gives one out-of-fold prediction
# per enzyme.

# In[ ]:


pred_rf = cross_val_predict(classifiers["random forest"], Xe, ye, cv=skf)
plot_confusion(ye, pred_rf, classes, title="random forest, random 5-fold CV");


# ❓ Which class is easiest to recognise, which is hardest? Any idea why?
# 
# **Answer.** Look at the diagonal: no class is easy (recall ≈ 0.30-0.45 each) and the errors
# spread over all columns. Oxidoreductases are the hardest: a huge and chemically diverse class
# (NAD-, FAD-, heme-, metal-dependent enzymes in many unrelated folds) with no common composition.
# EC classes describe the *chemistry* of the reaction, which evolution has invented many times in
# unrelated folds, so composition is a weak proxy for it.

# ## B4 - The homology trap
# 
# Proteins of the same **family** share a common ancestor and often have similar sequences - and
# similar compositions. With a random split, a test enzyme usually has a close relative
# (e.g. the same enzyme from another organism) in the training set, and the model can just
# "recognise the family" instead of learning anything general.
# 
# A fairer test for a *new* kind of enzyme: keep **whole families** out of the training set.
# `StratifiedGroupKFold` does this - all members of a group end up in the same fold.

# In[ ]:


from sklearn.model_selection import StratifiedGroupKFold

print(enz["family"].nunique(), "families;", (enz["family"].value_counts() > 1).sum(), "with more than one member")
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RNG)
groups = enz["family"]


# ✏️ **Exercise B4.1.** Repeat the cross-validation of the three classifiers with `cv=sgkf` and
# `groups=groups` (pass it as `cross_val_score(..., groups=groups)`), and print both accuracies
# side by side.

# In[ ]:


for name, clf in classifiers.items():
    acc_family = cross_val_score(clf, Xe, ye, cv=sgkf, groups=groups, scoring="accuracy").mean()
    print(f"{name:20s} random split = {acc_random[name]:.3f}   family split = {acc_family:.3f}")


# ❓ Which model loses the most when families are held out, and why? Which number would you report
# in a paper claiming "our model predicts the class of newly discovered enzymes"?
# 
# **Answer.** The random forest drops the most (≈ 0.40 → 0.30): it is the most flexible model and
# partly works as a "find the relative" lookup table. Logistic regression and the MLP drop less
# because they were learning less family-specific detail in the first place. For new enzymes, the
# family-split number is the honest one. The random-split number answers a different question
# ("can I annotate a new member of a family I already know?") - for which a BLAST search against
# annotated sequences would beat all of these models.

# ## Summary
# 
# * An MLP is a flexible function approximator; on a small table it overfits easily, needs scaled
#   inputs and regularisation, and does not beat a random forest.
# * **Data mining before modelling** told us what to expect: the enzyme classes barely differ in
#   composition, and models got 0.25-0.40 accuracy (chance: 0.17).
# * **How you split matters as much as which model you use.** Related sequences across the
#   train/test boundary inflate the scores; split by family (or by sequence identity) to estimate
#   performance on genuinely new proteins.
# * Composition throws away the **order** of the residues. Next session: a neural network that reads
#   the sequence residue by residue.
# 
# ### If you have time
# 
# ✏️ Replace composition by **dipeptide** frequencies (400 features: AA, AC, ..., YY) and redo the
# random vs family comparison for the random forest. Does the gap grow or shrink?

# In[ ]:


import itertools
DIPEPTIDES = ["".join(p) for p in itertools.product(AA, repeat=2)]
index = {dp: i for i, dp in enumerate(DIPEPTIDES)}

def dipeptide_composition(seq):
    v = np.zeros(len(DIPEPTIDES))
    for i in range(len(seq) - 1):
        v[index[seq[i:i + 2]]] += 1
    return v / (len(seq) - 1)

Xd = np.vstack(enz["sequence"].map(dipeptide_composition))
rf = classifiers["random forest"]
print(f"random split: {cross_val_score(rf, Xd, ye, cv=skf).mean():.3f}")
print(f"family split: {cross_val_score(rf, Xd, ye, cv=sgkf, groups=groups).mean():.3f}")


# **Answer.** Dipeptides do not help (≈ 0.38 random, ≈ 0.28 family split, vs 0.40 / 0.31 with
# composition): 400 noisy features for 2047 proteins is a lot, and the leakage gap stays ≈ 0.09.
# More features is not the same as more information.

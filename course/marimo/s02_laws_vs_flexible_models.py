import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Session 2 - What a flexible model cannot do

    **AI in Quantitative Biology - practical 2 of 3 (90 min)**

    In session 1 a two-parameter law described the metabolic rate of 422 mammals. Today we give the
    same data to *machine-learning* models that assume nothing about the shape of the relation: nearest
    neighbours, a random forest and a neural network.

    **Suppose we train every model only on mammals lighter than 1 kg. Which one predicts the
    elephant?** Write your guess down now.

    | Part | Topic | Time |
    |---|---|---|
    | 1 | Four models, one curve | 20 min |
    | 2 | Predict the big ones | 25 min |
    | 3 | Species are not independent | 20 min |
    | 4 | A law that breaks | 25 min |

    ✏️ marks exercises and ❓ questions to discuss with your neighbour: try them yourself first, the solution follows right after.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    Runs on a laptop CPU in about a minute. The notebook expects the `data/` folder of the course (it
    looks for it in the current folder and the folders above). On Google Colab, upload the `data/`
    folder to your Drive in `MyDrive/AI_QBio/data` and run the cell below.
    """)
    return


@app.cell(hide_code=True)
def _():
    # @title Setup: imports and data (run me)
    import sys
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold, GroupKFold, cross_val_score
    from sklearn.metrics import r2_score

    RNG = 0                                         # fixed seed: everyone gets the same numbers
    pd.set_option("display.precision", 3)
    warnings.filterwarnings("ignore", category=UserWarning)

    import os
    from pathlib import Path

    def find_data_dir():
        "$QBIO_DATA if set, else a `data/` folder here or above, else Google Drive on Colab."
        if os.environ.get("QBIO_DATA"):
            return Path(os.environ["QBIO_DATA"])
        here = Path.cwd().resolve()
        for p in [here, *here.parents]:
            if (p / "data" / "anage.tsv").exists():
                return p / "data"
        if "google.colab" in sys.modules:
            from google.colab import drive
            drive.mount("/content/drive")
            return Path("/content/drive/MyDrive/AI_QBio/data")
        raise FileNotFoundError("Cannot find the data/ folder: set QBIO_DATA to its path")

    DATA_DIR = find_data_dir()
    print("data from", DATA_DIR)

    anage = pd.read_csv(DATA_DIR / "anage.tsv", sep="\t")
    mam = anage[anage["class"] == "Mammalia"].dropna(subset=["bmr_W", "bmr_mass_g"])
    X = np.log10(mam[["bmr_mass_g"]].values)        # one feature: log10 body mass (g)
    y = np.log10(mam["bmr_W"].values)               # target: log10 metabolic rate (W)
    print(X.shape, y.shape)
    return (
        DATA_DIR,
        GroupKFold,
        KFold,
        KNeighborsRegressor,
        LinearRegression,
        MLPRegressor,
        RNG,
        RandomForestRegressor,
        StandardScaler,
        X,
        anage,
        cross_val_score,
        make_pipeline,
        np,
        pd,
        plt,
        r2_score,
        y,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1 - Four models, one curve

    | model | how it predicts | parameters |
    |---|---|---|
    | **linear** | the straight line of session 1 (Kleiber's law) | 2 |
    | **kNN** | average of the 10 species closest in mass | none: it keeps the data |
    | **random forest** | average of 200 decision trees, each a staircase of "mass < threshold?" questions | thousands of thresholds |
    | **neural network** (MLP) | two layers of 64 units, weights found by gradient descent | about 4 400 weights |

    To compare them fairly we use **5-fold cross-validation**: split the species in 5 groups, train on
    4, predict the 5th, rotate. Each species is predicted once by a model that never saw it.
    `cross_val_score` does it for us and returns one R² per fold.
    """)
    return


@app.cell
def _(
    KFold,
    KNeighborsRegressor,
    LinearRegression,
    MLPRegressor,
    RNG,
    RandomForestRegressor,
    StandardScaler,
    X,
    cross_val_score,
    make_pipeline,
    y,
):
    def make_models(seed=RNG):
        return {'linear': LinearRegression(), 'kNN': KNeighborsRegressor(n_neighbors=10), 'random forest': RandomForestRegressor(n_estimators=200, random_state=seed), 'neural network': make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=5000, random_state=seed))}
    cv = KFold(n_splits=5, shuffle=True, random_state=RNG)
    for _name, _model in make_models().items():
        r2 = cross_val_score(_model, X, y, cv=cv, scoring='r2')
        print(f'{_name:15s} R2 = {r2.mean():.3f} +/- {r2.std():.3f}')
    return cv, make_models


@app.cell
def _(X, make_models, np, plt, y):
    grid = np.linspace(X.min(), X.max(), 400)[:, None]
    _fig, _axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for _ax, (_name, _model) in zip(_axes, make_models().items()):
        _ax.scatter(X, y, s=5, color='lightgrey')
        _ax.plot(grid, _model.fit(X, y).predict(grid), 'r', lw=1.5)
        _ax.set_title(_name)
        _ax.set_xlabel('log10 body mass (g)')
    _axes[0].set_ylabel('log10 metabolic rate (W)')
    plt.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ❓ Which model is best? Describe the four curves.

    **Answer.** They cannot be told apart: all four reach R² ≈ 0.93 - 0.95, and the differences are
    of the size of the fold-to-fold spread. The curves differ in character: a line, a wobbly average
    (kNN), a staircase (forest), a smooth bent line (network). The flexible models wiggle around the
    law without improving on it. Two parameters were enough, and several thousand did not hurt.

    So far there is no reason to prefer the law, apart from taste.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2 - Predict the big ones

    Now the experiment of the first page. Imagine that only small mammals had ever been measured:
    we **train below 1 kg** and **test above 1 kg**.
    """)
    return


@app.cell
def _(X, make_models, np, pd, plt, r2_score, y):
    small = X[:, 0] < 3
    print(f'train: {small.sum()} mammals below 1 kg,  test: {(~small).sum()} above')
    elephant = np.log10([[3672000]])
    grid_1 = np.linspace(X.min(), 7, 400)[:, None]
    _fig, _axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    _rows = []
    for _ax, (_name, _model) in zip(_axes, make_models().items()):
        _model.fit(X[small], y[small])
        _rows.append({'model': _name, 'R2 above 1 kg': r2_score(y[~small], _model.predict(X[~small])), 'elephant (W)': 10 ** _model.predict(elephant)[0]})
        _ax.scatter(X[small], y[small], s=5, color='C0', label='train')
        _ax.scatter(X[~small], y[~small], s=5, color='C1', label='test')
        _ax.plot(grid_1, _model.predict(grid_1), 'r', lw=1.5)
        _ax.axvline(3, color='k', ls=':')
        _ax.set_title(_name)
        _ax.set_xlabel('log10 body mass (g)')
    _axes[0].set_ylabel('log10 metabolic rate (W)')
    _axes[0].legend()
    plt.tight_layout()
    print('measured elephant: 2336 W')
    pd.DataFrame(_rows).set_index('model')
    return elephant, grid_1, small


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ❓ What does each model do to the right of the dotted line? Who predicts the elephant?

    **Answer.**
    * **kNN and the random forest go flat.** Both predict by averaging training animals, so they can
      never return a value larger than the ones they have seen. For them an elephant is a big rat,
      about 3 W. R² is negative: worse than predicting the mean of the test set.
    * **The line keeps going** and predicts about 400 W. It is still 5 times too low, because the
      slope fitted on small mammals (0.63) is lower than the slope among large ones (remember the bend
      of session 1), but it is right within an order of magnitude, 3.5 orders of magnitude away from
      its training data.
    * **The neural network** continues along a straight line too: outside its training range a
      network of this kind (ReLU units) is linear. Here it lands near the law. Is that reliable?

    A neural network starts from random weights. Let us train it five times with different seeds.
    """)
    return


@app.cell
def _(X, elephant, grid_1, make_models, plt, small, y):
    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(X[small], y[small], s=5, color='C0')
    plt.scatter(X[~small], y[~small], s=5, color='C1')
    for seed in range(5):
        net = make_models(seed)['neural network'].fit(X[small], y[small])
        plt.plot(grid_1, net.predict(grid_1), lw=1.5, label=f'seed {seed}: elephant {10 ** net.predict(elephant)[0]:.0f} W')
    plt.axvline(3, color='k', ls=':')
    plt.xlabel('log10 body mass (g)')
    plt.ylabel('log10 metabolic rate (W)')
    plt.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ❓ Where do the five networks agree, and where not? What would you tell someone who used seed 0?

    **Answer.** Below 1 kg the five curves coincide: the data pin them down. Above, each one leaves
    in its own direction and the prediction for the elephant ranges from 16 W to 400 W. Nothing in
    the training data tells the network which direction is right, so the choice is made by the random
    initialisation. Seed 0 was luck.

    This is the difference between **interpolation** and **extrapolation**. Inside the data, any
    reasonable model works and the data decide. Outside, the answer comes from the *assumptions built
    into the model*: "nothing new happens" for a forest, "continue straight in log-log space" for the
    law. The law extrapolates because its assumption happens to be good physics, not because linear
    regression is a better algorithm.

    ✏️ **Exercise 2.1.** Do the reverse: train on mammals **above** 1 kg and predict the small ones.
    Compare the linear model, kNN and the random forest.
    """)
    return


@app.cell
def _(X, make_models, r2_score, small, y):
    for _name, _model in list(make_models().items())[:3]:
        _model.fit(X[~small], y[~small])
        print(f'{_name:15s} R2 below 1 kg = {r2_score(y[small], _model.predict(X[small])):.2f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Answer.** Same picture: kNN and the forest fail completely (R² about -2 to -3), the line still
    explains about half of the variance (R² 0.54). It is less good than before because the slope of
    large mammals (0.82) is too steep for small ones: the law is bent, and a straight line pays for
    it when it has to travel far.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 3 - Species are not independent

    Extrapolation is not only about size. In session 1 the lifespan of mammals was poorly predicted
    by mass alone (R² 0.50). Let us add three life-history traits and predict
    **maximum longevity** from mass, age at maturity, gestation time and litter size.
    """)
    return


@app.cell
def _(
    KNeighborsRegressor,
    LinearRegression,
    RNG,
    RandomForestRegressor,
    StandardScaler,
    anage,
    cross_val_score,
    cv,
    make_pipeline,
    np,
    pd,
):
    features = ["adult_mass_g", "female_maturity_days", "gestation_days", "litter_size"]
    life = anage[anage["class"] == "Mammalia"].dropna(subset=features + ["longevity_yr"])
    XL = np.log10(life[features].values)
    yL = np.log10(life["longevity_yr"].values)
    print(len(life), "mammals,", life["order"].nunique(), "orders,", life["family"].nunique(), "families")

    models3 = {
        "linear": LinearRegression(),
        "kNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
        "random forest": RandomForestRegressor(n_estimators=300, random_state=RNG),
    }
    random_cv = {name: cross_val_score(m, XL, yL, cv=cv, scoring="r2").mean() for name, m in models3.items()}
    pd.Series(random_cv, name="R2, random 5-fold CV").to_frame()
    return XL, life, models3, random_cv, yL


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With four features, R² rises to about 0.8, and here the flexible models do slightly better than
    the linear one.

    But what does "a species the model never saw" mean? With a random split, when the model is
    tested on the Norway rat it has trained on the black rat and on a hundred other rodents. Species
    are related by evolution: close relatives have similar bodies *and* similar lifespans, for
    reasons that our four features may not capture.

    A harder and often more relevant question: **how well do we predict a kind of mammal that is
    entirely new to the model?** `GroupKFold` keeps all species of the same **order** (rodents,
    bats, primates, ...) together, either in training or in test.
    """)
    return


@app.cell
def _(GroupKFold, XL, cross_val_score, life, models3, pd, random_cv, yL):
    orders = life["order"].values
    order_cv = {name: cross_val_score(m, XL, yL, cv=GroupKFold(n_splits=5), groups=orders, scoring="r2").mean()
                for name, m in models3.items()}
    table = pd.DataFrame({"random CV": random_cv, "orders held out": order_cv})
    table
    return (orders,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ❓ What happened? Which model loses most? Which number is "the" accuracy of the model?

    **Answer.** All models lose, and the flexible ones lose most: the random forest falls from 0.80
    to about 0.48, below the linear model (0.59). With random splits a flexible model can look up
    the close relatives of each test species; that is useful if your next species is another rodent,
    and useless if it is the first bat.

    Neither number is wrong. They answer different questions: "another species from the groups I
    know" and "a species from a group I have never seen". A published accuracy always describes
    some population; ask which one. We will meet the same problem in session 3 with proteins, where
    relatives are called *homologues*.

    ✏️ **Exercise 3.1.** Train the random forest on all mammals **except bats** (`Chiroptera`) and
    predict the bats. On average, are the predicted lifespans too long or too short, and by what
    factor? Connect this with Part 5 of session 1.
    """)
    return


@app.cell
def _(RNG, RandomForestRegressor, XL, np, orders, yL):
    bats = orders == "Chiroptera"
    rf = RandomForestRegressor(n_estimators=300, random_state=RNG).fit(XL[~bats], yL[~bats])
    pred = rf.predict(XL[bats])
    print(f"{bats.sum()} bats: predicted / true lifespan, geometric mean = {10 ** np.mean(pred - yL[bats]):.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Answer.** The predictions are too short, by about 20% on average (ratio 0.80). Bats live longer
    than other mammals with the same size and reproduction, and a model that has never seen a bat
    cannot know. In session 1 this was a residual; here it is an error on new data. It is the same
    fact seen from two sides.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 4 - A law that breaks

    A different kind of scaling: **how many genes fit in a genome?** The table lists the genome size
    (in megabases, Mb) and the number of annotated genes for about 13 800 prokaryotes (bacteria and
    archaea, complete genomes) and 4 700 eukaryotes (animals, plants, fungi, protists), from the NCBI
    genome reports.
    """)
    return


@app.cell
def _(DATA_DIR, RNG, pd):
    genomes = pd.read_csv(DATA_DIR / "genomes.tsv", sep="\t")
    print(genomes.groupby("domain").size())
    genomes.sample(5, random_state=RNG)
    return (genomes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ✏️ **Exercise 4.1.** Always look at the data first. Which eukaryotes have the most genes
    according to this table? Do you believe it?
    """)
    return


@app.cell
def _(genomes):
    genomes.sort_values("genes", ascending=False).head(10)[["group", "organism", "size_mb", "genes"]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Answer.** The Iberian lynx with 4.7 million genes, an olive tree with 2 million: these are
    errors in the submitted annotations (probably features miscounted as genes), not biology. For
    comparison, bread wheat, a famously gene-rich genome, has about 110 000. Public databases
    contain such records, and no model will warn you. We drop the 8 entries above 150 000 genes.
    """)
    return


@app.cell
def _(LinearRegression, genomes, np):
    genomes_1 = genomes[genomes['genes'] < 150000]
    pro = genomes_1[genomes_1['domain'] == 'prokaryote']
    euk = genomes_1[genomes_1['domain'] == 'eukaryote']
    Xp, yp = (np.log10(pro[['size_mb']].values), np.log10(pro['genes'].values))
    Xe, ye = (np.log10(euk[['size_mb']].values), np.log10(euk['genes'].values))
    law = LinearRegression().fit(Xp, yp)
    print(f'prokaryotes: genes = {10 ** law.intercept_:.0f} * size^{law.coef_[0]:.3f}     R2 = {law.score(Xp, yp):.3f}')
    print(f"median gene density: {(pro['genes'] / pro['size_mb']).median():.0f} genes per Mb")
    return Xe, Xp, euk, law, ye, yp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For prokaryotes the law is almost exact: the exponent is close to 1, so gene number is
    **proportional** to genome size, about one gene per 1 000 bases. A prokaryotic genome is genes
    packed side by side.

    The human genome has about 3 100 Mb. What does this law predict for us? We also ask a random
    forest trained on the same prokaryotes.
    """)
    return


@app.cell
def _(RNG, RandomForestRegressor, Xe, Xp, euk, law, np, plt, r2_score, ye, yp):
    forest = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=RNG).fit(Xp, yp)
    human = euk[euk['organism'] == 'Homo sapiens'].iloc[0]
    xh = np.log10([[human['size_mb']]])
    print(f"human, annotated          : {human['genes']:>10,.0f} genes")
    print(f'prokaryote law predicts   : {10 ** law.predict(xh)[0]:>10,.0f}')
    print(f'prokaryote forest predicts: {10 ** forest.predict(xh)[0]:>10,.0f}')
    grid_2 = np.linspace(Xp.min(), Xe.max(), 300)[:, None]
    plt.figure(figsize=(7, 4.8))
    plt.scatter(Xp, yp, s=3, alpha=0.3, label='prokaryotes')
    plt.scatter(Xe, ye, s=3, alpha=0.3, label='eukaryotes')
    plt.plot(grid_2, law.predict(grid_2), 'k', lw=1.5, label='law fitted on prokaryotes')
    plt.plot(grid_2, forest.predict(grid_2), 'r', lw=1.5, label='forest fitted on prokaryotes')
    plt.scatter(xh, np.log10(human['genes']), color='k', marker='*', s=150, zorder=5, label='human')
    plt.xlabel('log10 genome size (Mb)')
    plt.ylabel('log10 number of genes')
    plt.legend()
    print(f'\nR2 on eukaryotes: law {r2_score(ye, law.predict(Xe)):.1f}, forest {r2_score(ye, forest.predict(Xe)):.2f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ❓ The law that extrapolated so well in Part 2 now predicts 2.3 million human genes. The forest,
    which failed in Part 2, says 11 000, much closer to the truth (about 58 000 annotated genes, of
    which 20 000 code for proteins). Did the forest understand eukaryotes?

    **Answer.** No. The forest does here exactly what it did with the elephant: it repeats the
    largest value it has seen. This time "nothing new happens" is by chance closer to reality than
    "continue straight", because gene number nearly stops growing. Neither model had any
    information about eukaryotes, and neither gave any sign of being outside its domain. Extrapolation
    succeeds when the assumption of the model matches the world, and you cannot check that from
    inside the training data.

    What changed is the biology. Eukaryotic genomes are mostly *not* genes: introns, repeats and
    other non-coding DNA. The record holder in the table, a lungfish, has a genome of 40 000 Mb,
    13 times ours, with about 30 000 genes. Genome size tells little about gene number or about
    the complexity of the organism (the *C-value paradox*).

    ✏️ **Exercise 4.2.** Fit the exponent for eukaryotes as a whole, then separately for each `group`.
    Which group is closest to the prokaryotic behaviour, which is furthest?
    """)
    return


@app.cell
def _(Xe, euk, np, pd, ye):
    print(f'all eukaryotes: exponent {np.polyfit(Xe[:, 0], ye, 1)[0]:.2f}, R2 = {np.corrcoef(Xe[:, 0], ye)[0, 1] ** 2:.2f}\n')
    _rows = []
    for group, d in euk.groupby('group'):
        xg, yg = (np.log10(d['size_mb']), np.log10(d['genes']))
        _rows.append({'group': group, 'n': len(d), 'median size (Mb)': d['size_mb'].median(), 'exponent': np.polyfit(xg, yg, 1)[0], 'R2': np.corrcoef(xg, yg)[0, 1] ** 2})
    pd.DataFrame(_rows).set_index('group')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Answer.** For eukaryotes as a whole the exponent is 0.28 with R² 0.60: a genome 10 times larger
    has only about twice the genes. Fungi and protists, with small compact genomes, keep an exponent
    near 0.6 and a fairly tight relation. In animals the exponent is 0.2 and R² only 0.2: genome size
    says almost nothing about gene number. The proportionality law holds where genomes are compact
    and fades as non-coding DNA takes over.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    * **Inside the data, models agree.** A line, kNN, a random forest and a neural network all reach
      R² ≈ 0.95 on metabolic rate. Cross-validation cannot separate them.
    * **Outside the data, assumptions decide.** Averaging models go flat, neural networks leave in a
      direction set by their random seed, a law continues its trend. The law was right for the
      elephant and wrong for the human genome.
    * **"Unseen" has degrees.** A random split tests on close relatives of the training species;
      holding out whole orders cuts the R² of the random forest from 0.80 to 0.48.
    * **No score computed on your data tells you where the model stops being valid.** That
      takes knowledge of the system, or new data from outside.

    In session 3 we turn to protein sequences, where no two-parameter law exists and a neural network
    earns its place. The same questions come with us: what is the simple baseline, and what does
    "unseen" mean?
    """)
    return


if __name__ == "__main__":
    app.run()

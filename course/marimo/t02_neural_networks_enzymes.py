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
    # Session 2 - Neural networks, and classifying enzymes

    **AI in Quantitative Biology - practical 2 of 3 (90 min)**

    Two parts:

    | Part | Topic | Time |
    |---|---|---|
    | A | A small neural network on the fish toxicity data (regression) | 35 min |
    | B | Can we tell an enzyme's class from its amino-acid sequence? (classification) | 50 min |

    Part B introduces a trap that is everywhere in biological machine learning: **related sequences
    leaking between training and test set**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    Everything runs on a laptop CPU in a few minutes; no GPU needed. The data are in the `data/`
    folder of the course repository. The cell below finds it in the folders above
    the notebook.
    """)
    return


@app.cell(hide_code=True)
def _():
    # @title Setup: imports, data loading and helpers (run me)
    import sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    RNG = 0                                         # fixed seed: everyone gets the same numbers
    pd.set_option("display.precision", 3)

    import os
    from pathlib import Path

    def find_data_dir():
        """$QBIO_DATA if set, else the `data/` folder of the course repository (here or above)."""
        if os.environ.get("QBIO_DATA"):
            return Path(os.environ["QBIO_DATA"])
        here = Path.cwd().resolve()
        for p in [here, *here.parents]:
            if (p / "data" / "qsar_fish_toxicity.csv").exists():
                return p / "data"
        raise FileNotFoundError("Cannot find data/: run the notebook inside the course repository, "
                                "or set QBIO_DATA to the path of its data/ folder")

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
        ax = plt.subplots(figsize=(len(labels) + 2, len(labels) + 1))[1]
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

    return (
        AA,
        RNG,
        aa_composition,
        load_enzymes,
        load_fish,
        np,
        pd,
        plot_confusion,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part A - A neural network for toxicity

    A **multi-layer perceptron (MLP)** is a stack of layers. Each *neuron* computes a weighted sum of its
    inputs and passes it through a non-linear *activation* (here ReLU: $\max(0, z)$):

    $$ h_j = \mathrm{ReLU}\Big(\sum_i w_{ji} x_i + b_j\Big), \qquad \hat y = \sum_j v_j h_j + c $$

    The weights $w, v$ and biases $b, c$ start random and are adjusted step by step (gradient descent)
    to reduce the squared error on the training data. With enough hidden neurons an MLP can
    approximate any smooth function - which is both its strength and its danger.

    ```
     6 descriptors  ->  [32 hidden neurons, ReLU]  ->  1 output (LC50)
    ```
    """)
    return


@app.cell
def _(RNG, load_fish):
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
    return (
        KFold,
        MLPRegressor,
        RandomForestRegressor,
        StandardScaler,
        X,
        X_test,
        X_train,
        cross_val_score,
        make_pipeline,
        mlp,
        r2_score,
        y,
        y_test,
        y_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The network was trained iteratively. `loss_curve_` stores the training loss after each pass over
    the data (*epoch*).
    """)
    return


@app.cell
def _(mlp, plt):
    net = mlp[-1]
    plt.plot(net.loss_curve_)
    plt.yscale("log")
    plt.xlabel("epoch"); plt.ylabel("training loss (squared error / 2)")
    plt.title(f"{net.n_iter_} epochs");
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ✏️ **Exercise A.1.** Compare the MLP with the random forest of session 1 using the same 5-fold
    cross-validation (`cv` below). Which one wins?
    """)
    return


@app.cell
def _(KFold, RNG, RandomForestRegressor, X, cross_val_score, mlp, y):
    cv = KFold(n_splits=5, shuffle=True, random_state=RNG)
    _rf = RandomForestRegressor(n_estimators=300, random_state=RNG, n_jobs=-1)
    r2_mlp = cross_val_score(mlp, X, y, cv=cv, scoring='r2')
    r2_rf = cross_val_score(_rf, X, y, cv=cv, scoring='r2')
    print(f'MLP R2 = {r2_mlp.mean():.3f}   RF R2 = {r2_rf.mean():.3f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What function did each model learn?

    The models live in 6 dimensions, so we cannot draw them. We can take a **slice**: vary `MLOGP` and
    keep the other five descriptors fixed at their median. Each curve is the predicted LC50 along that slice.
    """)
    return


@app.cell(hide_code=True)
def _(
    MLPRegressor,
    RNG,
    RandomForestRegressor,
    StandardScaler,
    X,
    X_train,
    make_pipeline,
    mlp,
    np,
    pd,
    plt,
    y_train,
):
    # @title Support code: the function each model learned (figure)
    from sklearn.linear_model import LinearRegression
    line = pd.DataFrame(np.tile(X_train.median().values, (300, 1)), columns=X.columns)
    line['MLOGP'] = np.linspace(X['MLOGP'].min() - 1, X['MLOGP'].max() + 1, 300)
    wide_mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(512, 512), alpha=0.0001, max_iter=2000, random_state=RNG))
    fitted = {'linear regression': LinearRegression(), 'random forest': RandomForestRegressor(n_estimators=300, random_state=RNG, n_jobs=-1), 'MLP (32)': mlp, 'MLP (512, 512), no regularisation': wide_mlp}
    plt.figure(figsize=(8, 5))
    plt.scatter(X_train['MLOGP'], y_train, s=6, c='lightgrey', label='training chemicals')
    for _name, m in fitted.items():
        if m is not mlp:
            m.fit(X_train, y_train)
        plt.plot(line['MLOGP'], m.predict(line), lw=2, label=_name)
    plt.axvspan(line['MLOGP'].min(), X['MLOGP'].min(), color='k', alpha=0.07)  # already trained above
    plt.axvspan(X['MLOGP'].max(), line['MLOGP'].max(), color='k', alpha=0.07, label='outside the data')
    plt.xlabel('MLOGP (other descriptors at their median)')
    plt.ylabel('predicted LC50')
    plt.legend(fontsize=8)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ❓ Describe the shape of each curve. What happens outside the range of the data (grey bands)?

    **Answer.** Linear regression is a straight line everywhere. The random forest is a **staircase**:
    each tree cuts the axis into intervals and predicts a constant in each, and beyond the data it goes
    **flat** (it can never predict outside the range of the training LC50). The MLP is made of ReLUs,
    so it is **piecewise linear**: smooth bends inside the data, then it continues as a straight line
    outside, extrapolating in whatever direction the last piece pointed. Both MLPs bend **upwards** at low MLOGP,
    predicting that very hydrophilic chemicals are *more* toxic: there are few training chemicals there,
    the curve is not constrained, and the result is biologically implausible. The big unregularised
    network bends more and has small wiggles where it follows individual chemicals: overfitting, seen directly.
    None of the models can be trusted in the grey bands; QSAR guidelines call the region where a
    model is valid its **applicability domain**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How big should the network be?

    We train networks with two hidden layers of increasing width and compare the R² on the
    **training** set with the R² on the **test** set.
    """)
    return


@app.cell
def _(
    MLPRegressor,
    RNG,
    StandardScaler,
    X_test,
    X_train,
    make_pipeline,
    pd,
    plt,
    r2_score,
    y_test,
    y_train,
):
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
    plt.show()
    sweep
    return size_sweep, sweep


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ❓ What happens to the gap between training and test R² as the network grows? What is this called?

    ✏️ **Exercise A.2.** `alpha` adds a penalty on large weights (*L2 regularisation*, "weight decay").
    Run `size_sweep(alpha=1.0)` and compare with the plot above.
    """)
    return


@app.cell
def _(plt, size_sweep, sweep):
    sweep_reg = size_sweep(alpha=1.0)
    _ax = sweep.plot(logx=True, marker='o', style='--', alpha=0.5)
    sweep_reg.plot(logx=True, marker='o', ax=_ax)
    plt.legend(['train, alpha=1e-4', 'test, alpha=1e-4', 'train, alpha=1', 'test, alpha=1'])
    plt.ylabel('R2')
    plt.show()
    sweep_reg
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Answer.** Without regularisation, the bigger networks fit the training set better and better
    (train R² up to ≈ 0.93) while the test R² does not improve: this is **overfitting** - the network
    memorises the 681 training chemicals, noise included. With `alpha=1.0` the training R² stays
    lower (≈ 0.8), the gap shrinks, and the large networks generalise best (test R² ≈ 0.55).
    On a small table of 6 descriptors, an MLP does **not** beat a random forest (≈ 0.58 vs ≈ 0.60
    in cross-validation). Neural networks shine with large data and with raw, structured inputs such as
    sequences and images - which is where we go next.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Overfitting, epoch by epoch

    `loss_curve_` only shows the *training* loss, which always goes down. Let us train the big
    (512, 512) network one epoch at a time (`partial_fit`) and measure the error on the test chemicals
    after each epoch as well.
    """)
    return


@app.cell
def _(
    MLPRegressor,
    RNG,
    StandardScaler,
    X_test,
    X_train,
    np,
    plt,
    y_test,
    y_train,
):
    from sklearn.metrics import mean_squared_error
    scaler = StandardScaler().fit(X_train)
    Xtr_s, Xte_s = (scaler.transform(X_train), scaler.transform(X_test))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for _ax, alpha in zip(axes, [0.0001, 1.0]):
        net_e = MLPRegressor(hidden_layer_sizes=(512, 512), alpha=alpha, random_state=RNG)
        tr_mse, te_mse = ([], [])
        for epoch in range(300):
            net_e.partial_fit(Xtr_s, y_train)
            tr_mse.append(mean_squared_error(y_train, net_e.predict(Xtr_s)))
            te_mse.append(mean_squared_error(y_test, net_e.predict(Xte_s)))  # one more pass over the training data
        best = int(np.argmin(te_mse))
        _ax.plot(tr_mse, label='training set')
        _ax.plot(te_mse, label='test set')
        _ax.axvline(best, color='grey', ls=':', label=f'lowest test error (epoch {best + 1})')
        _ax.set_yscale('log')
        _ax.set_xlabel('epoch')
        _ax.set_title(f'MLP (512, 512), alpha = {alpha:g}')
        print(f'alpha={alpha:g}: best test MSE {min(te_mse):.3f} at epoch {best + 1}, after 300 epochs train {tr_mse[-1]:.3f} / test {te_mse[-1]:.3f}')
    axes[0].set_ylabel('mean squared error')
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ❓ When should we have stopped training the unregularised network? What does regularisation change?

    **Answer.** Without regularisation the training error keeps falling towards zero, while the test
    error reaches its minimum (here around epoch 110, MSE ≈ 0.86) and then **climbs** again (≈ 1.0 after
    300 epochs, while the training MSE is down to 0.14): from there on, every epoch spent memorising the training chemicals makes predictions on new ones worse. Stopping at the
    minimum is called **early stopping**. Since we must not choose it by looking at the test set, it is
    done on a validation set carved out of the training data (`MLPRegressor(early_stopping=True)`; we use
    it for the enzyme classifier below). With `alpha = 1` the training error falls much more slowly and the test error
    flattens out at ≈ 0.87 instead of rising: the penalty on large weights stops the network from memorising, whenever we stop.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Playground: a neural network on a curve you can see

    The same phenomena are much easier to *see* in one dimension. We use a toy problem where the true
    curve is known: a few noisy "measurements" of a dose-response-like curve. `mlp_playground()` trains an
    MLP on them one epoch at a time and draws:
    * **left**: the network's curve at several moments during training (light = early, dark = late);
    * **right**: training and test error at every epoch. The dotted line is the noise level (the error of
      the true curve itself): no model can do better than that on new data.

    Parameters: `width` and `layers` (network size), `alpha` (L2 penalty), `n_train` (number of points),
    `noise`, `epochs`, `seed`.
    """)
    return


@app.cell(hide_code=True)
def _(MLPRegressor, np, plt):
    # @title Support code: the MLP playground
    def true_curve(x):
        """The 'truth' we pretend not to know: a smooth dose-response-like curve."""
        return 1 / (1 + np.exp(-15 * (x - 0.45))) + 0.25 * np.sin(10 * x)

    def make_toy(n, noise, seed):
        """n noisy measurements of true_curve at random x in [0, 1]."""
        rng = np.random.default_rng(seed)
        x = np.sort(rng.uniform(0, 1, n))
        return x, true_curve(x) + rng.normal(0, noise, n)

    from sklearn.exceptions import ConvergenceWarning
    import warnings

    def mlp_playground(width=200, layers=2, alpha=1e-5, n_train=20, noise=0.2, epochs=3000, seed=0):
        x_tr, y_tr = make_toy(n_train, noise, seed)
        x_te, y_te = make_toy(500, noise, seed + 1000)
        grid = np.linspace(0, 1, 400)[:, None]
        net = MLPRegressor(hidden_layer_sizes=(width,) * layers, alpha=alpha,
                           learning_rate_init=0.003, random_state=seed)
        snap_epochs = sorted({int(e) for e in np.geomspace(10, epochs, 5)})
        tr_mse, te_mse, snaps = [], [], {}
        for epoch in range(1, epochs + 1):
            net.partial_fit(x_tr[:, None], y_tr)
            tr_mse.append(np.mean((net.predict(x_tr[:, None]) - y_tr) ** 2))
            te_mse.append(np.mean((net.predict(x_te[:, None]) - y_te) ** 2))
            if epoch in snap_epochs:
                snaps[epoch] = net.predict(grid)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
        ax1.plot(grid, true_curve(grid), "k--", lw=1.5, label="true curve")
        for i, (e, curve) in enumerate(snaps.items()):
            ax1.plot(grid, curve, color=plt.cm.Reds(0.3 + 0.7 * i / max(1, len(snaps) - 1)), lw=2, label=f"epoch {e}")
        ax1.scatter(x_tr, y_tr, color="k", s=20, zorder=3, label=f"{n_train} training points")
        ax1.set_ylim(-1, 2); ax1.set_xlabel("dose"); ax1.set_ylabel("response"); ax1.legend(fontsize=7, loc="upper left")
        ax1.set_title(f"MLP {(width,) * layers}, alpha = {alpha:g}")

        best = int(np.argmin(te_mse))
        ax2.plot(range(1, epochs + 1), tr_mse, label="training points")
        ax2.plot(range(1, epochs + 1), te_mse, label="new points (test)")
        ax2.axhline(noise ** 2, color="grey", ls=":", label="noise level")
        ax2.axvline(best + 1, color="grey", alpha=0.5, label=f"best epoch ({best + 1})")
        ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.set_xlabel("epoch"); ax2.set_ylabel("MSE")
        ax2.set_title(f"final: train {tr_mse[-1]:.3f}, test {te_mse[-1]:.3f} (best test {te_mse[best]:.3f})")
        ax2.legend(fontsize=8)
        plt.tight_layout(); plt.show()

    mlp_playground()
    plt.show()
    mlp_playground(alpha=0.1)
    return (mlp_playground,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Sliders:* set the sliders below and press **Submit** to redraw the figure. You can also call the function directly with other values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # @title Sliders for the playground
    mlp_controls = mo.ui.dictionary({
        'width': mo.ui.slider(2, 500, step=2, value=200, label='width'),
        'layers': mo.ui.slider(1, 4, value=2, label='layers'),
        'alpha': mo.ui.slider(steps=[1e-06, 2e-06, 5e-06, 1e-05, 2e-05, 5e-05, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0], value=1e-05, label='alpha'),
        'n_train': mo.ui.slider(10, 300, step=5, value=20, label='n_train'),
        'noise': mo.ui.slider(0.0, 0.6, step=0.05, value=0.2, label='noise'),
        'epochs': mo.ui.slider(100, 6000, step=100, value=3000, label='epochs'),
        'seed': mo.ui.slider(0, 20, value=0, label='seed'),
    }).form(submit_button_label='Submit')
    mlp_controls
    return (mlp_controls,)


@app.cell(hide_code=True)
def _(mlp_controls, mlp_playground):
    # @title Sliders for the playground
    if mlp_controls.value is not None:
        mlp_playground(**mlp_controls.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ✏️ **Exercise A.3.** Use the playground to answer:
    1. In the first figure, in what order does the network learn things? When does the test error start
       to rise?
    2. What changes with `alpha=0.1` (second figure)? And with `alpha=1`?
    3. Without regularisation, which helps more against overfitting: a smaller network (`width=8`) or
       more data (`n_train=200`)?
    4. Try `seed=1, 2, 3` with the default settings. Do you always get the same curve?

    **Answer.**
    1. First the **coarse trend** (epoch 10: almost a straight line), then the S shape (≈ epoch 170), and
       finally the details of individual points: by epoch 3000 the curve dips too deep around dose 0.35
       and keeps rising at the right edge where the true curve comes down. The test error is lowest at
       epoch ≈ 380 (MSE 0.048, close to the noise level 0.04) and then rises to 0.061, while the training
       error keeps falling to 0.007, far *below* the noise level: the network is memorising noise.
       Early stopping at epoch ≈ 380 would keep the best network.
    2. With `alpha=0.1` the weights cannot grow large enough for sharp bends: the curve stays smooth
       (it even smooths away the small bump at dose 0.2), and the test error keeps going *down* until the
       end (0.049): no early stopping needed. With `alpha=1` the penalty is too strong and the curve is
       too stiff to follow the S shape (test MSE ≈ 0.09): underfitting.
    3. Both work here. `width=8` cannot memorise 20 points and ends at test MSE 0.048, like the best epoch
       of the big network. `n_train=200` reaches the noise level (0.040) with training and test error equal.
       More data is the more reliable cure: a network that is too small can also miss real structure.
    4. No. Different random initial weights and a different sample of points give different curves,
       especially between and outside the points, and the best epoch moves too (≈ 135-380 for seeds 0-3).
       Neural networks do not have a unique solution; averaging several of them (an **ensemble**) is a
       common way to reduce this variance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part B - Can we tell an enzyme's class from its sequence?

    Enzymes are classified by the reaction they catalyse into **EC (Enzyme Commission) classes**:

    | EC | class | reaction |
    |---|---|---|
    | 1 | Oxidoreductases | transfer of electrons (redox) |
    | 2 | Transferases | transfer of a group (methyl, phosphate, ...) |
    | 3 | Hydrolases | cleavage by water |
    | 4 | Lyases | cleavage without water or oxidation |
    | 5 | Isomerases | rearrangement within a molecule |
    | 6 | Ligases | joining two molecules, using ATP |

    Our dataset contains **2047 enzymes** from UniProt/Swiss-Prot (manually reviewed), from five model
    organisms (human, *E. coli*, baker's yeast, *Arabidopsis*, *B. subtilis*), about 350 per class.
    Each enzyme also has its **protein family** (groups of evolutionarily related proteins).

    ## B1 - Meet the data
    """)
    return


@app.cell
def _(load_enzymes):
    enz = load_enzymes()
    print(enz.shape)
    enz.head()
    return (enz,)


@app.cell
def _(enz, pd):
    print(enz["ec_name"].value_counts(), "\n")
    pd.crosstab(enz["organism"], enz["ec_name"])
    return


@app.cell
def _(enz, plt):
    classes = sorted(enz["ec_name"].unique())
    enz.boxplot(column="length", by="ec_name", figsize=(8, 4), rot=30)
    plt.suptitle(""); plt.title("sequence length by class"); plt.ylabel("residues");
    plt.show()
    return (classes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ✏️ **Exercise B1.1.** Print one example sequence and its protein name. How many proteins come from
    the largest protein family, and what is that family? *Hint:* `value_counts()` on `family`.
    """)
    return


@app.cell
def _(enz):
    row = enz.iloc[0]
    print(row["protein_name"], "|", row["organism"], "| EC", row["ec"])
    print(row["sequence"])
    print()
    print(enz["family"].value_counts().head(5))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B2 - Amino-acid composition

    The simplest way to turn a sequence of any length into a fixed-size vector: the **fraction of
    each of the 20 amino acids**. This forgets the order completely.
    """)
    return


@app.cell
def _(AA, aa_composition, enz, np, pd):
    comp = pd.DataFrame(np.vstack(enz["sequence"].map(aa_composition)), columns=list(AA))
    comp.head()
    return (comp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Is any amino acid enriched in a class? We compare each class's mean composition with the overall
    mean, on a $\log_2$ scale (+1 = twice as frequent).
    """)
    return


@app.cell
def _(AA, classes, comp, enz, np, plt):
    class_mean = comp.groupby(enz["ec_name"]).mean()
    enrichment = np.log2(class_mean / comp.mean())

    plt.figure(figsize=(10, 3.5))
    plt.imshow(enrichment, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    plt.colorbar(label="log2 enrichment")
    plt.xticks(range(20), AA); plt.yticks(range(len(classes)), enrichment.index);
    plt.show()
    return


@app.cell
def _(StandardScaler, classes, comp, enz, plt):
    from sklearn.decomposition import PCA

    Z = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(comp))
    plt.figure(figsize=(6, 5))
    for c in classes:
        sel = (enz["ec_name"] == c).values
        plt.scatter(Z[sel, 0], Z[sel, 1], s=6, alpha=0.5, label=c)
    plt.legend(markerscale=3, fontsize=8); plt.xlabel("PC1"); plt.ylabel("PC2");
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ❓ How large are the differences in composition between classes? Do the classes separate in the
    PCA map? What do you expect for a classifier?

    **Answer.** The enrichments are tiny (within ±0.25 on a log2 scale, i.e. < 20%) and the classes
    overlap completely in the PCA map. Composition carries only a faint signal about the catalysed
    reaction: expect a classifier to do better than chance, but not much. (Some enrichments have a
    story: Cys is enriched among transferases because ~15% of them transfer ubiquitin (EC 2.3.2.x,
    e.g. RING-type E3 ligases), whose zinc-binding domains contain twice the average Cys.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B3 - Classifiers

    Features: the 20 composition values plus log(length). We measure **accuracy** (fraction of
    correct predictions) with stratified 5-fold cross-validation (each fold keeps the class
    proportions). With 6 balanced classes, **random guessing gives about 1/6 = 0.17**.
    """)
    return


@app.cell
def _(RNG, StandardScaler, comp, cross_val_score, enz, make_pipeline, np):
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    Xe = comp.assign(log_length=np.log(enz['length']))
    ye = enz['ec_name']
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    classifiers = {'logistic regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), 'random forest': RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1), 'MLP (64)': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), alpha=0.001, early_stopping=True, max_iter=500, random_state=RNG))}
    acc_random = {}
    for _name, _clf in classifiers.items():
        acc_random[_name] = cross_val_score(_clf, Xe, ye, cv=skf, scoring='accuracy').mean()
        print(f'{_name:20s} accuracy = {acc_random[_name]:.3f}')
    return Xe, acc_random, classifiers, cross_val_predict, skf, ye


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Accuracy hides *which* classes get confused. A **confusion matrix** shows, for each true class (row),
    how the predictions are distributed (columns). `cross_val_predict` gives one out-of-fold prediction
    per enzyme.
    """)
    return


@app.cell
def _(Xe, classes, classifiers, cross_val_predict, plot_confusion, plt, skf, ye):
    pred_rf = cross_val_predict(classifiers["random forest"], Xe, ye, cv=skf)
    plot_confusion(ye, pred_rf, classes, title="random forest, random 5-fold CV");
    plt.show()
    return (pred_rf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ❓ Which class is easiest to recognise, which is hardest? Any idea why?

    **Answer.** Look at the diagonal: no class is easy (recall ≈ 0.30-0.45 each) and the errors
    spread over all columns. Oxidoreductases are the hardest: a huge and chemically diverse class
    (NAD-, FAD-, heme-, metal-dependent enzymes in many unrelated folds) with no common composition.
    EC classes describe the *chemistry* of the reaction, which evolution has invented many times in
    unrelated folds, so composition is a weak proxy for it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B4 - The homology trap

    Proteins of the same **family** share a common ancestor and often have similar sequences - and
    similar compositions. With a random split, a test enzyme usually has a close relative
    (e.g. the same enzyme from another organism) in the training set, and the model can just
    "recognise the family" instead of learning anything general.

    A fairer test for a *new* kind of enzyme: keep **whole families** out of the training set.
    `StratifiedGroupKFold` does this - all members of a group end up in the same fold.
    """)
    return


@app.cell
def _(RNG, enz):
    from sklearn.model_selection import StratifiedGroupKFold

    print(enz["family"].nunique(), "families;", (enz["family"].value_counts() > 1).sum(), "with more than one member")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RNG)
    groups = enz["family"]
    return groups, sgkf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ✏️ **Exercise B4.1.** Repeat the cross-validation of the three classifiers with `cv=sgkf` and
    `groups=groups` (pass it as `cross_val_score(..., groups=groups)`), and print both accuracies
    side by side.
    """)
    return


@app.cell
def _(Xe, acc_random, classifiers, cross_val_score, groups, sgkf, ye):
    for _name, _clf in classifiers.items():
        acc_family = cross_val_score(_clf, Xe, ye, cv=sgkf, groups=groups, scoring='accuracy').mean()
        print(f'{_name:20s} random split = {acc_random[_name]:.3f}   family split = {acc_family:.3f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ❓ Which model loses the most when families are held out, and why? Which number would you report
    in a paper claiming "our model predicts the class of newly discovered enzymes"?

    **Answer.** The random forest drops the most (≈ 0.40 → 0.30): it is the most flexible model and
    partly works as a "find the relative" lookup table. Logistic regression and the MLP drop less
    because they were learning less family-specific detail in the first place. For new enzymes, the
    family-split number is the honest one. The random-split number answers a different question
    ("can I annotate a new member of a family I already know?") - for which a BLAST search against
    annotated sequences would beat all of these models.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Where exactly does the inflated random-split accuracy come from? For each enzyme, count how many
    members of its family were in the **training folds** when it was predicted (in the random 5-fold CV
    of B3), and compare the accuracy of the random forest across these groups.
    """)
    return


@app.cell(hide_code=True)
def _(Xe, acc_random, enz, np, pd, plt, pred_rf, skf, ye):
    # @title Support code: accuracy vs number of relatives in training (figure)
    fold = np.zeros(len(enz), dtype=int)
    for i, (_, test_idx) in enumerate(skf.split(Xe, ye)):
        fold[test_idx] = i
    fam = enz['family'].fillna('').values
    n_relatives = np.array([0 if fam[j] == '' else ((fam == fam[j]) & (fold != fold[j])).sum() for j in range(len(enz))])
    correct = pred_rf == ye.values
    bins = pd.cut(n_relatives, [-1, 0, 1, 3, 100], labels=['0', '1', '2-3', '4+'])
    by_rel = pd.DataFrame({'relatives in training': bins, 'correct': correct}).groupby('relatives in training', observed=True)['correct'].agg(['mean', 'size'])
    print(by_rel)
    _ax = by_rel['mean'].plot.bar(rot=0, color='tab:blue', figsize=(6, 4), label='accuracy')
    for i, (acc, n) in enumerate(zip(by_rel['mean'], by_rel['size'])):
        _ax.text(i, acc + 0.01, f'n = {n}', ha='center', fontsize=8)
    _ax.axhline(1 / 6, color='grey', ls=':', label='chance')
    _ax.axhline(acc_random['random forest'], color='k', ls='--', label='overall (random split)')
    _ax.set_ylabel('random forest accuracy')
    _ax.set_xlabel('family members in the training folds')
    _ax.legend(fontsize=8)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ❓ What does this plot say about the random-split accuracy?

    **Answer.** Accuracy grows with the number of relatives in the training set. Enzymes with no relative
    in training get ≈ 0.33, close to the family-split number (≈ 0.31), while those with 4 or more
    relatives get ≈ 0.55: the random forest is largely recognising the family.
    The random-split accuracy is an average over these groups, so it depends on how many relatives the
    dataset happens to contain, not only on how good the model is.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    * An MLP is a flexible function approximator; on a small table it overfits easily, needs scaled
      inputs and regularisation, and does not beat a random forest. Watching the **test loss epoch by
      epoch** shows when it starts memorising (early stopping).
    * **Data mining before modelling** told us what to expect: the enzyme classes barely differ in
      composition, and models got 0.25-0.40 accuracy (chance: 0.17).
    * **How you split matters as much as which model you use.** Related sequences across the
      train/test boundary inflate the scores; split by family (or by sequence identity) to estimate
      performance on genuinely new proteins.
    * Composition throws away the **order** of the residues. Next session: a neural network that reads
      the sequence residue by residue.

    ### If you have time

    ✏️ Replace composition by **dipeptide** frequencies (400 features: AA, AC, ..., YY) and redo the
    random vs family comparison for the random forest. Does the gap grow or shrink?
    """)
    return


@app.cell
def _(AA, classifiers, cross_val_score, enz, groups, np, sgkf, skf, ye):
    import itertools
    DIPEPTIDES = [''.join(p) for p in itertools.product(AA, repeat=2)]
    index = {dp: i for i, dp in enumerate(DIPEPTIDES)}

    def dipeptide_composition(seq):
        v = np.zeros(len(DIPEPTIDES))
        for i in range(len(seq) - 1):
            v[index[seq[i:i + 2]]] += 1
        return v / (len(seq) - 1)
    Xd = np.vstack(enz['sequence'].map(dipeptide_composition))
    _rf = classifiers['random forest']
    print(f'random split: {cross_val_score(_rf, Xd, ye, cv=skf).mean():.3f}')
    print(f'family split: {cross_val_score(_rf, Xd, ye, cv=sgkf, groups=groups).mean():.3f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Answer.** Dipeptides do not help (≈ 0.38 random, ≈ 0.28 family split, vs 0.40 / 0.31 with
    composition): 400 noisy features for 2047 proteins is a lot, and the leakage gap stays ≈ 0.09.
    More features is not the same as more information.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References and data sources

    **Data**

    * `qsar_fish_toxicity.csv`: 908 compounds, 96-h LC50 for the fathead minnow (*Pimephales promelas*) with six molecular descriptors (UCI Machine Learning Repository, "QSAR fish toxicity"; Cassotti et al. 2015).
    * Enzyme and targeting tables: reviewed entries of UniProtKB/Swiss-Prot (UniProt Consortium 2023), downloaded with the UniProt REST API on 2026-11-10 from human, *E. coli* K-12, yeast, *Arabidopsis* and *B. subtilis*; Swiss-Prot changes at every release, so the committed TSVs are the reference. See `course/README.md` and `data_prep/`.

    **Further reading**

    * Kapoor S, Narayanan A (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns* 4:100804. [doi:10.1016/j.patter.2023.100804](https://doi.org/10.1016/j.patter.2023.100804) — the vocabulary for the leakage in Part B
    * Yu T, Cui H, Li JC, et al. (2023). Enzyme function prediction using contrastive learning. *Science* 379:1358–1363. [doi:10.1126/science.adf2465](https://doi.org/10.1126/science.adf2465) — CLEAN, a recent EC-number predictor: composition alone genuinely cannot do this task
    * Lin Z, Akin H, Rao R, et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 379:1123–1130. [doi:10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574) — the protein language model used in extra03
    """)
    return


if __name__ == "__main__":
    app.run()

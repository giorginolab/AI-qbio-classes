"""Shared helpers for the 'AI in Quantitative Biology' ML practical.

Keeps the notebooks about machine learning rather than about file paths.

Data: two QSAR datasets from the UCI Machine Learning Repository
(Cassotti, Ballabio, Todeschini, Consonni / Milano Chemometrics group).

  qsar_fish_toxicity.csv    908 molecules, 6 descriptors
                            target: LC50 toward Pimephales promelas
                            (fathead minnow), 96 h, as -log10(mol/L)

  qsar_aquatic_toxicity.csv 546 molecules, 8 descriptors
                            target: LC50 toward Daphnia magna, 48 h,
                            as -log10(mol/L)

Both files are semicolon-separated and have NO header row.
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd

FISH_COLUMNS = ["CIC0", "SM1_Dz(Z)", "GATS1i", "NdsCH", "NdssC", "MLOGP", "LC50"]
AQUA_COLUMNS = ["TPSA(Tot)", "SAacc", "H-050", "MLOGP", "RDCHI", "GATS1p", "nN", "C-040", "LC50"]

#: One-line reminder of what each descriptor is, for axis labels and discussion.
DESCRIPTOR_NOTES = {
    "CIC0": "complementary information content, order 0 (molecular complexity / symmetry)",
    "SM1_Dz(Z)": "spectral moment from the Barysz matrix weighted by atomic number",
    "GATS1i": "Geary autocorrelation, lag 1, weighted by ionisation potential",
    "NdsCH": "count of =CH- fragments",
    "NdssC": "count of =C< fragments",
    "MLOGP": "Moriguchi octanol-water partition coefficient (lipophilicity)",
    "TPSA(Tot)": "topological polar surface area (using N, O, S, P)",
    "SAacc": "surface area of hydrogen-bond acceptor atoms",
    "H-050": "count of H attached to a heteroatom",
    "RDCHI": "reciprocal distance sum / Randic-type connectivity index",
    "GATS1p": "Geary autocorrelation, lag 1, weighted by polarisability",
    "nN": "number of nitrogen atoms",
    "C-040": "count of R-C(=X)-X / R-C#X / X=C=X carbon fragments",
}

TARGET = "LC50"
TARGET_LABEL = "LC50  [-log10(mol/L)]"


def find_data_dir(start=None) -> Path:
    """Locate the folder holding the two QSAR CSVs.

    Looks at $QSAR_DATA first, then walks up from `start` (default: cwd)
    checking ./data and . at each level. Raises FileNotFoundError listing the
    places it looked, so a wrong working directory is obvious immediately.
    """
    env = os.environ.get("QSAR_DATA")
    tried = []
    if env:
        p = Path(env).expanduser()
        tried.append(p)
        if (p / "qsar_fish_toxicity.csv").exists():
            return p
    here = Path(start) if start is not None else Path.cwd()
    for parent in [here, *here.parents][:6]:
        for cand in (parent / "data", parent):
            tried.append(cand)
            if (cand / "qsar_fish_toxicity.csv").exists():
                return cand
    raise FileNotFoundError(
        "Could not find qsar_fish_toxicity.csv. Looked in:\n  "
        + "\n  ".join(str(t) for t in tried)
        + "\nSet the QSAR_DATA environment variable to the folder holding the CSVs."
    )


def _load(filename, columns, data_dir=None) -> pd.DataFrame:
    d = Path(data_dir) if data_dir is not None else find_data_dir()
    df = pd.read_csv(d / filename, sep=";", header=None, names=columns)
    if df.isna().to_numpy().any():
        raise ValueError(f"{filename} parsed with missing values - check the separator")
    return df


def load_fish(data_dir=None) -> pd.DataFrame:
    """908 x 7 DataFrame: 6 descriptors + LC50 (fathead minnow)."""
    return _load("qsar_fish_toxicity.csv", FISH_COLUMNS, data_dir)


def load_aquatic(data_dir=None) -> pd.DataFrame:
    """546 x 9 DataFrame: 8 descriptors + LC50 (Daphnia magna)."""
    return _load("qsar_aquatic_toxicity.csv", AQUA_COLUMNS, data_dir)


def xy(df: pd.DataFrame):
    """Split a loaded frame into (X: DataFrame, y: 1-D ndarray)."""
    return df.drop(columns=TARGET), df[TARGET].to_numpy()


def fold_error(rmse_log_units: float) -> float:
    """Translate an RMSE in -log10(mol/L) into a concentration fold-error.

    An RMSE of 1.0 log unit means the predicted lethal concentration is off
    by a factor of 10. This is the number a toxicologist actually cares about.
    """
    return float(10.0 ** rmse_log_units)


def molecule_groups(X: pd.DataFrame) -> np.ndarray:
    """Integer group label per row, shared by rows with identical descriptors.

    Two molecules with the same values for every descriptor are
    indistinguishable to any model built on those descriptors. Use this as
    `groups=` for GroupKFold so that a row and its twin cannot be split
    across training and test.
    """
    key = X.astype(str).agg("|".join, axis=1)
    return pd.factorize(key)[0]


def score_table(results, name="R2") -> pd.DataFrame:
    """Tidy mean +/- sd table from {label: array_of_fold_scores}."""
    rows = [
        {"model": k, f"{name}_mean": np.mean(v), f"{name}_sd": np.std(v), "n_folds": len(v)}
        for k, v in results.items()
    ]
    return pd.DataFrame(rows).sort_values(f"{name}_mean", ascending=False).reset_index(drop=True)

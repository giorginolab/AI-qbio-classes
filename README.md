# AI in Quantitative Biology: practical classes

Teaching material (not an application) for the practicals of *AI in Quantitative Biology*. Each
practical is one self-contained Jupyter notebook with the solutions inline: exercises are marked ✏️
and questions ❓, each followed by its answer. Everything is fixed-seed (`RNG = 0`) and runs on a laptop CPU.

## Repository organization

First, clone the repository on your machine. This is important so that the data files are accessible.

    git clone ...
    

| Path | Contents |
|---|---|
| `course/` | **The notebooks** (the source; there is no generator). See the table below. |
| `data/` | Datasets read by the notebooks (committed, so nothing needs downloading). |
| `data_prep/` | Scripts that built the datasets from public sources; instructor use only, they need internet. |
| `slides/` | Teaching-notes decks (`.pptx`), one per notebook, plus the Class 1 deck. |
| `literature/` | Reference papers (ProteinBERT). |
| `REFERENCES.md` | Papers cited on the lecture slides. |
| `protein_bert/`, `env_proteinbert/` | Upstream ProteinBERT code and a pinned TensorFlow environment for its demo notebook. |


## Install uv

UV simplified installation and run of all things Python. Do yourself a favour and 
use it. See [instructions](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2).

## Running the notebooks


```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv run jupyter lab          # run from the repository root, then open a notebook in course/
```


## Data sources

| File | Source |
|---|---|
| `qsar_fish_toxicity.csv` | UCI *QSAR fish toxicity*: fathead minnow 96 h LC50 of 908 chemicals, 6 descriptors. Headerless, `;`-separated. |
| `qsar_aquatic_toxicity.csv` | UCI *QSAR aquatic toxicity*: *Daphnia magna* 48 h LC50 of 546 chemicals, 8 descriptors. Headerless, `;`-separated. |
| `enzymes.tsv` | UniProtKB/Swiss-Prot: 350 reviewed enzymes per top-level EC class (1-6) from human, *E. coli*, yeast, *Arabidopsis* and *B. subtilis*, at most 8 per protein family. Built by `data_prep/make_enzyme_dataset.py`. |
| `targeting.tsv` | UniProtKB/Swiss-Prot: first 50 residues of proteins labelled signal peptide / mitochondrial transit peptide / other, same five organisms. Built by `data_prep/make_targeting_dataset.py`. |
| `enzymes_esm2.npz` | Mean-pooled ESM-2 embeddings (8M to 650M models, Lin et al., *Science* 2023) of the proteins in `enzymes.tsv`. Built by `data_prep/make_esm2_embeddings.py`. |
| `anage.tsv` | AnAge, Human Ageing Genomic Resources (Tacutu et al., *NAR* 2018): mass, metabolic rate and longevity per animal species. |
| `genomes.tsv` | NCBI genome reports: genome size and gene count per species. |
| `proteinbert_model/` | ProteinBERT pretrained weights (192 MB). Git-ignored; downloaded on first use. |

UCI repositories are in https://archive.ics.uci.edu/ 

`anage.tsv` and `genomes.tsv` come from `data_prep/make_scaling_datasets.py`. Swiss-Prot changes with every
release, so re-running a data-prep script gives slightly different data. **The committed files are the
reference**; the numbers quoted in the notebook answers refer to them.


## Notes

- Notebooks locate `data/` by walking up from the working directory. If you run them elsewhere (for
  example on Colab), set `QBIO_DATA` to the data folder.
- Runtimes are short: about 20 s for class 1, 55 s for class 2, 25 s for class 3, and a few minutes for the rest.
- Regression test: execute a notebook headless, e.g.
  `uv run jupyter nbconvert --to notebook --execute --output-dir "$TMPDIR" course/class1_from_regression_to_networks.ipynb`.
  Any cell that raises aborts the run.
- The ProteinBERT demo needs TensorFlow and has its own environment:
  `uv sync --project env_proteinbert && uv run --project env_proteinbert jupyter lab`.
- Requires scikit-learn >= 1.4. PyTorch (CPU build is enough) is needed only by `t03` and `class2`;
  `ipywidgets` is optional (sliders in the overfitting playgrounds).

If you edit a notebook, re-run it and update the numbers quoted in its answers.

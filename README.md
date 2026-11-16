# AI in Quantitative Biology: hands-on classes

Toni Giorgino - toni.giorgino@cnr.it - www.giorginolab.it

Teaching material (not an application) for the practicals of *AI in Quantitative Biology*. Each
notebook is self-contained, with the solutions inline: exercises are marked ✏️ and questions ❓,
each followed by its answer. Everything is fixed-seed (`RNG = 0`) and runs on a laptop CPU.

The three classes actually taught are in `classes/`. `extra_topics/` holds supplementary notebooks:
topics that the classes touch only on slides, and the scaling-law thread.


## Running the notebooks

### Preparation

1. First, clone the repository on your machine. This is important so that the data files are accessible.

    ```bash
    git clone https://github.com/giorginolab/AI-qbio-classes.git
    ```

2. Second, install UV. UV simplifies installation and run of all things Python. Do yourself a favour and 
use it. See [instructions](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2).

3. Third, create an environment

    ```bash
    cd AI-qbio-classes
    uv venv --python 3.12
    uv pip install -r requirements.txt
    ```


### Open the notebooks in jupyter

Run jupyter from the repository directory:

```bash
cd AI-qbio-classes
uv run jupyter lab
```

### Alternative: use conda

This is **untested**. You likely need to install pytorch separately, so remove it from the requirements. Then...

```bash
conda create -n ai-qbio-classes-env conda create -n ai-qbio-classes-env --file requirements.txt
conda activate ai-qbio-classes-env
conda install pytorch::pytorch
```

After that you run with...

```bash
conda activate ai-qbio-classes-env
jupyter lab
```


## Repository organization


| Path | Contents |
|---|---|
| `classes/` | **The three classes** (one folder each: deck, teaching notes, notebook), 150 / 90 / 150 minutes. See its own `README.md`. |
| `data/` | Datasets read by the notebooks (committed, so nothing needs downloading). |
| `extra_topics/` | Supplementary notebooks. Standalone expansions to the topics. Recommended. |
| `REFERENCES.md` | Papers cited on the lecture slides. Possible suggestions for your presentation. |
| `data_prep/` | Scripts that built the datasets from public sources; instructor use only, they need internet. |
| `protein_bert/`, `env_proteinbert/` | Upstream ProteinBERT code and a pinned TensorFlow environment for its demo notebook. |



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
- Runtimes are short: seconds to a few minutes per notebook.
- Regression test: execute a notebook headless, e.g.
  `uv run jupyter nbconvert --to notebook --execute --output-dir "$TMPDIR" extra_topics/extra05_convolutional_networks.ipynb`.
  Any cell that raises aborts the run.
- The ProteinBERT demo needs TensorFlow and has its own environment:
  `uv sync --project env_proteinbert && uv run --project env_proteinbert jupyter lab`.
- Requires scikit-learn >= 1.4. PyTorch (CPU build is enough) is needed by `extra03`-`extra05` and by
  class 3; `ipywidgets` is optional (sliders in the overfitting playgrounds).

If you edit a notebook, re-run it and update the numbers quoted in its answers.

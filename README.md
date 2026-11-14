# AI in Quantitative Biology: practical classes

Teaching material (not an application) for the practicals of *AI in Quantitative Biology*. Each
practical is one self-contained Jupyter notebook with the solutions inline: exercises are marked ✏️
and questions ❓, each followed by its answer. Everything is fixed-seed (`RNG = 0`) and runs on a laptop CPU.

## Repository organization

| Path | Contents |
|---|---|
| `course/` | **The notebooks** (the source; there is no generator). See the table below. |
| `data/` | Datasets read by the notebooks (committed, so nothing needs downloading). |
| `data_prep/` | Scripts that built the datasets from public sources; instructor use only, they need internet. |
| `slides/` | Teaching-notes decks (`.pptx`), one per notebook, plus the Class 1 deck. |
| `literature/` | Reference papers (ProteinBERT). |
| `REFERENCES.md` | Papers cited on the lecture slides. |
| `protein_bert/`, `env_proteinbert/` | Upstream ProteinBERT code and a pinned TensorFlow environment for its demo notebook. |
| `carlos-course/` | A separate, older QM notebook (`t07_QM`). |
| `old/`, `README.old.md`, `opus-ideas.md` | Earlier material, notes and proposals. |

### Notebooks (`course/`)

| Notebook | Topic | Data |
|---|---|---|
| `class1_from_regression_to_networks` | Lecture class 1 (slides 1-26): regression, the neuron, gradient descent, overfitting | fish toxicity |
| `class2_architectures` | Lecture class 2 (slides 27-46): CNNs, RNNs, autoencoders, first look at transformers | targeting, enzymes |
| `class3_applications` | Lecture class 3 (slides 47-79): transfer, scale, and which split to believe | ESM-2 embeddings |
| `t01_toxicity_qsar` | Session 1: mining a toxicity dataset (QSAR) | fish, aquatic toxicity |
| `t02_neural_networks_enzymes` | Session 2: MLPs, and classifying enzymes by EC class | fish toxicity, enzymes |
| `t03_rnn_protein_targeting` | Session 3: a GRU reads protein N-termini | targeting |
| `extra01_scaling_laws` | Alternative to t01: scaling laws | anage |
| `extra02_laws_vs_flexible_models` | Alternative to t02: laws against flexible models | anage, genomes |
| `extra03_enzymes_esm2` | Optional: what a protein language model knows about enzymes | enzymes, ESM-2 embeddings |

`extra01`/`extra02` are toxicity-free replacements for `t01`/`t02`; both sets are kept.

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

`anage.tsv` and `genomes.tsv` come from `data_prep/make_scaling_datasets.py`. Swiss-Prot changes with every
release, so re-running a data-prep script gives slightly different data. **The committed files are the
reference**; the numbers quoted in the notebook answers refer to them.

## Running the notebooks

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv run jupyter lab          # run from the repository root, then open a notebook in course/
```

- Requires scikit-learn >= 1.4. PyTorch (CPU build is enough) is needed only by `t03` and `class2`;
  `ipywidgets` is optional (sliders in the overfitting playgrounds).
- Notebooks locate `data/` by walking up from the working directory. If you run them elsewhere (for
  example on Colab), set `QBIO_DATA` to the data folder.
- Runtimes are short: about 20 s for class 1, 55 s for class 2, 25 s for class 3, and a few minutes for the rest.
- Regression test: execute a notebook headless, e.g.
  `uv run jupyter nbconvert --to notebook --execute --output-dir "$TMPDIR" course/class1_from_regression_to_networks.ipynb`.
  Any cell that raises aborts the run.
- The ProteinBERT demo needs TensorFlow and has its own environment:
  `uv sync --project env_proteinbert && uv run --project env_proteinbert jupyter lab`.

If you edit a notebook, re-run it and update the numbers quoted in its answers.

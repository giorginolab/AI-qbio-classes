"""Build data/anage.tsv and data/genomes.tsv for the scaling-law practicals (s01, s02).

Run once by the instructor (needs internet); students only use the TSVs.

    python course/data_prep/make_scaling_datasets.py             # download + curate
    python course/data_prep/make_scaling_datasets.py CACHE_DIR   # reuse/keep downloads in CACHE_DIR

Sources
  AnAge (Human Ageing Genomic Resources), one row per animal species:
      https://genomics.senescence.info/species/dataset.zip
      Tacutu et al., Nucleic Acids Res. 46:D1083 (2018).
  NCBI genome reports, one row per genome assembly:
      https://ftp.ncbi.nlm.nih.gov/genomes/GENOME_REPORTS/{eukaryotes,prokaryotes}.txt

anage.tsv    animals with an adult weight or a metabolic rate. `bmr_W` and `bmr_mass_g` come
             from the same measurement (basal metabolic rate and the mass of the animals
             measured); `adult_mass_g` and `longevity_yr` are species-level values.
genomes.tsv  genome size (Mb) and number of annotated genes (protein-coding and RNA), one row
             per species (TaxID): the median over its annotated assemblies.
             Prokaryotes: complete genomes only, and only the size, the gene count and the
             taxonomic group are kept (no organism names). Eukaryotes: with the species name.

Note: the committed TSVs are the reference for the numbers quoted in the notebooks.
"""
import io
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ANAGE_URL = "https://genomics.senescence.info/species/dataset.zip"
NCBI_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/GENOME_REPORTS/"
OUT = Path(__file__).resolve().parents[2] / "data"


def fetch(url, cache):
    path = Path(cache) / url.rsplit("/", 1)[1]
    if not path.exists():
        print("downloading", url)
        urllib.request.urlretrieve(url, path)
    return path


def make_anage(cache):
    with zipfile.ZipFile(fetch(ANAGE_URL, cache)) as z:
        raw = pd.read_csv(io.BytesIO(z.read("anage_data.txt")), sep="\t")
    raw = raw[raw["Kingdom"] == "Animalia"]
    df = pd.DataFrame({
        "class": raw["Class"],
        "order": raw["Order"],
        "family": raw["Family"],
        "species": raw["Genus"] + " " + raw["Species"],
        "common_name": raw["Common name"],
        "adult_mass_g": raw["Adult weight (g)"],
        "longevity_yr": raw["Maximum longevity (yrs)"],
        "bmr_W": raw["Metabolic rate (W)"],
        "bmr_mass_g": raw["Body mass (g)"],
        "body_temp_K": raw["Temperature (K)"],
        "gestation_days": raw["Gestation/Incubation (days)"],
        "litter_size": raw["Litter/Clutch size"],
        "female_maturity_days": raw["Female maturity (days)"],
        "birth_mass_g": raw["Birth weight (g)"],
    })
    df = df[df["adult_mass_g"].notna() | df["bmr_W"].notna()]
    df = df.sort_values(["class", "order", "family", "species"]).reset_index(drop=True)
    df.to_csv(OUT / "anage.tsv", sep="\t", index=False, float_format="%.6g")
    print("anage.tsv", df.shape, "with BMR:", df["bmr_W"].notna().sum())


def per_species(df):
    """One row per TaxID: median size and gene count over its annotated assemblies."""
    df = df.dropna(subset=["Size (Mb)", "Genes"])
    g = df.groupby("TaxID")
    out = g[["Size (Mb)", "Genes"]].median()
    return out.join(g[[c for c in df.columns if c not in ("TaxID", "Size (Mb)", "Genes")]].first())


def make_genomes(cache):
    euk = pd.read_csv(fetch(NCBI_URL + "eukaryotes.txt", cache), sep="\t", na_values="-",
                      low_memory=False)
    euk = per_species(euk[["TaxID", "Group", "SubGroup", "#Organism/Name", "Size (Mb)", "Genes"]])
    euk = euk[euk["Genes"] >= 1000]                       # drop organelle-only / partial annotations
    euk = pd.DataFrame({
        "domain": "eukaryote",
        "group": euk["Group"],
        "subgroup": euk["SubGroup"],
        "organism": euk["#Organism/Name"],
        "size_mb": euk["Size (Mb)"],
        "genes": euk["Genes"].round().astype(int),
    })

    cols = ["TaxID", "Group", "SubGroup", "Size (Mb)", "Genes", "Status"]
    pro = pd.read_csv(fetch(NCBI_URL + "prokaryotes.txt", cache), sep="\t", na_values="-",
                      usecols=cols, low_memory=False)
    pro = per_species(pro[pro["Status"] == "Complete Genome"].drop(columns="Status"))
    pro = pro[pro["Genes"] >= 300]
    pro = pd.DataFrame({
        "domain": "prokaryote",
        "group": pro["Group"],
        "subgroup": pro["SubGroup"],
        "organism": "",
        "size_mb": pro["Size (Mb)"],
        "genes": pro["Genes"].round().astype(int),
    })

    # No upper filter on purpose: a few eukaryote annotations report absurd gene counts,
    # and finding them is an exercise in s02.
    df = pd.concat([pro, euk]).sort_values(["domain", "group", "subgroup", "size_mb", "genes"])
    df.to_csv(OUT / "genomes.tsv", sep="\t", index=False, float_format="%.6g")
    print("genomes.tsv", df.shape, df["domain"].value_counts().to_dict())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        Path(sys.argv[1]).mkdir(parents=True, exist_ok=True)
        make_anage(sys.argv[1]); make_genomes(sys.argv[1])
    else:
        with tempfile.TemporaryDirectory() as tmp:
            make_anage(tmp); make_genomes(tmp)

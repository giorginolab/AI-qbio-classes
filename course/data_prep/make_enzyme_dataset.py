"""Build data/enzymes.tsv: reviewed enzymes (Swiss-Prot) labelled by top-level EC class.

Run once by the instructor (needs internet); students only use the TSV.

    python course/data_prep/make_enzyme_dataset.py            # download + curate
    python course/data_prep/make_enzyme_dataset.py raw.tsv    # curate an existing download

Source: UniProtKB/Swiss-Prot (reviewed), entries with an EC number, length 80-1000,
from five model organisms (human, E. coli K-12, budding yeast, Arabidopsis, B. subtilis).
Curation:
  * keep entries whose EC numbers all share one top-level class, classes 1-6
    (EC 7, translocases, has too few entries);
  * protein family = UniProt "Protein families" (similarity comment); entries without
    one get their own singleton family;
  * at most 8 entries per family (limits redundancy, e.g. ~400 protein kinases);
  * 350 entries per class, sampled with a fixed seed.

Note: re-downloading later gives a slightly different set (Swiss-Prot changes every
release), which shifts the numbers quoted in the instructor guide. The committed TSV
is the reference.
"""
import gzip
import io
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

SEED = 0
PER_FAMILY = 8
PER_CLASS = 350
ORGANISMS = [9606, 83333, 559292, 3702, 224308]
CLASS_NAMES = {"1": "Oxidoreductase", "2": "Transferase", "3": "Hydrolase",
               "4": "Lyase", "5": "Isomerase", "6": "Ligase"}
FIELDS = "accession,id,protein_name,organism_name,ec,length,protein_families,sequence"

QUERY = ("(reviewed:true) AND (ec:*) AND (length:[80 TO 1000]) AND ("
         + " OR ".join(f"organism_id:{o}" for o in ORGANISMS) + ")")
URL = ("https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=tsv"
       f"&fields={FIELDS}&query=" + urllib.parse.quote(QUERY))


def download():
    with urllib.request.urlopen(URL) as r:
        return pd.read_csv(io.BytesIO(gzip.decompress(r.read())), sep="\t")


def curate(raw):
    d = raw.rename(columns={"Entry": "accession", "Entry Name": "entry_name",
                            "Protein names": "protein_name", "Organism": "organism",
                            "EC number": "ec", "Length": "length",
                            "Protein families": "family", "Sequence": "sequence"})
    d = d[["accession", "entry_name", "protein_name", "organism", "ec",
           "length", "family", "sequence"]].copy()
    tops = d["ec"].str.split("; ").apply(lambda l: {x.split(".")[0] for x in l})
    d = d[tops.str.len() == 1].copy()
    d["ec_class"] = tops[d.index].apply(lambda s: next(iter(s)))
    d = d[d["ec_class"].isin(CLASS_NAMES)]
    d = d[d["sequence"].str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+")]   # 20 standard aa only
    d["family"] = d["family"].fillna("singleton_" + d["accession"])
    d["organism"] = d["organism"].str.replace(r"\s*\(.*", "", regex=True)
    d["protein_name"] = d["protein_name"].str.replace(r"\s*\(.*", "", regex=True)

    d = d.sample(frac=1, random_state=SEED)
    d = d.groupby("family", group_keys=False).head(PER_FAMILY)
    d = d.groupby("ec_class", group_keys=False).head(PER_CLASS)
    d["ec_class"] = d["ec_class"].astype(int)
    d["ec_name"] = d["ec_class"].astype(str).map(CLASS_NAMES)
    cols = ["accession", "entry_name", "protein_name", "organism", "ec", "ec_class",
            "ec_name", "family", "length", "sequence"]
    return d[cols].sort_values(["ec_class", "accession"]).reset_index(drop=True)


if __name__ == "__main__":
    raw = pd.read_csv(sys.argv[1], sep="\t") if len(sys.argv) > 1 else download()
    out = curate(raw)
    path = Path(__file__).resolve().parents[2] / "data" / "enzymes.tsv"
    out.to_csv(path, sep="\t", index=False)
    print(f"{len(out)} sequences, {out['family'].nunique()} families -> {path}")
    print(out["ec_name"].value_counts().to_string())

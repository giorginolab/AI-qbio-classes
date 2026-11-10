"""Build data/targeting.tsv: N-terminal 50 residues labelled by targeting signal.

Run once by the instructor (needs internet); students only use the TSV.

    python course/data_prep/make_targeting_dataset.py            # download + curate
    python course/data_prep/make_targeting_dataset.py raw.tsv    # curate an existing download

Source: UniProtKB/Swiss-Prot (reviewed), length 80-1000, same five model organisms as
enzymes.tsv. Three classes (a small "TargetP"-style problem):
  SP     secretory signal peptide: SIGNAL feature starting at residue 1
  mTP    mitochondrial transit peptide: TRANSIT feature starting at 1, note=Mitochondrion
  other  no SIGNAL / TRANSIT / TRANSMEM feature, subcellular location mentions
         Cytoplasm or Nucleus and none of the membrane-bounded compartments
Only the first 50 residues are kept (signal peptides are ~15-35 aa, transit peptides
~20-60 aa). Identical 50-mers are dropped; at most 1200 per class with a fixed seed.

Note: re-downloading later gives a slightly different set (Swiss-Prot changes every
release). The committed TSV is the reference.
"""
import gzip
import io
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

SEED = 0
NTERM = 50
PER_CLASS = 1200
ORGANISMS = [9606, 83333, 559292, 3702, 224308]
FIELDS = ("accession,id,protein_name,organism_name,length,ft_signal,ft_transit,"
          "ft_transmem,cc_subcellular_location,sequence")
QUERY = ("(reviewed:true) AND (length:[80 TO 1000]) AND ("
         + " OR ".join(f"organism_id:{o}" for o in ORGANISMS) + ")")
URL = ("https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=tsv"
       f"&fields={FIELDS}&query=" + urllib.parse.quote(QUERY))
EXCLUDE = ("Secreted|Membrane|membrane|Mitochondri|Plastid|Chloroplast|Endoplasmic|"
           "Golgi|Peroxisome|Lysosome|Vacuole|Periplasm|Cell wall|Endosome|Vesicle|vesicle")


def download():
    with urllib.request.urlopen(URL) as r:
        return pd.read_csv(io.BytesIO(gzip.decompress(r.read())), sep="\t")


def curate(raw):
    d = raw.rename(columns={"Entry": "accession", "Entry Name": "entry_name",
                            "Protein names": "protein_name", "Organism": "organism",
                            "Signal peptide": "signal", "Transit peptide": "transit",
                            "Transmembrane": "transmem",
                            "Subcellular location [CC]": "location",
                            "Sequence": "sequence"})
    for c in ["signal", "transit", "transmem", "location"]:
        d[c] = d[c].fillna("")
    sp = d["signal"].str.match(r"SIGNAL 1\.\.\d+")
    mtp = d["transit"].str.match(r"TRANSIT 1\.\.\d+") & d["transit"].str.contains(
        'note="Mitochondrion')
    other = ((d["signal"] == "") & (d["transit"] == "") & (d["transmem"] == "")
             & d["location"].str.contains("Cytoplasm|Nucleus")
             & ~d["location"].str.contains(EXCLUDE))
    d["label"] = None
    d.loc[other, "label"] = "other"
    d.loc[sp & ~mtp, "label"] = "SP"
    d.loc[mtp & ~sp, "label"] = "mTP"
    d = d[d["label"].notna()].copy()

    # length of the annotated signal (cleavage site position), 0 for "other"
    feat = d["signal"].where(d["label"] == "SP", d["transit"])
    d["signal_len"] = pd.to_numeric(feat.str.extract(r"^\w+ 1\.\.(\d+)")[0]).fillna(0).astype(int)
    d["nterm"] = d["sequence"].str[:NTERM]
    d = d[d["nterm"].str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]{%d}" % NTERM)]
    d = d.drop_duplicates("nterm")
    d["organism"] = d["organism"].str.replace(r"\s*\(.*", "", regex=True)
    d["protein_name"] = d["protein_name"].str.replace(r"\s*\(.*", "", regex=True)

    d = d.sample(frac=1, random_state=SEED).groupby("label", group_keys=False).head(PER_CLASS)
    cols = ["accession", "entry_name", "protein_name", "organism", "label",
            "signal_len", "nterm"]
    return d[cols].sort_values(["label", "accession"]).reset_index(drop=True)


if __name__ == "__main__":
    raw = pd.read_csv(sys.argv[1], sep="\t") if len(sys.argv) > 1 else download()
    out = curate(raw)
    path = Path(__file__).resolve().parents[2] / "data" / "targeting.tsv"
    out.to_csv(path, sep="\t", index=False)
    print(f"{len(out)} sequences -> {path}")
    print(out.groupby(["label", "organism"]).size().unstack(fill_value=0).to_string())

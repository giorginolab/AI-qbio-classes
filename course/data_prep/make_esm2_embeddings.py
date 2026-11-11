"""Build data/enzymes_esm2.npz: ESM-2 embeddings of the enzymes in data/enzymes.tsv.

Run once by the instructor; students only use the .npz and need neither PyTorch nor the models.

    pip install torch transformers
    python course/data_prep/make_esm2_embeddings.py              # 8M, 35M, 150M, 650M
    python course/data_prep/make_esm2_embeddings.py 8M 35M       # only some sizes (merged into the file)

ESM-2 (Lin et al., Science 379:1123, 2023) is a protein language model trained to guess masked
residues in about 65 million UniRef sequences; it never sees a functional annotation. The
frozen model turns each residue into a vector; we average the vectors of the last layer over the
sequence (mean pooling, special tokens excluded) to get one vector per protein.

Output arrays (float16, rows in the order of enzymes.tsv):
    accession                 UniProt accessions, to check the alignment with the table
    esm2_8M   (n, 320)        facebook/esm2_t6_8M_UR50D
    esm2_35M  (n, 480)        facebook/esm2_t12_35M_UR50D
    esm2_150M (n, 640)        facebook/esm2_t30_150M_UR50D
    esm2_650M (n, 1280)       facebook/esm2_t33_650M_UR50D

CPU time on 8 cores for the 2047 enzymes: see the timings printed by the script (the 650M model
is by far the slowest).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmModel

MODELS = {
    "8M": "facebook/esm2_t6_8M_UR50D",
    "35M": "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
}
DATA = Path(__file__).resolve().parents[2] / "data"
OUT = DATA / "enzymes_esm2.npz"
MAX_RESIDUES_PER_BATCH = 6000


def batches_by_length(seqs):
    """Indices grouped in batches of similar length (little padding, bounded memory)."""
    order = np.argsort([len(s) for s in seqs])
    batch, longest = [], 0
    for i in order:
        longest = max(longest, len(seqs[i]))
        if batch and longest * (len(batch) + 1) > MAX_RESIDUES_PER_BATCH:
            yield batch
            batch, longest = [], len(seqs[i])
        batch.append(i)
    if batch:
        yield batch


@torch.no_grad()
def embed(seqs, model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name, add_pooling_layer=False).eval()
    out = np.zeros((len(seqs), model.config.hidden_size), dtype=np.float32)
    for idx in batches_by_length(seqs):
        enc = tok([seqs[i] for i in idx], return_tensors="pt", padding=True)
        hidden = model(**enc).last_hidden_state                  # (batch, tokens, dim)
        special = torch.tensor([tok.get_special_tokens_mask(ids, already_has_special_tokens=True)
                                for ids in enc["input_ids"].tolist()], dtype=torch.bool)
        keep = (enc["attention_mask"].bool() & ~special).unsqueeze(-1)   # residues only
        out[idx] = ((hidden * keep).sum(1) / keep.sum(1)).numpy()
    return out


if __name__ == "__main__":
    sizes = sys.argv[1:] or list(MODELS)
    enz = pd.read_csv(DATA / "enzymes.tsv", sep="\t")
    seqs = enz["sequence"].tolist()                                # longest is 993 < 1022 residues
    arrays = dict(np.load(OUT)) if OUT.exists() else {}
    arrays["accession"] = np.array(enz["accession"].tolist(), dtype="U")   # plain strings, no pickle
    for size in sizes:
        t0 = time.time()
        arrays[f"esm2_{size}"] = embed(seqs, MODELS[size]).astype(np.float16)
        print(f"esm2_{size}: {arrays[f'esm2_{size}'].shape}, {time.time() - t0:.0f} s", flush=True)
        np.savez_compressed(OUT, **arrays)
    print("wrote", OUT)

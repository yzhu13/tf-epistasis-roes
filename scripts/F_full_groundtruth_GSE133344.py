"""
Direction F: Full GSE133344 ground truth expansion.

Builds pseudobulk expression profiles per guide_identity condition from the
full GEO raw dataset (290 conditions: 1 control + 157 singles + 132 doubles),
then computes epistasis ground truth for all double-perturbation pairs that
have matching single-KO controls.

eps_vector = E_AB - (E_A + E_B - E_ctrl)
eps_l2 = ||eps_vector||
eps_relative = eps_l2 / ||E_AB - E_ctrl||
classification: additive / synergistic / buffering / complex via cosine sim
with single-perturbation deltas.

Run: python scripts/F_full_groundtruth_GSE133344.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
from pathlib import Path
import gzip
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
RAW = BASE / "data" / "raw" / "Norman2019_GSE133344"
OUT = BASE / "results" / "F_full_groundtruth"
OUT.mkdir(parents=True, exist_ok=True)
FIG = BASE / "figures" / "F_full_groundtruth"
FIG.mkdir(parents=True, exist_ok=True)

# ── 1. Load matrix (genes x cells, 10x mtx format) ───────────────
print("Loading matrix.mtx.gz (genes x cells) ...")
with gzip.open(RAW / "matrix.mtx.gz", "rb") as f:
    mat = sio.mmread(f).tocsc()
print(f"  Matrix shape: {mat.shape}")  # (n_genes, n_cells)

genes = pd.read_csv(RAW / "genes.tsv", sep="\t", header=None, names=["ensembl", "symbol"])
barcodes = pd.read_csv(RAW / "barcodes.tsv", header=None, names=["barcode"])
print(f"  Genes: {len(genes)}  Barcodes: {len(barcodes)}")

# Some barcodes may have duplicate names across gem groups -> use positional index
barcode_to_idx = {bc: i for i, bc in enumerate(barcodes["barcode"])}

# ── 2. Load cell identities ──────────────────────────────────────
print("\nLoading cell_identities.csv ...")
cid = pd.read_csv(RAW / "cell_identities.csv")
print(f"  Cells with identity: {len(cid)}")
print(f"  Unique guide_identity: {cid['guide_identity'].nunique()}")

# Map cell_barcode (with gemgroup suffix, e.g. "TTGAACGAGACTCGGA-2") to matrix column index.
# barcodes.tsv has plain barcodes with -N suffix matching gemgroup-based ordering already
cid["col_idx"] = cid["cell_barcode"].map(barcode_to_idx)
n_missing = cid["col_idx"].isna().sum()
print(f"  Cells not found in barcodes.tsv: {n_missing} / {len(cid)}")
cid = cid.dropna(subset=["col_idx"])
cid["col_idx"] = cid["col_idx"].astype(int)

# ── 3. Normalize: CPM-like + log1p (per cell, then aggregate) ────
print("\nNormalizing (per-cell total-count -> CPM -> log1p) ...")
total_counts = np.asarray(mat.sum(axis=0)).flatten()
total_counts[total_counts == 0] = 1
cpm = mat.multiply(1.0 / total_counts) * 1e4  # CP10K
cpm = cpm.tocsc()

# ── 4. Pseudobulk per condition (mean log1p expression) ──────────
print("\nBuilding pseudobulk profiles per guide_identity ...")
conditions = cid["guide_identity"].unique()
print(f"  Total conditions: {len(conditions)}")

pseudobulk = {}
ncells_per_cond = {}
for cond in conditions:
    idx = cid.loc[cid["guide_identity"] == cond, "col_idx"].values
    if len(idx) < 5:
        continue
    sub = cpm[:, idx]
    mean_expr = np.asarray(sub.mean(axis=1)).flatten()
    pseudobulk[cond] = np.log1p(mean_expr)
    ncells_per_cond[cond] = len(idx)

print(f"  Conditions with >=5 cells: {len(pseudobulk)}")

pb_df = pd.DataFrame(pseudobulk, index=genes["symbol"].values)
# Collapse duplicate gene symbols by mean
pb_df = pb_df.groupby(pb_df.index).mean()
print(f"  Pseudobulk matrix: {pb_df.shape} (genes x conditions)")

pb_df.to_csv(OUT / "pseudobulk_all_conditions.csv.gz", compression="gzip")
print(f"  Saved: {OUT / 'pseudobulk_all_conditions.csv.gz'}")

ncells_df = pd.Series(ncells_per_cond, name="n_cells")
ncells_df.to_csv(OUT / "n_cells_per_condition.csv")

# ── 5. Identify control / single / double conditions ─────────────
def parse_guide(g):
    parts = g.split("__")[0].split("_")
    a, b = parts[0], "_".join(parts[1:]) if len(parts) > 2 else parts[1]
    return a, b

ctrl_conds = [c for c in pb_df.columns if c.split("__")[0].replace("NegCtrl0", "").replace("_", "") == ""]
print(f"\nControl conditions found: {ctrl_conds}")

singles = {}  # TF -> condition name
doubles = []  # (TF_A, TF_B, condition)
for c in pb_df.columns:
    a, b = parse_guide(c)
    if a == "NegCtrl0" and b == "NegCtrl0":
        continue
    elif a == "NegCtrl0":
        singles[b] = c
    elif b == "NegCtrl0":
        singles[a] = c
    else:
        doubles.append((a, b, c))

print(f"Singles: {len(singles)}  Doubles: {len(doubles)}")

if not ctrl_conds:
    raise RuntimeError("No control condition found!")
E_ctrl = pb_df[ctrl_conds].mean(axis=1).values

# ── 6. Compute epistasis for all double pairs with both singles ──
print("\nComputing epistasis ground truth ...")
results = []
for tf_a, tf_b, cond_ab in doubles:
    if tf_a not in singles or tf_b not in singles:
        continue
    E_A = pb_df[singles[tf_a]].values
    E_B = pb_df[singles[tf_b]].values
    E_AB = pb_df[cond_ab].values

    delta_A = E_A - E_ctrl
    delta_B = E_B - E_ctrl
    delta_AB = E_AB - E_ctrl
    expected_additive = delta_A + delta_B

    eps_vector = delta_AB - expected_additive
    eps_l2 = np.linalg.norm(eps_vector)
    denom = np.linalg.norm(delta_AB)
    eps_relative = eps_l2 / denom if denom > 0 else np.nan

    # classification via cosine similarity with single deltas
    def cos(u, v):
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu == 0 or nv == 0:
            return 0.0
        return np.dot(u, v) / (nu * nv)

    cos_obs_expected = cos(delta_AB, expected_additive)
    cos_A = cos(eps_vector, delta_A)
    cos_B = cos(eps_vector, delta_B)

    if eps_relative < 0.1:
        eps_type = "additive"
    elif cos_obs_expected < -0.1:
        eps_type = "synergistic"
    elif cos_obs_expected > 0.5 and np.linalg.norm(delta_AB) < np.linalg.norm(expected_additive):
        eps_type = "buffering"
    else:
        eps_type = "complex"

    # top epistatic genes
    top_idx = np.argsort(-np.abs(eps_vector))[:10]
    top_genes = ";".join(pb_df.index[top_idx].astype(str))

    results.append({
        "TF_A": tf_a, "TF_B": tf_b, "perturbation_key": cond_ab,
        "eps_l2": eps_l2, "eps_relative": eps_relative, "eps_type": eps_type,
        "cos_with_A": cos_A, "cos_with_B": cos_B,
        "n_cells_AB": ncells_per_cond.get(cond_ab, np.nan),
        "n_cells_A": ncells_per_cond.get(singles[tf_a], np.nan),
        "n_cells_B": ncells_per_cond.get(singles[tf_b], np.nan),
        "top_epistatic_genes": top_genes,
    })

df = pd.DataFrame(results)
print(f"\nTotal pairs with full ground truth: {len(df)}")
print(df["eps_type"].value_counts())

out_path = OUT / "epistasis_groundtruth_GSE133344.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# ── 7. Summary stats ──────────────────────────────────────────────
with open(OUT / "summary.txt", "w") as f:
    f.write("="*60 + "\n")
    f.write("GSE133344 FULL EPISTASIS GROUND TRUTH - SUMMARY\n")
    f.write("="*60 + "\n")
    f.write(f"Total double-perturbation conditions: {len(doubles)}\n")
    f.write(f"Pairs with matching singles (full GT):  {len(df)}\n")
    f.write(f"Singles available: {len(singles)}\n\n")
    f.write("Epistasis type distribution:\n")
    for k, v in df["eps_type"].value_counts().items():
        f.write(f"  {k:15s}: {v}\n")
    f.write(f"\neps_l2 mean  : {df['eps_l2'].mean():.4f}\n")
    f.write(f"eps_l2 median: {df['eps_l2'].median():.4f}\n")
    f.write(f"eps_l2 max   : {df['eps_l2'].max():.4f}\n")
    f.write(f"\nTop 15 by eps_l2:\n")
    f.write(df.nlargest(15, "eps_l2")[["TF_A","TF_B","eps_l2","eps_relative","eps_type"]].to_string(index=False))

print("Summary written.")
print("\nDone.")

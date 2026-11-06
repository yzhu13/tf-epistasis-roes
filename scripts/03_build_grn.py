"""
Script 03: Build Gene Regulatory Network (GRN) using STREAM or SCENIC+.

We use STREAM (Li et al., Briefings in Bioinformatics 2024) as our primary tool,
which outperforms SCENIC+ and Pando in benchmarking for:
  - TF recovery
  - TF-enhancer relation identification
  - Enhancer-gene relation prediction

For this project, we use Norman 2019 K562 perturbation data to build a
perturbation-calibrated GRN (since we have ground truth single-KO effects).

Strategy:
  Phase 1 (current): perturbation-derived proxy GRN from Norman 2019 + Replogle 2022
    - Empirically measured TF→gene effects from CRISPRi screens
    - More reliable than inferred GRN for K562 specifically
    - Directly anchored to the cell type we're studying

  Phase 2 (full): STREAM eGRN from HCA 10x Multiome
    - For broader cell-type generalization
    - Requires installing STREAM and ~10x Multiome data

Input:
  data/raw/Norman2019/norman2019_perturbseq.h5ad
  data/raw/Replogle2022/replogle2022_k562_essential.h5ad

Output:
  data/processed/grn_tf_target_network.csv
    Columns: TF, target_gene, weight, edge_type, source, confidence
"""

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROC = BASE_DIR / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Phase 1: Perturbation-calibrated GRN from Norman + Replogle
# ─────────────────────────────────────────────────────────────────

def get_mean_profiles(adata, pert_col):
    """Get mean expression profile per perturbation."""
    X = adata.X
    if scipy.sparse.issparse(X):
        X = X.toarray()
    profiles = {}
    for pert in adata.obs[pert_col].unique():
        mask = adata.obs[pert_col] == pert
        profiles[pert] = X[mask].mean(axis=0)
    return profiles

def find_pert_col(adata):
    for col in ["perturbation", "condition", "gene", "perturbation_name", "guide_ids"]:
        if col in adata.obs.columns:
            return col
    raise ValueError(f"Cannot find perturbation column. obs columns: {list(adata.obs.columns)}")

def find_control(profiles):
    for key in profiles:
        for ctrl in ["control", "ctrl", "non-targeting", "NT", "scramble"]:
            if ctrl.lower() in str(key).lower():
                return key
    raise ValueError(f"Cannot find control. Keys: {list(profiles.keys())[:20]}")

def build_grn_from_perturb(h5ad_path, dataset_name, effect_threshold=0.5, max_targets_per_tf=200):
    """
    Build TF→gene edges from perturbation screen data.

    For each single TF perturbation:
      - Compute log2FC vs control
      - Add edge TF→gene for all significantly changed genes
      - Weight = |log2FC|, edge_type = activation/repression

    This is more reliable than inferred GRN because it's empirically measured.
    """
    print(f"\nBuilding GRN from {dataset_name}: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    pert_col = find_pert_col(adata)
    genes = list(adata.var_names)
    profiles = get_mean_profiles(adata, pert_col)
    ctrl_key = find_control(profiles)
    e_ctrl = profiles[ctrl_key]

    # Filter for known TFs
    tf_file = DATA_RAW / "TF_annotations" / "human_TFs_Lambert2018.txt"
    if tf_file.exists():
        tf_list = set(pd.read_csv(tf_file, sep="\t", header=None)[0].tolist())
    else:
        # Fallback: use all single perturbations as TFs (they mostly are TFs in Norman)
        tf_list = None

    edges = []
    single_perts = []
    for key in profiles:
        if key == ctrl_key:
            continue
        key_str = str(key)
        # Skip double perturbations
        if any(sep in key_str for sep in ['+', ',']):
            continue
        single_perts.append(key)

    print(f"  Single perturbations: {len(single_perts)}")

    for tf in single_perts:
        # Filter to TFs if list available
        tf_clean = tf.strip()
        if tf_list and tf_clean not in tf_list:
            continue

        delta = profiles[tf] - e_ctrl

        # Significant targets
        sig_idx = np.where(np.abs(delta) > effect_threshold)[0]
        if len(sig_idx) == 0:
            continue

        # Keep top targets by effect size
        sorted_idx = sig_idx[np.argsort(np.abs(delta[sig_idx]))[::-1]][:max_targets_per_tf]

        for idx in sorted_idx:
            edge_type = "activation" if delta[idx] > 0 else "repression"
            edges.append({
                "TF": tf_clean,
                "target_gene": genes[idx],
                "weight": abs(float(delta[idx])),
                "edge_type": edge_type,
                "source": dataset_name,
                "confidence": min(1.0, abs(float(delta[idx])) / 2.0),
            })

    print(f"  GRN edges from {dataset_name}: {len(edges)}")
    return pd.DataFrame(edges)

def build_combined_grn():
    """Combine GRN evidence from Norman 2019 and Replogle 2022."""
    all_edges = []

    # Norman 2019 (CRISPRa - overexpression, but shows TF target genes)
    norman_h5ad = DATA_RAW / "Norman2019" / "norman2019_perturbseq.h5ad"
    if norman_h5ad.exists():
        edges_norman = build_grn_from_perturb(norman_h5ad, "Norman2019_CRISPRa")
        all_edges.append(edges_norman)
    else:
        print("Norman 2019 not found. Run 01_download_data.py first.")

    # Replogle 2022 (CRISPRi - knockdown, more directly represents KO effects)
    replogle_h5ad = DATA_RAW / "Replogle2022" / "replogle2022_k562_essential.h5ad"
    if replogle_h5ad.exists():
        edges_replogle = build_grn_from_perturb(replogle_h5ad, "Replogle2022_CRISPRi",
                                                  effect_threshold=0.3)
        all_edges.append(edges_replogle)
    else:
        print("Replogle 2022 not found. Run 01_download_data.py first.")

    if not all_edges:
        raise RuntimeError("No data available. Run 01_download_data.py first.")

    grn_df = pd.concat(all_edges, ignore_index=True)

    # Aggregate: if same TF→gene appears in both datasets, take max weight and mark as high-confidence
    grn_agg = (
        grn_df
        .groupby(["TF", "target_gene", "edge_type"])
        .agg(
            weight=("weight", "max"),
            confidence=("confidence", "mean"),
            n_sources=("source", "nunique"),
            sources=("source", lambda x: "|".join(sorted(set(x))))
        )
        .reset_index()
    )
    grn_agg["confidence"] = grn_agg["confidence"] * (1 + 0.5 * (grn_agg["n_sources"] - 1))
    grn_agg["confidence"] = grn_agg["confidence"].clip(0, 1)

    print(f"\nCombined GRN: {len(grn_agg)} edges")
    print(f"  TFs: {grn_agg['TF'].nunique()}")
    print(f"  Target genes: {grn_agg['target_gene'].nunique()}")
    print(f"  High-confidence (both datasets): {(grn_agg['n_sources'] > 1).sum()}")

    out_path = DATA_PROC / "grn_tf_target_network.csv"
    grn_agg.to_csv(out_path, index=False)
    print(f"\nGRN saved: {out_path}")

    # Summary stats
    tf_sizes = grn_agg.groupby("TF").size().sort_values(ascending=False)
    print(f"\nTop 10 TFs by number of targets:")
    print(tf_sizes.head(10))

    return grn_agg

if __name__ == "__main__":
    grn = build_combined_grn()
    print("\nNext step: python scripts/04_compute_roes.py")

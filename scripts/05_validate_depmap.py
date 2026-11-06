"""
Script 05: Validate ROES predictions using DepMap co-essentiality.

Logic:
  TF pairs with high ROES (predicted synthetic essential = functionally redundant)
  should show HIGH CO-ESSENTIALITY in DepMap — i.e., their CERES scores should
  be positively correlated across cancer cell lines.

  Rationale: if TF_A and TF_B are redundant, then:
    - Cell lines where TF_A is essential → TF_B is also essential (compensating)
    - Cell lines where one is disrupted → the other becomes more essential
    - This creates a positive CERES correlation across cell lines

  This is a completely INDEPENDENT validation from Norman 2019.

Input:
  data/raw/DepMap/CRISPRGeneEffect.csv
  data/processed/roes_all_tf_pairs.csv

Output:
  data/processed/depmap_coessentiality_tf_pairs.csv
  figures/roes_vs_depmap_validation.png
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROC = BASE_DIR / "data" / "processed"
FIGURES = BASE_DIR / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

def load_depmap_ceres():
    """Load DepMap CERES gene effect scores."""
    ceres_path = DATA_RAW / "DepMap" / "CRISPRGeneEffect.csv"
    if not ceres_path.exists():
        raise FileNotFoundError(
            f"DepMap data missing: {ceres_path}\n"
            "Download from: https://depmap.org/portal/download/all/?release=25Q2\n"
            "File: CRISPRGeneEffect.csv"
        )
    print(f"Loading DepMap CERES scores: {ceres_path}")
    df = pd.read_csv(ceres_path, index_col=0)
    print(f"  Shape: {df.shape} (cell_lines × genes)")

    # DepMap column format: "GENENAME (ENTREZID)" → extract gene name
    df.columns = [c.split(" (")[0] for c in df.columns]
    return df

def compute_coessentiality(ceres_df, tf_pairs):
    """
    Compute Pearson correlation of CERES scores for each TF pair.
    High positive correlation = co-essential = functionally redundant.
    """
    results = []
    genes_in_depmap = set(ceres_df.columns)

    print(f"Computing co-essentiality for {len(tf_pairs)} TF pairs...")
    skipped = 0

    for i, (_, row) in enumerate(tf_pairs.iterrows()):
        tf_a = row["TF_A"]
        tf_b = row["TF_B"]

        if tf_a not in genes_in_depmap or tf_b not in genes_in_depmap:
            skipped += 1
            continue

        scores_a = ceres_df[tf_a].dropna()
        scores_b = ceres_df[tf_b].dropna()

        # Use cell lines with both scores
        common_lines = scores_a.index.intersection(scores_b.index)
        if len(common_lines) < 50:
            skipped += 1
            continue

        a = scores_a[common_lines].values
        b = scores_b[common_lines].values

        r, p = pearsonr(a, b)
        rho, p_spear = spearmanr(a, b)

        results.append({
            "TF_A": tf_a,
            "TF_B": tf_b,
            "depmap_pearson_r": r,
            "depmap_pearson_p": p,
            "depmap_spearman_rho": rho,
            "depmap_spearman_p": p_spear,
            "n_cell_lines": len(common_lines),
        })

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(tf_pairs)} done ({skipped} skipped)")

    print(f"  Completed: {len(results)} pairs, {skipped} skipped (not in DepMap)")
    return pd.DataFrame(results)

def plot_roes_vs_depmap(merged_df):
    """
    Key figure: ROES (predicted redundancy) vs DepMap co-essentiality (observed).
    High ROES → high DepMap correlation validates our network topology approach.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: ROES jaccard_overlap vs DepMap pearson r
    ax = axes[0]
    scatter = ax.scatter(
        merged_df["jaccard_overlap"],
        merged_df["depmap_pearson_r"],
        alpha=0.4, s=20, c=merged_df["serial_score"],
        cmap="RdYlBu_r"
    )
    plt.colorbar(scatter, ax=ax, label="Serial connectivity score")
    ax.set_xlabel("ROES: Jaccard overlap (parallel redundancy)", fontsize=12)
    ax.set_ylabel("DepMap co-essentiality (Pearson r)", fontsize=12)

    # Add correlation
    valid = merged_df.dropna(subset=["jaccard_overlap", "depmap_pearson_r"])
    if len(valid) > 5:
        r, p = pearsonr(valid["jaccard_overlap"], valid["depmap_pearson_r"])
        ax.text(0.05, 0.95, f"r = {r:.3f}, p = {p:.3e}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_title("ROES vs DepMap: Parallel Redundancy Validation", fontsize=11)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Plot 2: Distribution of co-essentiality for high vs low ROES pairs
    ax = axes[1]
    median_roes = merged_df["jaccard_overlap"].median()
    high_roes = merged_df[merged_df["jaccard_overlap"] > median_roes]["depmap_pearson_r"].dropna()
    low_roes = merged_df[merged_df["jaccard_overlap"] <= median_roes]["depmap_pearson_r"].dropna()

    ax.hist(low_roes, bins=30, alpha=0.6, label=f"Low ROES (n={len(low_roes)})",
            color="steelblue", density=True)
    ax.hist(high_roes, bins=30, alpha=0.6, label=f"High ROES (n={len(high_roes)})",
            color="darkorange", density=True)

    from scipy.stats import mannwhitneyu
    if len(high_roes) > 5 and len(low_roes) > 5:
        stat, p_mw = mannwhitneyu(high_roes, low_roes, alternative="greater")
        ax.text(0.05, 0.95, f"Mann-Whitney p = {p_mw:.3e}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("DepMap co-essentiality (Pearson r)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("High-ROES pairs are more co-essential in DepMap", fontsize=11)
    ax.legend()

    plt.tight_layout()
    fig_path = FIGURES / "roes_vs_depmap_validation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig_path}")

def run_depmap_validation():
    roes_df = pd.read_csv(DATA_PROC / "roes_all_tf_pairs.csv")
    ceres_df = load_depmap_ceres()

    # Compute co-essentiality for all ROES pairs
    coess_df = compute_coessentiality(ceres_df, roes_df)

    out_path = DATA_PROC / "depmap_coessentiality_tf_pairs.csv"
    coess_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Merge with ROES
    merged = pd.merge(roes_df, coess_df, on=["TF_A", "TF_B"], how="inner")
    print(f"\nMerged ROES + DepMap: {len(merged)} pairs")

    # Validation stats
    print("\n=== Validation: ROES vs DepMap Co-essentiality ===")
    for roes_feat in ["jaccard_overlap", "n_shared_targets", "path_independence"]:
        valid = merged.dropna(subset=[roes_feat, "depmap_pearson_r"])
        if len(valid) > 5:
            r, p = pearsonr(valid[roes_feat], valid["depmap_pearson_r"])
            rho, p2 = spearmanr(valid[roes_feat], valid["depmap_pearson_r"])
            print(f"  {roes_feat:30s}: Pearson r={r:.3f} p={p:.4f} | Spearman ρ={rho:.3f} p={p2:.4f}")

    # Top predicted synthetic essential pairs (high ROES + high DepMap)
    merged["composite_score"] = (
        merged["jaccard_overlap"].rank(pct=True) +
        merged["depmap_pearson_r"].rank(pct=True)
    )
    top_pairs = merged.nlargest(20, "composite_score")[
        ["TF_A", "TF_B", "jaccard_overlap", "depmap_pearson_r", "serial_score", "n_cell_lines"]
    ]
    print(f"\nTop 20 synthetic essential TF pairs (network + DepMap concordant):")
    print(top_pairs.to_string(index=False))

    top_pairs.to_csv(DATA_PROC / "top_synthetic_essential_pairs.csv", index=False)

    # Generate validation figure
    plot_roes_vs_depmap(merged)

    return merged

if __name__ == "__main__":
    run_depmap_validation()
    print("\nNext step: python scripts/06_disease_mapping.py")

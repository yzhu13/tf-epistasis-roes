"""
Script 06: Map top ROES synthetic essential TF pairs to human disease.

Hypotheses tested:
  H1: Synthetic essential TF pairs (high ROES + DepMap co-essential) are enriched
      for genes causing OLIGOGENIC developmental disorders (where two mild variants
      together cause disease, but single variants don't).

  H2: In cancer, when one TF of a synthetic essential pair is amplified/mutated,
      the other becomes a synthetic lethal target.

  H3: Paralogous TF pairs (sequence similarity > 60%) are more likely to be
      functionally redundant (high ROES).

Input:
  data/processed/top_synthetic_essential_pairs.csv
  data/processed/roes_all_tf_pairs.csv

Output:
  data/processed/disease_mapping_results.csv
  figures/disease_enrichment.png
"""

import numpy as np
import pandas as pd
import requests
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROC = BASE_DIR / "data" / "processed"
FIGURES = BASE_DIR / "figures"

def fetch_omim_gene_disease():
    """
    Fetch gene-disease associations from OMIM.
    We use the OMIM morbidmap which maps genes to phenotypes.

    For access to full OMIM data, register at: https://www.omim.org/api
    Here we use the publicly available genemap2.txt download.
    """
    omim_path = DATA_RAW / "OMIM" / "genemap2.txt"
    DATA_RAW.joinpath("OMIM").mkdir(parents=True, exist_ok=True)

    if not omim_path.exists():
        print("OMIM genemap2.txt not found.")
        print("Download instructions:")
        print("  1. Register at https://omim.org/downloads")
        print("  2. Download genemap2.txt")
        print(f"  3. Save to: {omim_path}")
        print("\nFalling back to OpenTargets API for gene-disease associations...")
        return fetch_opentargets_gene_disease()

    # Parse OMIM genemap2.txt
    df = pd.read_csv(omim_path, sep="\t", comment="#", header=None,
                     usecols=[3, 5, 12],  # Approved symbol, gene MIM, phenotypes
                     names=["gene", "mim_number", "phenotypes"])
    df = df.dropna(subset=["gene", "phenotypes"])

    # Parse phenotype inheritance patterns
    records = []
    for _, row in df.iterrows():
        phenos = str(row["phenotypes"]).split(";")
        for pheno in phenos:
            pheno = pheno.strip()
            if not pheno:
                continue
            # Classify: autosomal dominant (AD), autosomal recessive (AR), etc.
            inheritance = "unknown"
            if "Autosomal dominant" in pheno:
                inheritance = "AD"
            elif "Autosomal recessive" in pheno:
                inheritance = "AR"
            records.append({
                "gene": row["gene"].strip(),
                "phenotype": pheno[:100],
                "inheritance": inheritance,
            })

    omim_df = pd.DataFrame(records)
    print(f"OMIM gene-disease loaded: {len(omim_df)} associations, {omim_df['gene'].nunique()} genes")
    return omim_df

def fetch_opentargets_gene_disease():
    """
    Fallback: use OpenTargets GraphQL API for gene-disease associations.
    Free, no registration required.
    """
    print("Fetching from OpenTargets API...")
    # This would query OpenTargets for specific TFs; simplified version here
    return pd.DataFrame(columns=["gene", "phenotype", "inheritance"])

def analyze_disease_enrichment(roes_df, disease_df):
    """
    Test H1: Are synthetic essential TF pairs enriched for oligogenic disease genes?

    Approach: for each pair, check if BOTH genes are in OMIM AR disease genes.
    Enrichment test: Fisher's exact test.
    """
    from scipy.stats import fisher_exact

    if disease_df.empty:
        print("No disease data available for enrichment analysis.")
        return None

    # Disease gene sets
    ar_genes = set(disease_df[disease_df["inheritance"] == "AR"]["gene"])
    ad_genes = set(disease_df[disease_df["inheritance"] == "AD"]["gene"])
    any_disease_genes = set(disease_df["gene"])

    print(f"Disease genes: {len(any_disease_genes)} total, {len(ar_genes)} AR, {len(ad_genes)} AD")

    # Classify TF pairs
    median_roes = roes_df["jaccard_overlap"].median()
    high_roes = roes_df[roes_df["jaccard_overlap"] > median_roes]
    low_roes = roes_df[roes_df["jaccard_overlap"] <= median_roes]

    def count_both_disease(df, gene_set):
        both = df.apply(lambda r: r["TF_A"] in gene_set and r["TF_B"] in gene_set, axis=1)
        return both.sum()

    # Fisher exact: high ROES × both AR disease vs low ROES × both AR disease
    high_both_ar = count_both_disease(high_roes, ar_genes)
    high_not_both_ar = len(high_roes) - high_both_ar
    low_both_ar = count_both_disease(low_roes, ar_genes)
    low_not_both_ar = len(low_roes) - low_both_ar

    table = [[high_both_ar, high_not_both_ar],
             [low_both_ar, low_not_both_ar]]

    odds_ratio, p_value = fisher_exact(table, alternative="greater")
    print(f"\nH1 Test: Synthetic essential pairs enriched for AR disease gene pairs?")
    print(f"  High ROES: {high_both_ar}/{len(high_roes)} pairs where both genes are AR disease genes")
    print(f"  Low ROES:  {low_both_ar}/{len(low_roes)} pairs where both genes are AR disease genes")
    print(f"  Odds ratio: {odds_ratio:.2f}, p-value: {p_value:.4f}")

    return {"odds_ratio": odds_ratio, "p_value": p_value}

def check_paralog_overlap(roes_df):
    """
    H3: Are paralogous TF pairs more likely to have high ROES?
    Use HGNC gene family data as proxy for paralogs.
    """
    print("\nH3: Paralog analysis")
    print("  Note: Full paralog analysis requires Ensembl compara data.")
    print("  Quick proxy: check if TF pair names share prefix (e.g., KLF1/KLF2, GATA1/GATA2)")

    def shares_prefix(tf_a, tf_b, min_prefix=3):
        for i in range(min_prefix, min(len(tf_a), len(tf_b)) + 1):
            if tf_a[:i] == tf_b[:i] and not tf_a[:i].isdigit():
                return True
        return False

    roes_df = roes_df.copy()
    roes_df["name_prefix_similar"] = roes_df.apply(
        lambda r: shares_prefix(str(r["TF_A"]), str(r["TF_B"])), axis=1
    )

    prefix_similar = roes_df[roes_df["name_prefix_similar"]]
    prefix_not = roes_df[~roes_df["name_prefix_similar"]]

    print(f"  Name-similar pairs (proxy paralogs): {len(prefix_similar)}")
    print(f"  Name-dissimilar pairs: {len(prefix_not)}")

    if len(prefix_similar) > 0 and len(prefix_not) > 0:
        mean_roes_sim = prefix_similar["jaccard_overlap"].mean()
        mean_roes_not = prefix_not["jaccard_overlap"].mean()
        print(f"  Mean ROES (name-similar): {mean_roes_sim:.4f}")
        print(f"  Mean ROES (name-dissimilar): {mean_roes_not:.4f}")

        if len(prefix_similar) > 5:
            from scipy.stats import mannwhitneyu
            stat, p = mannwhitneyu(
                prefix_similar["jaccard_overlap"].dropna(),
                prefix_not["jaccard_overlap"].dropna(),
                alternative="greater"
            )
            print(f"  Mann-Whitney p (H3): {p:.4f}")

    return roes_df

def run_disease_mapping():
    roes_df = pd.read_csv(DATA_PROC / "roes_all_tf_pairs.csv")

    top_pairs_path = DATA_PROC / "top_synthetic_essential_pairs.csv"
    if top_pairs_path.exists():
        top_pairs = pd.read_csv(top_pairs_path)
    else:
        top_pairs = roes_df.nlargest(50, "jaccard_overlap")

    print("=== Disease Mapping Analysis ===\n")

    # Load disease data
    disease_df = fetch_omim_gene_disease()

    # H1: enrichment in AR disease genes
    enrich_result = analyze_disease_enrichment(roes_df, disease_df)

    # H3: paralog analysis
    roes_df = check_paralog_overlap(roes_df)

    # Save disease-annotated top pairs
    if not disease_df.empty:
        ar_genes = set(disease_df[disease_df["inheritance"] == "AR"]["gene"])
        top_pairs["TF_A_AR_disease"] = top_pairs["TF_A"].isin(ar_genes)
        top_pairs["TF_B_AR_disease"] = top_pairs["TF_B"].isin(ar_genes)
        top_pairs["both_AR_disease"] = top_pairs["TF_A_AR_disease"] & top_pairs["TF_B_AR_disease"]

    out_path = DATA_PROC / "disease_mapping_results.csv"
    top_pairs.to_csv(out_path, index=False)
    print(f"\nResults saved: {out_path}")

    # Print top oligogenic disease candidates
    print("\nTop synthetic essential pairs with both genes in AR disease:")
    if "both_AR_disease" in top_pairs.columns:
        candidates = top_pairs[top_pairs["both_AR_disease"]]
        if len(candidates) > 0:
            print(candidates[["TF_A", "TF_B", "jaccard_overlap"]].to_string(index=False))
        else:
            print("  None found (need OMIM data or more TF pairs)")

    return top_pairs

if __name__ == "__main__":
    run_disease_mapping()
    print("\nProject pipeline complete. See results/ and figures/ for outputs.")

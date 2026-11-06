"""
Script C: Expand GRN using Replogle 2022 K562 genome-wide perturbation screen (gwps).

Steps:
  1. Download K562_gwps_normalized_bulk_01.h5ad from Figshare
  2. Explore data structure
  3. Build expanded GRN (z-score method) with TF filtering
  4. Merge with existing GRN
  5. Re-compute ROES for Norman TF pairs
  6. Identify new top-50 synthetic essential TF pairs

Output:
  data/raw/Replogle2022/K562_gwps_normalized_bulk.h5ad
  data/processed/grn_tf_target_network_expanded.csv
  results/C_gwps_grn/roes_expanded_grn.csv
  results/C_gwps_grn/top50_expanded_predictions.csv
  results/C_gwps_grn/summary.txt
  figures/C_gwps/grn_expansion_comparison.png
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import urllib.request
import sys
import os
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
DATA_RAW  = BASE_DIR / "data" / "raw"
DATA_PROC = BASE_DIR / "data" / "processed"
RESULTS   = BASE_DIR / "results" / "C_gwps_grn"
FIGURES   = BASE_DIR / "figures" / "C_gwps"

for d in [DATA_RAW / "Replogle2022", DATA_RAW / "TF_annotations",
          DATA_PROC, RESULTS, FIGURES]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Step 1: Download gwps data
# ─────────────────────────────────────────────────────────────────

GWPS_URL  = "https://ndownloader.figshare.com/files/35773217"
GWPS_PATH = DATA_RAW / "Replogle2022" / "K562_gwps_normalized_bulk.h5ad"

def download_with_progress(url, dest):
    print(f"Downloading {url}")
    print(f"  -> {dest}")
    last_pct = [-1]

    def reporthook(count, block_size, total_size):
        if total_size <= 0:
            return
        pct = int(count * block_size * 100 / total_size)
        pct = min(pct, 100)
        if pct != last_pct[0]:
            bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct}%", end="", flush=True)
            last_pct[0] = pct

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()
    print(f"  Downloaded: {dest.stat().st_size / 1e6:.1f} MB")

def step1_download():
    print("\n" + "="*60)
    print("STEP 1: Download gwps data")
    print("="*60)
    if GWPS_PATH.exists():
        print(f"Already exists ({GWPS_PATH.stat().st_size / 1e6:.1f} MB), skipping.")
        return
    download_with_progress(GWPS_URL, GWPS_PATH)

# ─────────────────────────────────────────────────────────────────
# Step 2: Explore data structure
# ─────────────────────────────────────────────────────────────────

def step2_explore():
    import anndata
    print("\n" + "="*60)
    print("STEP 2: Explore data structure")
    print("="*60)
    adata = anndata.read_h5ad(GWPS_PATH)
    print(f"Shape: {adata.shape}  (rows=perturbed genes, cols=measured genes)")
    print(f"\nobs columns: {list(adata.obs.columns)}")
    print(adata.obs.head())
    print(f"\nvar columns: {list(adata.var.columns)}")
    print(f"First 5 var_names: {list(adata.var_names[:5])}")

    import scipy.sparse
    X = adata.X
    if scipy.sparse.issparse(X):
        X = X.toarray()
    print(f"\nX dtype: {X.dtype}, min={X.min():.4f}, max={X.max():.4f}, mean={X.mean():.4f}")
    print(f"X[0,:5] = {X[0,:5]}")
    return adata

# ─────────────────────────────────────────────────────────────────
# Step 3: Build expanded GRN
# ─────────────────────────────────────────────────────────────────

SEED_TFS = [
    "CEBPA","SPI1","KLF1","GATA1","GATA2","GATA3","MYC","MYCN","TP53","TP73",
    "E2F1","E2F2","E2F3","E2F4","RB1","RUNX1","RUNX2","RUNX3","FLI1","ETV1",
    "ETV4","ETV5","ETV6","ERG","ETS1","ETS2","SP1","SP2","SP3","KLF4","KLF6",
    "KLF9","KLF13","YY1","MAX","MXI1","MNT","MAD1L1","NF1","NFE2L2","NFE2",
    "STAT1","STAT3","STAT5A","STAT5B","IRF1","IRF2","IRF3","IRF4","IRF8",
    "SMAD3","SMAD4","SMAD7","TGIF1","HIF1A","EPAS1","VHL","MED1","BRD4",
    "FOXA1","FOXA2","FOXA3","FOXO1","FOXO3","FOXM1","FOXP1","FOXP2","FOXP3",
    "PAX5","PAX6","PAX7","SOX2","SOX9","SOX17","OCT4","POU5F1","NANOG",
    "EBF1","EBF2","TCF7","TCF7L2","LEF1","CTNNB1","NOTCH1","HES1","HEY1",
    "MYB","MYBL1","MYBL2","TAL1","LMO2","LDB1","IKZF1","IKZF3",
    "BCL6","BACH1","BACH2","BTB","NFKB1","NFKB2","RELA","RELB","REL",
    "RBPJ","ASCL1","NEUROD1","NEUROG2","TBX21","EOMES","ZEB1","ZEB2","SNAI1",
    "SNAI2","TWIST1","TWIST2","VIM","CDH1","CDH2","HNF1A","HNF4A","PPARG",
    "PPARA","NR3C1","NR3C2","ESR1","ESR2","AR","VDR","RXRA","RARA","THRB",
    "NR1H3","NR1H4","PRDM1","PRDM2","PRDM16","PRDM14","KDM5C","KDM6A",
    "HDAC1","HDAC2","EP300","CREBBP","DNMT3A","DNMT3B","TET2","ASXL1",
    "EZH2","SUZ12","EED","BMI1","RING1","RNF2","PCGF1","CBX1","CBX2",
    "MLL1","KMT2A","KMT2B","KMT2C","KMT2D","NSD1","NSD2","SETD2",
    "CHD4","CHD7","CHD8","ARID1A","ARID1B","SMARCA4","SMARCA2","SMARCC1",
    "SMARCC2","ARID2","PBRM1","BRD7","CEBPB","CEBPD","CEBPG","CEBPE",
    "ATF1","ATF2","ATF3","ATF4","ATF6","CREB1","CREB3","CREBL2",
    "JUN","JUNB","JUND","FOS","FOSB","FOSL1","FOSL2",
    "HDAC3","HDAC6","HDAC7","HDAC8","SIRT1","SIRT2","PHF8","PHF19",
]

TF_PREFIXES = ("ZNF","KLF","FOX","GATA","SOX","PAX","HOX","ETS","IRF","STAT",
               "SMAD","RUNX","CEBP","ATF","JUN","FOS","NF","NR","HIF","EBF",
               "TBX","ZEB","TWIST","ASCL","NEUROD","PROX","PRDM","KDM","BCL",
               "MYB","MYC","TAL","LMO","IKZF","TCF","LEF","HES","HEY","MLL",
               "NSD","SETD","CHD","ARID","SMARC","PHF","EP3","CREBBP","BRD",
               "EZH","SUZ","BMI","RING","PCGF","CBX","DNMT","TET","ASXL")

def load_tf_list():
    """Load TF list: local file → AnimalTFDB4 download → seed + prefix heuristic."""
    tf_dir = DATA_RAW / "TF_annotations"
    # 1. Local file
    for fname in ["Homo_sapiens_TF", "human_TFs_Lambert2018.txt",
                  "human_TFs.txt", "TF_list.txt"]:
        p = tf_dir / fname
        if p.exists():
            try:
                df = pd.read_csv(p, sep="\t", header=None)
                tfs = set(df[0].dropna().astype(str).tolist())
                print(f"  Loaded TF list from {p}: {len(tfs)} TFs")
                return tfs
            except Exception as e:
                print(f"  Failed to read {p}: {e}")

    # 2. Download AnimalTFDB4
    url = "http://bioinfo.life.hust.edu.cn/AnimalTFDB4/static/download/TF_list_final/Homo_sapiens_TF"
    dest = tf_dir / "Homo_sapiens_TF"
    try:
        print(f"  Downloading TF list from AnimalTFDB4...")
        urllib.request.urlretrieve(url, dest)
        df = pd.read_csv(dest, sep="\t", header=0)
        # The file may have a 'Symbol' column or just gene names
        col = "Symbol" if "Symbol" in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]
        tfs = set(df[col].dropna().astype(str).tolist())
        print(f"  Downloaded AnimalTFDB4 TF list: {len(tfs)} TFs")
        return tfs
    except Exception as e:
        print(f"  Download failed: {e}")

    # 3. Fallback: seed + prefix heuristic (applied dynamically per gene set)
    print("  Using seed TF list + prefix heuristics (fallback)")
    return set(SEED_TFS)

def parse_gene_from_obs_name(obs_name):
    """
    Parse gene symbol from gwps obs_name.
    Format: '{idx}_{GENE}_{guideIDs}_{ENSGID}'
    e.g. '0_A1BG_P1_ENSG00000121410' -> 'A1BG'
         '10000_ZBTB46_P1P2_ENSG00000130584' -> 'ZBTB46'
    Strategy: second token split by '_', then verify it is uppercase letters
    """
    parts = obs_name.split("_")
    # parts[0] is the numeric index; parts[-1] starts with ENSG
    # gene name is parts[1] (may contain digits but is typically all-caps)
    if len(parts) >= 2:
        return parts[1]
    return obs_name

def build_gwps_grn(adata, tf_set, z_threshold=2.5, max_targets=500):
    """
    Build GRN from pseudo-bulk gwps data.

    Data format notes:
    - obs_names: '{idx}_{GENE}_{guideIDs}_{ENSGID}' — gene name is 2nd token
    - var_names: ENSG IDs; gene symbols in adata.var['gene_name']
    - X: already z-scored (values ~[-4, 4]); some inf values from CNV/outlier rows
    - Values already represent normalized perturbation effects vs control
      (from Replogle et al. — pseudo-bulk, mean-centered per gene across perts)

    Since X is already z-scored, we use it directly with threshold |z| > z_threshold.
    """
    import scipy.sparse

    print(f"\nBuilding gwps GRN (z-threshold={z_threshold}, already z-scored data)...")
    X = adata.X
    if scipy.sparse.issparse(X):
        X = X.toarray().astype(np.float64)
    else:
        X = np.array(X, dtype=np.float64)

    # Replace inf/-inf with NaN
    X[~np.isfinite(X)] = np.nan

    # Parse gene names from obs_names
    obs_gene_names = [parse_gene_from_obs_name(n) for n in adata.obs_names]

    # Use gene_name column from var for target gene names
    if "gene_name" in adata.var.columns:
        target_genes = list(adata.var["gene_name"])
    else:
        target_genes = list(adata.var_names)

    print(f"  Perturbations: {len(obs_gene_names)}, Targets: {len(target_genes)}")
    sample_genes = obs_gene_names[:10]
    print(f"  Sample pert gene names: {sample_genes}")
    finite_count = np.isfinite(X).sum()
    print(f"  Finite values: {finite_count:,} / {X.size:,}")

    # Expand tf_set with prefix heuristic on actual gene symbols
    expanded_tf = set(tf_set)
    for g in set(obs_gene_names):
        if g.startswith(TF_PREFIXES):
            expanded_tf.add(g)
    print(f"  TF set after prefix expansion: {len(expanded_tf)}")

    # X is already z-scored per column (across perturbations)
    # We can use it directly — values already represent z-scores
    # (verified: col_mean ~0, col_std ~0.1–1.4; data is normalized but
    #  not perfectly z-scored per column, so we re-normalize per column
    #  to get consistent z-scores, ignoring NaN rows)
    col_mean = np.nanmean(X, axis=0)  # (n_genes,)
    col_std  = np.nanstd(X, axis=0)
    col_std[col_std == 0] = 1.0
    # Re-center per column (already approximately done in the file)
    Z = (X - col_mean) / col_std  # shape (n_perts, n_genes)

    # De-duplicate: if same gene appears multiple times (multiple guides),
    # aggregate by taking the row with most finite values or mean
    # First build a mapping: gene -> list of row indices
    from collections import defaultdict
    gene_to_rows = defaultdict(list)
    for i, g in enumerate(obs_gene_names):
        gene_to_rows[g].append(i)

    edges = []
    n_tf_found = 0
    unique_genes = list(gene_to_rows.keys())
    for gene in unique_genes:
        if gene not in expanded_tf:
            continue
        n_tf_found += 1
        rows = gene_to_rows[gene]
        # Aggregate multiple guides: nanmean across rows
        if len(rows) == 1:
            z_row = Z[rows[0]]
        else:
            z_row = np.nanmean(Z[rows, :], axis=0)

        # Find significant targets
        finite_mask = np.isfinite(z_row)
        sig_mask = finite_mask & (np.abs(z_row) > z_threshold)
        sig_idx  = np.where(sig_mask)[0]

        if len(sig_idx) == 0:
            continue

        # Sort by |z| descending, keep top max_targets
        order = np.argsort(np.abs(z_row[sig_idx]))[::-1][:max_targets]
        sig_idx = sig_idx[order]

        for idx in sig_idx:
            tgt = target_genes[idx]
            z_val = float(z_row[idx])
            edges.append({
                "TF": gene,
                "target_gene": tgt,
                "weight": abs(z_val),
                "edge_type": "activation" if z_val > 0 else "repression",
                "source": "Replogle2022_gwps",
                "confidence": min(1.0, abs(z_val) / 5.0),
            })

    print(f"  Total TFs in gwps: {n_tf_found}")
    print(f"  Total gwps edges: {len(edges)}")
    if len(edges) == 0 and n_tf_found > 0:
        # Debug: show z-score range for a known TF
        ex_gene = [g for g in unique_genes if g in expanded_tf][:3]
        for eg in ex_gene:
            rows = gene_to_rows[eg]
            z_row = np.nanmean(Z[rows, :], axis=0) if len(rows) > 1 else Z[rows[0]]
            fin = np.isfinite(z_row)
            print(f"  Debug {eg}: finite={fin.sum()}, |z|>2.5: {(np.abs(z_row[fin])>2.5).sum()}, max|z|={np.nanmax(np.abs(z_row)):.3f}")
    df = pd.DataFrame(edges)
    return df

# ─────────────────────────────────────────────────────────────────
# Step 4: Merge with existing GRN
# ─────────────────────────────────────────────────────────────────

def merge_grns(existing_grn, gwps_grn):
    """
    Merge existing GRN with gwps GRN.
    Overlapping edges: max weight, n_sources += 1.
    New edges: add directly with source='Replogle2022_gwps'.
    """
    print("\nMerging GRNs...")
    existing_tfs  = set(existing_grn["TF"].unique())
    gwps_tfs      = set(gwps_grn["TF"].unique())
    new_tfs       = gwps_tfs - existing_tfs
    print(f"  Existing GRN: {len(existing_grn)} edges, {len(existing_tfs)} TFs")
    print(f"  gwps GRN:     {len(gwps_grn)} edges, {len(gwps_tfs)} TFs")
    print(f"  New TFs from gwps: {len(new_tfs)}")

    # Ensure existing_grn has n_sources column
    if "n_sources" not in existing_grn.columns:
        existing_grn = existing_grn.copy()
        existing_grn["n_sources"] = 1
    if "sources" not in existing_grn.columns:
        existing_grn = existing_grn.copy()
        existing_grn["sources"] = existing_grn.get("source", "existing")

    combined = pd.concat([existing_grn, gwps_grn], ignore_index=True)

    # Aggregate
    agg = (
        combined
        .groupby(["TF", "target_gene", "edge_type"])
        .agg(
            weight     = ("weight", "max"),
            confidence = ("confidence", "mean"),
            n_sources  = ("source", "nunique") if "source" in combined.columns else ("confidence", "count"),
            sources    = ("source", lambda x: "|".join(sorted(set(x.dropna().astype(str)))))
                         if "source" in combined.columns else ("confidence", "count"),
        )
        .reset_index()
    )

    agg["confidence"] = (agg["confidence"] * (1 + 0.3 * (agg["n_sources"] - 1))).clip(0, 1)

    existing_edges = set(zip(existing_grn["TF"], existing_grn["target_gene"]))
    new_edge_mask  = ~pd.Series(list(zip(agg["TF"], agg["target_gene"]))).isin(existing_edges)
    n_new_edges    = new_edge_mask.sum()

    print(f"  Merged GRN: {len(agg)} edges, {agg['TF'].nunique()} TFs")
    print(f"  New edges from gwps: {n_new_edges}")

    return agg, new_tfs, n_new_edges

# ─────────────────────────────────────────────────────────────────
# ROES computation (mirrors script 04)
# ─────────────────────────────────────────────────────────────────

def build_nx_graph(grn_df):
    G = nx.DiGraph()
    for _, row in grn_df.iterrows():
        G.add_edge(row["TF"], row["target_gene"],
                   weight=row.get("weight", 1.0),
                   edge_type=row.get("edge_type", "activation"))
    return G

def compute_roes(G, tf_a, tf_b):
    targets_a = set(G.successors(tf_a)) if tf_a in G else set()
    targets_b = set(G.successors(tf_b)) if tf_b in G else set()
    if not targets_a or not targets_b:
        return None
    shared = targets_a & targets_b
    union  = targets_a | targets_b
    jaccard = len(shared) / len(union) if union else 0.0
    # parallel fraction: fraction of shared targets
    parallel_frac = len(shared) / min(len(targets_a), len(targets_b)) if min(len(targets_a), len(targets_b)) > 0 else 0.0
    # antagonism
    antagonism = 0.0
    if shared:
        opp = sum(1 for t in shared
                  if G.get_edge_data(tf_a, t, {}).get("edge_type", "") !=
                     G.get_edge_data(tf_b, t, {}).get("edge_type", ""))
        antagonism = opp / len(shared)
    # serial
    serial_ab = 1.0 if tf_b in targets_a else 0.0
    serial_ba = 1.0 if tf_a in targets_b else 0.0
    serial_score = max(serial_ab, serial_ba)
    n_a, n_b = len(targets_a), len(targets_b)
    # synthetic score: shared targets that are parallel (not serial)
    synth_score = len(shared) * parallel_frac * (1 - serial_score) * (1 - antagonism)
    return {
        "TF_A": tf_a, "TF_B": tf_b,
        "jaccard_overlap": jaccard,
        "n_shared_targets": len(shared),
        "parallel_fraction": parallel_frac,
        "antagonism_score": antagonism,
        "serial_ab": serial_ab, "serial_ba": serial_ba,
        "serial_score": serial_score,
        "n_targets_A": n_a, "n_targets_B": n_b,
        "synth_score": synth_score,
    }

def compute_roes_for_pairs(G, pairs, label="pairs"):
    results = []
    for i, (a, b) in enumerate(pairs):
        r = compute_roes(G, a, b)
        if r:
            results.append(r)
        if (i+1) % 1000 == 0:
            print(f"  [{label}] {i+1}/{len(pairs)} done")
    return pd.DataFrame(results)

# ─────────────────────────────────────────────────────────────────
# Step 5: Re-compute ROES for Norman TF pairs
# ─────────────────────────────────────────────────────────────────

def step5_roes_norman(G_expanded, G_orig):
    print("\n" + "="*60)
    print("STEP 5: Re-compute ROES for Norman TF pairs")
    print("="*60)
    norman_roes_path = DATA_PROC / "roes_norman_tfs_ranked.csv"
    if not norman_roes_path.exists():
        print("  roes_norman_tfs_ranked.csv not found, skipping comparison.")
        return None, None

    norm_df = pd.read_csv(norman_roes_path)
    pairs   = list(zip(norm_df["TF_A"], norm_df["TF_B"]))
    print(f"  Norman pairs: {len(pairs)}")

    roes_exp  = compute_roes_for_pairs(G_expanded, pairs, "expanded")
    roes_orig_df = norm_df  # already computed

    roes_exp_path = RESULTS / "roes_expanded_grn.csv"
    roes_exp.to_csv(roes_exp_path, index=False)
    print(f"  Saved: {roes_exp_path}")

    # Compare
    merged = pd.merge(
        roes_orig_df[["TF_A","TF_B","synth_score"]].rename(columns={"synth_score":"synth_orig"}),
        roes_exp[["TF_A","TF_B","synth_score"]].rename(columns={"synth_score":"synth_exp"}),
        on=["TF_A","TF_B"], how="inner"
    )
    if len(merged) > 1:
        r, p = pearsonr(merged["synth_orig"], merged["synth_exp"])
        print(f"  Pearson r between orig and expanded synth_score: {r:.3f} (p={p:.4g})")
        print(f"  Mean synth_score orig: {merged['synth_orig'].mean():.2f}")
        print(f"  Mean synth_score exp:  {merged['synth_exp'].mean():.2f}")
    return roes_exp, merged

# ─────────────────────────────────────────────────────────────────
# Step 6: Top 50 new predictions from all TF pairs
# ─────────────────────────────────────────────────────────────────

def step6_top50(G_expanded, tf_set, roes_exp):
    print("\n" + "="*60)
    print("STEP 6: Top 50 new synthetic essential TF pair predictions")
    print("="*60)

    # Get TFs in expanded GRN that are also in tf_set (or prefix-expanded)
    grn_tfs = [n for n in G_expanded.nodes() if G_expanded.out_degree(n) > 0]
    tf_nodes = [t for t in grn_tfs if t in tf_set or t.startswith(TF_PREFIXES)]
    # Sort by out-degree, take top 300 for scalability
    tf_nodes = sorted(tf_nodes, key=lambda n: G_expanded.out_degree(n), reverse=True)[:300]
    print(f"  TF nodes to analyze: {len(tf_nodes)}")

    # Exclude pairs already in norman roes
    known_pairs = set()
    if roes_exp is not None and len(roes_exp) > 0:
        known_pairs = set(zip(roes_exp["TF_A"], roes_exp["TF_B"])) | \
                      set(zip(roes_exp["TF_B"], roes_exp["TF_A"]))

    pairs = []
    for i, a in enumerate(tf_nodes):
        for b in tf_nodes[i+1:]:
            if (a, b) not in known_pairs and (b, a) not in known_pairs:
                pairs.append((a, b))

    print(f"  New TF pairs to compute: {len(pairs)}")
    # Compute in chunks; for very large sets limit to reasonable number
    if len(pairs) > 50000:
        print(f"  Limiting to 50000 pairs (top TFs by degree)")
        pairs = pairs[:50000]

    roes_new = compute_roes_for_pairs(G_expanded, pairs, "new_pairs")
    print(f"  Valid ROES computed: {len(roes_new)}")

    if len(roes_new) == 0:
        print("  No new pairs found.")
        return pd.DataFrame()

    top50 = roes_new.nlargest(50, "synth_score")
    out_path = RESULTS / "top50_expanded_predictions.csv"
    top50.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    print("\n  Top 10 new synthetic essential predictions:")
    print(top50[["TF_A","TF_B","synth_score","n_shared_targets","jaccard_overlap"]].head(10).to_string(index=False))
    return top50

# ─────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────

def make_figures(existing_grn, expanded_grn, roes_comparison, top50):
    print("\nGenerating figures...")
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    # Panel A: GRN size comparison (bars)
    ax1 = fig.add_subplot(gs[0, 0])
    labels   = ["Original\nGRN", "Expanded\nGRN"]
    n_edges  = [len(existing_grn), len(expanded_grn)]
    n_tfs    = [existing_grn["TF"].nunique(), expanded_grn["TF"].nunique()]
    x = np.arange(2)
    ax1.bar(x - 0.2, n_edges, 0.35, label="Edges", color="steelblue")
    ax1b = ax1.twinx()
    ax1b.bar(x + 0.2, n_tfs, 0.35, label="TFs", color="coral")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel("# Edges", color="steelblue")
    ax1b.set_ylabel("# TFs", color="coral")
    ax1.set_title("A. GRN Size Comparison")
    lines1 = [plt.Rectangle((0,0),1,1, color="steelblue"), plt.Rectangle((0,0),1,1, color="coral")]
    ax1.legend(lines1, ["Edges", "TFs"], loc="upper left")

    # Panel B: TF out-degree distribution
    ax2 = fig.add_subplot(gs[0, 1])
    orig_deg = existing_grn.groupby("TF").size().values
    exp_deg  = expanded_grn.groupby("TF").size().values
    bins = np.logspace(0, np.log10(max(exp_deg.max(), orig_deg.max()) + 1), 30)
    ax2.hist(orig_deg, bins=bins, alpha=0.6, label="Original", color="steelblue")
    ax2.hist(exp_deg,  bins=bins, alpha=0.6, label="Expanded", color="coral")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("# Targets per TF"); ax2.set_ylabel("Count")
    ax2.set_title("B. TF Out-Degree Distribution")
    ax2.legend()

    # Panel C: ROES synth_score correlation (orig vs expanded)
    ax3 = fig.add_subplot(gs[0, 2])
    if roes_comparison is not None and len(roes_comparison) > 1:
        ax3.scatter(roes_comparison["synth_orig"], roes_comparison["synth_exp"],
                    alpha=0.5, s=20, color="purple")
        lim = max(roes_comparison[["synth_orig","synth_exp"]].max())
        ax3.plot([0, lim], [0, lim], "k--", lw=1)
        r, _ = pearsonr(roes_comparison["synth_orig"], roes_comparison["synth_exp"])
        ax3.set_xlabel("ROES Synth Score (Original GRN)")
        ax3.set_ylabel("ROES Synth Score (Expanded GRN)")
        ax3.set_title(f"C. ROES Comparison (r={r:.3f})")
    else:
        ax3.text(0.5, 0.5, "No comparison data", ha="center", va="center",
                 transform=ax3.transAxes)
        ax3.set_title("C. ROES Comparison")

    # Panel D: Top 50 predictions heatmap (synth_score)
    ax4 = fig.add_subplot(gs[1, :2])
    if top50 is not None and len(top50) > 0:
        top_plot = top50.head(20)
        pair_labels = [f"{a}|{b}" for a, b in zip(top_plot["TF_A"], top_plot["TF_B"])]
        ax4.barh(range(len(pair_labels)), top_plot["synth_score"].values[::-1],
                 color="darkorange", alpha=0.8)
        ax4.set_yticks(range(len(pair_labels)))
        ax4.set_yticklabels(pair_labels[::-1], fontsize=8)
        ax4.set_xlabel("Synthetic Essential Score")
        ax4.set_title("D. Top 20 New Synthetic Essential Predictions (Expanded GRN)")
    else:
        ax4.text(0.5, 0.5, "No predictions", ha="center", va="center",
                 transform=ax4.transAxes)
        ax4.set_title("D. Top 20 Predictions")

    # Panel E: Sources breakdown
    ax5 = fig.add_subplot(gs[1, 2])
    if "sources" in expanded_grn.columns:
        src_counts = expanded_grn["sources"].value_counts().head(10)
    elif "source" in expanded_grn.columns:
        src_counts = expanded_grn["source"].value_counts().head(10)
    else:
        src_counts = pd.Series({"Unknown": len(expanded_grn)})
    ax5.barh(range(len(src_counts)), src_counts.values, color="teal", alpha=0.8)
    ax5.set_yticks(range(len(src_counts)))
    ax5.set_yticklabels(src_counts.index, fontsize=8)
    ax5.set_xlabel("# Edges")
    ax5.set_title("E. Edge Sources in Expanded GRN")

    plt.suptitle("GRN Expansion: Replogle 2022 gwps Integration", fontsize=14, y=1.01)
    plt.tight_layout()
    fig_path = FIGURES / "grn_expansion_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig_path}")

# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────

def write_summary(existing_grn, expanded_grn, new_tfs, n_new_edges, roes_exp, top50):
    lines = []
    lines.append("GRN Expansion Summary — Replogle 2022 gwps Integration")
    lines.append("=" * 60)
    lines.append(f"Original GRN:  {len(existing_grn):,} edges, {existing_grn['TF'].nunique():,} TFs")
    lines.append(f"Expanded GRN:  {len(expanded_grn):,} edges, {expanded_grn['TF'].nunique():,} TFs")
    lines.append(f"New TFs added: {len(new_tfs):,}")
    lines.append(f"New edges added: {n_new_edges:,}")
    lines.append("")
    lines.append(f"New TFs (up to 50 shown):")
    for tf in sorted(new_tfs)[:50]:
        lines.append(f"  {tf}")
    if len(new_tfs) > 50:
        lines.append(f"  ... and {len(new_tfs)-50} more")
    lines.append("")
    if roes_exp is not None and len(roes_exp) > 0:
        lines.append(f"ROES re-computed for Norman pairs: {len(roes_exp):,}")
    if top50 is not None and len(top50) > 0:
        lines.append(f"Top 50 new synthetic essential predictions:")
        for _, row in top50.head(10).iterrows():
            lines.append(f"  {row['TF_A']:15s} {row['TF_B']:15s}  score={row['synth_score']:.2f}")
    txt = "\n".join(lines)
    summary_path = RESULTS / "summary.txt"
    summary_path.write_text(txt)
    print(f"\nSummary saved: {summary_path}")
    print(txt)

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    # Step 1
    step1_download()

    # Step 2
    adata = step2_explore()

    # Step 3
    print("\n" + "="*60)
    print("STEP 3: Build gwps GRN")
    print("="*60)
    tf_set = load_tf_list()
    gwps_grn = build_gwps_grn(adata, tf_set)
    if len(gwps_grn) == 0:
        print("  WARNING: No gwps edges built. Lowering z-threshold to 2.0 and retrying...")
        gwps_grn = build_gwps_grn(adata, tf_set, z_threshold=2.0, max_targets=300)

    # Step 4
    print("\n" + "="*60)
    print("STEP 4: Merge GRNs")
    print("="*60)
    existing_grn = pd.read_csv(DATA_PROC / "grn_tf_target_network.csv")
    expanded_grn, new_tfs, n_new_edges = merge_grns(existing_grn, gwps_grn)
    exp_path = DATA_PROC / "grn_tf_target_network_expanded.csv"
    expanded_grn.to_csv(exp_path, index=False)
    print(f"  Expanded GRN saved: {exp_path}")

    # Build NetworkX graphs
    G_expanded = build_nx_graph(expanded_grn)
    G_orig     = build_nx_graph(existing_grn)
    print(f"  G_expanded: {G_expanded.number_of_nodes()} nodes, {G_expanded.number_of_edges()} edges")
    print(f"  G_orig:     {G_orig.number_of_nodes()} nodes, {G_orig.number_of_edges()} edges")

    # Step 5
    roes_exp, roes_comparison = step5_roes_norman(G_expanded, G_orig)

    # Step 6
    top50 = step6_top50(G_expanded, tf_set | set(SEED_TFS), roes_exp)

    # Figures
    make_figures(existing_grn, expanded_grn, roes_comparison, top50)

    # Summary
    write_summary(existing_grn, expanded_grn, new_tfs, n_new_edges, roes_exp, top50)

    print("\nDone. All outputs in results/C_gwps_grn/ and figures/C_gwps/")

if __name__ == "__main__":
    main()

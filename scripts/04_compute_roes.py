"""
Script 04: Compute Regulatory Overlap Epistasis Score (ROES) for all TF pairs.

ROES is our core innovation: instead of simulating virtual KO with unreliable DL models,
we use gene regulatory network topology to predict epistasis type.

Theory:
  For TF pair (A, B), epistasis arises from their structural relationship in the GRN:

  1. PARALLEL REDUNDANCY: A and B both regulate the same target genes through independent
     paths (different enhancers, different binding sites). Single KO of either is buffered
     by the other. Double KO removes both → synthetic essentiality.

  2. SERIAL CONNECTIVITY: A regulates B (or B regulates A). Effect of double KO is not
     additive because one perturbation is "upstream" of the other → epistatic masking.

  3. INDEPENDENCE: A and B regulate largely non-overlapping target sets → additive effects.

  4. ANTAGONISM: A and B regulate shared targets in opposite directions → buffering.

ROES vector for pair (A, B):
  [parallel_overlap, path_independence, serial_score, antagonism_score, hub_centrality]

After computing ROES for Norman 2019 pairs, train a linear model to predict eps_l2
(validates our theory) and then extrapolate to all ~40,000 TF pair combinations.

Input:
  data/processed/grn_tf_target_network.csv  (from script 03)
  data/processed/epistasis_groundtruth_norman2019.csv  (from script 02)

Output:
  data/processed/roes_all_tf_pairs.csv
  data/processed/roes_model_validation.csv
"""

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
DATA_PROC = BASE_DIR / "data" / "processed"

# ─────────────────────────────────────────────────────────────────
# GRN Loading
# ─────────────────────────────────────────────────────────────────

def load_grn(grn_path=None):
    """
    Load TF→gene regulatory network as a directed NetworkX graph.

    If STREAM GRN not yet available, fall back to a proxy using
    Replogle 2022 correlation-based network (temporary measure).
    """
    if grn_path and Path(grn_path).exists():
        print(f"Loading GRN from {grn_path}")
        df = pd.read_csv(grn_path)
        G = nx.DiGraph()
        for _, row in df.iterrows():
            G.add_edge(row["TF"], row["target_gene"],
                       weight=row.get("weight", 1.0),
                       edge_type=row.get("edge_type", "activation"))
        print(f"  GRN: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    else:
        print("GRN file not found. Building proxy from Norman 2019 data...")
        return build_proxy_grn()

def build_proxy_grn():
    """
    Fallback: build a TF-target proxy GRN from Norman 2019 perturbation profiles.

    Logic: if knocking out TF_A significantly changes gene G's expression,
    then TF_A -> G is an edge (with weight = effect size).
    This is a coarse but empirically grounded proxy until STREAM GRN is ready.
    """
    import scanpy as sc
    from pathlib import Path

    h5ad = BASE_DIR / "data" / "raw" / "Norman2019" / "norman2019_perturbseq.h5ad"
    if not h5ad.exists():
        raise FileNotFoundError("Run 01_download_data.py first")

    adata = sc.read_h5ad(h5ad)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Identify perturbation column
    pert_col = None
    for col in ["perturbation", "condition", "gene", "perturbation_name"]:
        if col in adata.obs.columns:
            pert_col = col
            break

    import scipy.sparse
    X = adata.X
    if scipy.sparse.issparse(X):
        X = X.toarray()

    pert_labels = adata.obs[pert_col].values
    genes = list(adata.var_names)

    # Find control
    ctrl_mask = None
    for ctrl in ["control", "ctrl", "CTRL", "non-targeting", "NT"]:
        mask = np.array([ctrl.lower() in str(p).lower() for p in pert_labels])
        if mask.sum() > 10:
            ctrl_mask = mask
            break

    e_ctrl = X[ctrl_mask].mean(axis=0)

    G = nx.DiGraph()
    # Add TF->gene edges where single KO causes significant expression change
    threshold = 0.5  # log2FC threshold for calling TF-target relationship

    single_perts = [p for p in np.unique(pert_labels)
                    if '+' not in str(p) and ',' not in str(p)
                    and not any(c in str(p).lower() for c in ["control", "ctrl", "nt"])]

    for pert in single_perts[:100]:  # limit for speed in proxy mode
        mask = pert_labels == pert
        if mask.sum() < 5:
            continue
        e_pert = X[mask].mean(axis=0)
        delta = e_pert - e_ctrl
        # Add edges to top affected genes
        top_idx = np.where(np.abs(delta) > threshold)[0]
        for idx in top_idx[:50]:  # top 50 targets per TF
            direction = "activation" if delta[idx] > 0 else "repression"
            G.add_edge(pert, genes[idx], weight=abs(delta[idx]), edge_type=direction)

    print(f"  Proxy GRN: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Save proxy GRN
    edges = [(u, v, G[u][v]["weight"], G[u][v]["edge_type"])
             for u, v in G.edges()]
    df = pd.DataFrame(edges, columns=["TF", "target_gene", "weight", "edge_type"])
    df.to_csv(DATA_PROC / "grn_proxy_norman2019.csv", index=False)

    return G

# ─────────────────────────────────────────────────────────────────
# ROES Computation
# ─────────────────────────────────────────────────────────────────

def get_tf_targets(G, tf_name, max_depth=2):
    """Get all target genes reachable from a TF within max_depth hops."""
    if tf_name not in G:
        return set()
    try:
        targets = nx.descendants(G, tf_name)
        # Filter to only gene nodes (not TF nodes for depth-1)
        return targets
    except Exception:
        return set(G.successors(tf_name))

def get_direct_targets(G, tf_name):
    """Get direct target genes of a TF (depth 1)."""
    if tf_name not in G:
        return set()
    return set(G.successors(tf_name))

def compute_path_independence(G, tf_a, tf_b, shared_targets):
    """
    For each shared target, check if paths from A and B are independent
    (don't share intermediate nodes). Independence → true parallel redundancy.
    """
    if len(shared_targets) == 0:
        return 0.0

    independent_count = 0
    for target in list(shared_targets)[:20]:  # sample for speed
        try:
            paths_a = list(nx.all_simple_paths(G, tf_a, target, cutoff=3))
            paths_b = list(nx.all_simple_paths(G, tf_b, target, cutoff=3))
            if not paths_a or not paths_b:
                continue
            # Check if any path pair shares no intermediate nodes
            for pa in paths_a:
                intermediates_a = set(pa[1:-1])
                for pb in paths_b:
                    intermediates_b = set(pb[1:-1])
                    if not (intermediates_a & intermediates_b):
                        independent_count += 1
                        break
        except Exception:
            pass

    return independent_count / len(shared_targets)

def compute_roes(G, tf_a, tf_b):
    """
    Compute ROES (Regulatory Overlap Epistasis Score) for a TF pair.

    Returns a dict of features used to predict epistasis type.
    """
    targets_a = get_direct_targets(G, tf_a)
    targets_b = get_direct_targets(G, tf_b)

    if not targets_a or not targets_b:
        return None

    # 1. Parallel overlap: shared targets (Jaccard similarity)
    shared = targets_a & targets_b
    union = targets_a | targets_b
    jaccard = len(shared) / len(union) if union else 0.0

    # 2. Path independence score
    path_indep = compute_path_independence(G, tf_a, tf_b, shared)

    # 3. Serial connectivity: does A regulate B or vice versa?
    serial_ab = 1.0 if (tf_b in targets_a or nx.has_path(G, tf_a, tf_b) if tf_b in G else False) else 0.0
    serial_ba = 1.0 if (tf_a in targets_b or nx.has_path(G, tf_b, tf_a) if tf_a in G else False) else 0.0
    serial_score = max(serial_ab, serial_ba)

    # 4. Antagonism: shared targets regulated in opposite directions
    antagonism = 0.0
    if shared and G.number_of_edges() > 0:
        opposite_count = 0
        for t in shared:
            edge_a = G.get_edge_data(tf_a, t, default={})
            edge_b = G.get_edge_data(tf_b, t, default={})
            dir_a = edge_a.get("edge_type", "")
            dir_b = edge_b.get("edge_type", "")
            if dir_a and dir_b and dir_a != dir_b:
                opposite_count += 1
        antagonism = opposite_count / len(shared) if shared else 0.0

    # 5. Hub centrality (both TFs highly connected = more likely to have broad epistasis)
    try:
        cent_a = nx.out_degree_centrality(G).get(tf_a, 0.0)
        cent_b = nx.out_degree_centrality(G).get(tf_b, 0.0)
        hub_score = (cent_a + cent_b) / 2
    except Exception:
        hub_score = 0.0

    # 6. Absolute sizes
    size_a = len(targets_a)
    size_b = len(targets_b)
    shared_n = len(shared)

    return {
        "TF_A": tf_a,
        "TF_B": tf_b,
        "jaccard_overlap": jaccard,
        "n_shared_targets": shared_n,
        "path_independence": path_indep,
        "serial_score": serial_score,
        "serial_ab": serial_ab,
        "serial_ba": serial_ba,
        "antagonism_score": antagonism,
        "hub_score": hub_score,
        "n_targets_A": size_a,
        "n_targets_B": size_b,
    }

# ─────────────────────────────────────────────────────────────────
# Validation with Norman 2019
# ─────────────────────────────────────────────────────────────────

def validate_roes_on_norman(roes_df, gt_df):
    """
    Train a linear model to predict epistasis magnitude from ROES features.
    Validate with leave-one-out cross-validation.

    This validates: can network topology predict empirical epistasis?
    """
    feature_cols = [
        "jaccard_overlap", "n_shared_targets", "path_independence",
        "serial_score", "antagonism_score", "hub_score"
    ]

    # Merge ROES with ground truth
    merged = pd.merge(
        gt_df[["TF_A", "TF_B", "eps_l2", "eps_relative", "eps_type"]],
        roes_df[["TF_A", "TF_B"] + feature_cols],
        on=["TF_A", "TF_B"], how="inner"
    )
    # Also try reverse order
    merged2 = pd.merge(
        gt_df[["TF_A", "TF_B", "eps_l2", "eps_relative", "eps_type"]],
        roes_df.rename(columns={"TF_A": "TF_B", "TF_B": "TF_A"})[["TF_A", "TF_B"] + feature_cols],
        on=["TF_A", "TF_B"], how="inner"
    )
    merged = pd.concat([merged, merged2]).drop_duplicates()

    print(f"\nValidation set: {len(merged)} pairs with both ROES and ground truth")

    if len(merged) < 5:
        print("  Too few overlapping pairs for validation. Saving ROES for all pairs.")
        return None

    X = merged[feature_cols].fillna(0).values
    y = merged["eps_l2"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Linear model (Ridge regression — validated by Nature Methods 2025 finding)
    model = Ridge(alpha=1.0)
    loo_scores = cross_val_score(model, X_scaled, y, cv=min(10, len(merged)),
                                  scoring="r2")

    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    r, p = pearsonr(y, y_pred)
    rho, p_spear = spearmanr(y, y_pred)

    print(f"  Ridge regression R² (CV): {loo_scores.mean():.3f} ± {loo_scores.std():.3f}")
    print(f"  Pearson r (train):  {r:.3f}  p={p:.4f}")
    print(f"  Spearman ρ (train): {rho:.3f}  p={p_spear:.4f}")
    print(f"\n  Feature importances:")
    for feat, coef in zip(feature_cols, model.coef_):
        print(f"    {feat:30s}: {coef:.4f}")

    merged["eps_l2_predicted"] = y_pred
    val_path = DATA_PROC / "roes_model_validation.csv"
    merged.to_csv(val_path, index=False)
    print(f"\n  Validation results saved: {val_path}")

    return model, scaler

def run_roes_analysis():
    # Load GRN
    grn_path = DATA_PROC / "grn_tf_target_network.csv"
    G = load_grn(grn_path if grn_path.exists() else None)

    # Get all TF nodes in the GRN
    all_tfs = [n for n in G.nodes() if G.out_degree(n) > 0]
    print(f"\nTFs in GRN: {len(all_tfs)}")

    # Load ground truth
    gt_path = DATA_PROC / "epistasis_groundtruth_norman2019.csv"
    if gt_path.exists():
        gt_df = pd.read_csv(gt_path)
        print(f"Loaded ground truth: {len(gt_df)} pairs")
        validation_pairs = set(
            list(zip(gt_df["TF_A"], gt_df["TF_B"])) +
            list(zip(gt_df["TF_B"], gt_df["TF_A"]))
        )
    else:
        gt_df = None
        validation_pairs = set()
        print("No ground truth found. Run 02_epistasis_groundtruth.py first.")

    # Compute ROES for all TF pairs
    # Priority: first do validation pairs (from Norman 2019), then expand
    all_pairs = []
    done = set()

    # Validation pairs first
    for tf_a, tf_b in validation_pairs:
        if (tf_a, tf_b) in done or (tf_b, tf_a) in done:
            continue
        roes = compute_roes(G, tf_a, tf_b)
        if roes:
            all_pairs.append(roes)
            done.add((tf_a, tf_b))

    print(f"\nComputed ROES for {len(all_pairs)} validation pairs")

    # Expand to all hub TF pairs (top 200 by out-degree)
    hub_tfs = sorted(all_tfs, key=lambda n: G.out_degree(n), reverse=True)[:200]
    print(f"Expanding to all {len(hub_tfs)}×{len(hub_tfs)} hub TF pairs...")

    total = len(hub_tfs) * (len(hub_tfs) - 1) // 2
    i = 0
    for idx_a, tf_a in enumerate(hub_tfs):
        for tf_b in hub_tfs[idx_a+1:]:
            if (tf_a, tf_b) in done or (tf_b, tf_a) in done:
                continue
            roes = compute_roes(G, tf_a, tf_b)
            if roes:
                all_pairs.append(roes)
                done.add((tf_a, tf_b))
            i += 1
            if i % 1000 == 0:
                print(f"  {i}/{total} pairs computed ({len(all_pairs)} valid)")

    roes_df = pd.DataFrame(all_pairs)
    roes_path = DATA_PROC / "roes_all_tf_pairs.csv"
    roes_df.to_csv(roes_path, index=False)
    print(f"\nROES saved: {roes_path} ({len(roes_df)} pairs)")

    # Validate against Norman 2019 ground truth
    if gt_df is not None and len(roes_df) > 0:
        validate_roes_on_norman(roes_df, gt_df)

    # Top synthetic essential predictions
    print(f"\nTop 20 predicted SYNTHETIC ESSENTIAL pairs (high parallel redundancy):")
    top_se = roes_df.nlargest(20, "jaccard_overlap")[
        ["TF_A", "TF_B", "jaccard_overlap", "n_shared_targets", "path_independence", "serial_score"]
    ]
    print(top_se.to_string(index=False))

    return roes_df

if __name__ == "__main__":
    roes_df = run_roes_analysis()
    print("\nNext step: python scripts/05_validate_depmap.py")

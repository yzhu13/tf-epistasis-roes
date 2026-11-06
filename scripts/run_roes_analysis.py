"""
Combined ROES analysis: build GRN -> compute ROES -> validate against Norman 2019.
Run: python scripts/run_roes_analysis.py
"""
import numpy as np
import pandas as pd
import networkx as nx
import scanpy as sc
import scipy.sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = Path(__file__).parent.parent
DATA_PROC = BASE / "data" / "processed"
DATA_PROC.mkdir(exist_ok=True)

# ─── 1. Build GRN ───────────────────────────────────────────────
def build_grn():
    grn_path = DATA_PROC / "grn_tf_target_network.csv"
    if grn_path.exists():
        grn = pd.read_csv(grn_path)
        print(f"Loaded GRN: {len(grn)} edges, {grn['TF'].nunique()} TFs")
    else:
        print("Building GRN...")
        edges = []

        # Norman 2019
        adata_n = sc.read_h5ad(BASE / "data/raw/Norman2019/norman2019_perturbseq.h5ad")
        X_n = adata_n.X
        if scipy.sparse.issparse(X_n): X_n = X_n.toarray()
        genes_n = list(adata_n.var_names)
        perts_n = adata_n.obs["guide_identity"].values
        ctrl_mask = np.array([all(p.startswith("NegCtrl") for p in v.split("__")[0].split("_")) for v in perts_n])
        e_ctrl_n = X_n[ctrl_mask].mean(axis=0)
        for key in np.unique(perts_n):
            real = [p for p in key.split("__")[0].split("_") if not p.startswith("NegCtrl")]
            if len(real) != 1: continue
            tf = real[0]
            mask = perts_n == key
            if mask.sum() < 5: continue
            delta = X_n[mask].mean(axis=0) - e_ctrl_n
            for idx in np.where(np.abs(delta) > 0.3)[0]:
                edges.append({"TF": tf, "target_gene": genes_n[idx],
                              "weight": abs(float(delta[idx])),
                              "edge_type": "activation" if delta[idx] > 0 else "repression",
                              "source": "Norman2019"})

        # Replogle 2022
        adata_r = sc.read_h5ad(BASE / "data/raw/Replogle2022/K562_essential_raw_bulk.h5ad")
        X_r = adata_r.X
        if scipy.sparse.issparse(X_r): X_r = X_r.toarray()
        gene_names = adata_r.var["gene_name"].values if "gene_name" in adata_r.var.columns else adata_r.var_names

        def parse_gene(obs_name):
            parts = obs_name.split("_")
            for p in parts:
                if not p[0].isdigit() and p not in ("P1","P2","P1P2","P3") and not p.startswith("ENSG"):
                    return p
            return parts[1] if len(parts) > 1 else obs_name

        kd_genes = [parse_gene(n) for n in adata_r.obs_names]
        for row_idx, kd_gene in enumerate(kd_genes):
            expr = X_r[row_idx]
            mean_e, std_e = np.mean(expr), np.std(expr)
            z = (expr - mean_e) / (std_e + 1e-10)
            for idx in np.where(np.abs(z) > 2.5)[0][:30]:
                target = str(gene_names[idx])
                if target == kd_gene: continue
                edges.append({"TF": kd_gene, "target_gene": target,
                              "weight": abs(float(z[idx])),
                              "edge_type": "activation" if z[idx] > 0 else "repression",
                              "source": "Replogle2022"})

        grn = (pd.DataFrame(edges)
               .groupby(["TF", "target_gene", "edge_type"])
               .agg(weight=("weight","max"), n_sources=("source","nunique"),
                    sources=("source", lambda x: "|".join(sorted(set(x)))))
               .reset_index())
        grn.to_csv(grn_path, index=False)
        print(f"GRN saved: {len(grn)} edges, {grn['TF'].nunique()} TFs")

    G = nx.DiGraph()
    for _, row in grn.iterrows():
        G.add_edge(row["TF"], row["target_gene"],
                   weight=row["weight"], edge_type=row["edge_type"])
    return G

# ─── 2. ROES computation ────────────────────────────────────────
def compute_roes(G, tf_a, tf_b):
    if tf_a not in G or tf_b not in G:
        return None
    ta = set(G.successors(tf_a))
    tb = set(G.successors(tf_b))
    if not ta or not tb:
        return None

    ta_act = {t for t in ta if G[tf_a][t].get("edge_type","") == "activation"}
    ta_rep = {t for t in ta if G[tf_a][t].get("edge_type","") == "repression"}
    tb_act = {t for t in tb if G[tf_b][t].get("edge_type","") == "activation"}
    tb_rep = {t for t in tb if G[tf_b][t].get("edge_type","") == "repression"}

    shared = ta & tb
    union = ta | tb
    jaccard = len(shared) / len(union) if union else 0.0
    parallel_same = len((ta_act & tb_act) | (ta_rep & tb_rep))
    parallel_n = len(shared)
    parallel_frac = parallel_same / parallel_n if parallel_n > 0 else 0.0
    antagonism = len((ta_act & tb_rep) | (ta_rep & tb_act)) / parallel_n if parallel_n > 0 else 0.0

    return {
        "TF_A": tf_a, "TF_B": tf_b,
        "jaccard_overlap": jaccard,
        "n_shared_targets": len(shared),
        "parallel_fraction": parallel_frac,
        "antagonism_score": antagonism,
        "serial_ab": 1.0 if tf_b in ta else 0.0,
        "serial_ba": 1.0 if tf_a in tb else 0.0,
        "serial_score": float(tf_b in ta or tf_a in tb),
        "n_targets_A": len(ta),
        "n_targets_B": len(tb),
    }

# ─── 3. Main ────────────────────────────────────────────────────
G = build_grn()
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

gt = pd.read_csv(DATA_PROC / "epistasis_groundtruth_norman2019.csv")
print(f"Ground truth pairs: {len(gt)}")

# ROES for GT pairs
roes_gt = []
for _, row in gt.iterrows():
    r = compute_roes(G, row["TF_A"], row["TF_B"]) or compute_roes(G, row["TF_B"], row["TF_A"])
    if r:
        r.update({"eps_l2": row["eps_l2"], "eps_relative": row["eps_relative"], "eps_type": row["eps_type"]})
        roes_gt.append(r)
    else:
        print(f"  Missing GRN for: {row['TF_A']} / {row['TF_B']}")

df_val = pd.DataFrame(roes_gt)
df_val.to_csv(DATA_PROC / "roes_validation_norman2019.csv", index=False)
print(f"\nROES validation table ({len(df_val)} pairs):")
print(df_val[["TF_A","TF_B","jaccard_overlap","n_shared_targets","parallel_fraction","eps_l2","eps_type"]].to_string(index=False))

# Correlation
print("\n=== Can ROES predict epistasis? ===")
for feat in ["jaccard_overlap", "n_shared_targets", "parallel_fraction", "antagonism_score"]:
    v = df_val.dropna(subset=[feat, "eps_l2"])
    if len(v) > 3:
        r, p = pearsonr(v[feat], v["eps_l2"])
        rho, p2 = spearmanr(v[feat], v["eps_l2"])
        print(f"  {feat}: Pearson r={r:+.3f} (p={p:.3f})  Spearman rho={rho:+.3f} (p={p2:.3f})")

# Linear model (Ridge)
feat_cols = ["jaccard_overlap", "n_shared_targets", "parallel_fraction", "antagonism_score", "serial_score"]
v = df_val.dropna(subset=feat_cols + ["eps_l2"])
if len(v) >= 5:
    from sklearn.model_selection import cross_val_score
    X_feat = v[feat_cols].values
    y = v["eps_l2"].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_feat)
    ridge = Ridge(alpha=1.0)
    cv_r2 = cross_val_score(ridge, X_sc, y, cv=min(5, len(v)), scoring="r2")
    print(f"\n  Ridge LOO R2 (n={len(v)}): {cv_r2.mean():.3f} +/- {cv_r2.std():.3f}")

# ─── 4. Scale to all hub TF pairs ───────────────────────────────
hub_tfs = sorted([tf for tf in G.nodes() if G.out_degree(tf) > 5])
print(f"\nComputing ROES for all {len(hub_tfs)} hub TFs x hub TFs ({len(hub_tfs)*(len(hub_tfs)-1)//2} pairs)...")

all_roes = []
for i, tf_a in enumerate(hub_tfs):
    for tf_b in hub_tfs[i+1:]:
        r = compute_roes(G, tf_a, tf_b)
        if r: all_roes.append(r)

roes_all = pd.DataFrame(all_roes)
roes_all.to_csv(DATA_PROC / "roes_all_tf_pairs.csv", index=False)
print(f"Total ROES pairs: {len(roes_all)}")

print("\nTop 20 predicted SYNTHETIC ESSENTIAL TF pairs (high jaccard + parallel):")
roes_all["synth_essential_score"] = roes_all["jaccard_overlap"] * roes_all["parallel_fraction"]
top = roes_all.nlargest(20, "synth_essential_score")
print(top[["TF_A","TF_B","jaccard_overlap","n_shared_targets","parallel_fraction","antagonism_score"]].to_string(index=False))

print("\nDone. Next: python scripts/05_validate_depmap.py")

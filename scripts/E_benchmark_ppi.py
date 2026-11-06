"""
E_benchmark_ppi.py
==================
ROES Benchmark + STRING PPI Integration
Tasks E1-E5: Download STRING, compute PPI features, benchmark all methods,
feature importance, and visualization.
"""

import os
import sys
import gzip
import time
import urllib.request
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA_PROC  = BASE / "data" / "processed"
DATA_RAW   = BASE / "data" / "raw"
STRING_DIR = DATA_RAW / "STRING"
RES_DIR    = BASE / "results" / "E_benchmark"
FIG_DIR    = BASE / "figures" / "E_benchmark"

for d in [STRING_DIR, RES_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("E_benchmark_ppi.py  –  ROES Benchmark + STRING PPI")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# E1: Download STRING PPI
# ══════════════════════════════════════════════════════════════════════════════
print("\n── E1: STRING PPI Download ──")

STRING_LINKS   = STRING_DIR / "9606.protein.links.v12.0.txt.gz"
STRING_ALIASES = STRING_DIR / "9606.protein.aliases.v12.0.txt.gz"

LINKS_URLS = [
    "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
    "https://version-12-0.string-db.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
]
ALIASES_URLS = [
    "https://stringdb-downloads.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz",
    "https://version-12-0.string-db.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz",
]

def try_download(urls, dest, label):
    if dest.exists():
        print(f"  [OK] {label} already exists ({dest.stat().st_size/1e6:.1f} MB)")
        return True
    for url in urls:
        try:
            print(f"  Trying {url} ...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest, 'wb') as f:
                total = 0
                while True:
                    chunk = resp.read(1024*1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
                    print(f"\r  Downloaded {total/1e6:.1f} MB", end='', flush=True)
            print(f"\n  [OK] Saved to {dest}")
            return True
        except Exception as e:
            print(f"  [FAIL] {e}")
    return False

links_ok   = try_download(LINKS_URLS,   STRING_LINKS,   "STRING protein.links")
aliases_ok = try_download(ALIASES_URLS, STRING_ALIASES, "STRING protein.aliases")

STRING_AVAILABLE = links_ok and aliases_ok

# ── Load STRING ────────────────────────────────────────────────────────────────
ppi_gene_graph = None   # will be nx.Graph keyed by gene symbol
id2gene = {}            # STRING_ID -> gene symbol

if STRING_AVAILABLE:
    print("\n  Loading STRING aliases (ID -> gene name) ...")
    try:
        # Preferred source: Ensembl_HGNC_symbol (official HGNC gene symbol)
        # Fallback: Ensembl_HGNC_alias_symbol
        alias_chunks = []
        with gzip.open(STRING_ALIASES, 'rt') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue  # header
                parts = line.rstrip('\n').split('\t')
                if len(parts) >= 3 and parts[2] == 'Ensembl_HGNC_symbol':
                    alias_chunks.append((parts[0], parts[1]))
        if not alias_chunks:
            # fallback: any HGNC symbol source
            with gzip.open(STRING_ALIASES, 'rt') as f:
                for i, line in enumerate(f):
                    if i == 0:
                        continue
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) >= 3 and 'HGNC_symbol' in parts[2]:
                        alias_chunks.append((parts[0], parts[1]))

        id2gene = dict(alias_chunks)
        print(f"  Loaded {len(id2gene):,} ID→gene mappings")

        print("  Loading STRING protein.links (score >= 700) ...")
        edges = []
        with gzip.open(STRING_LINKS, 'rt') as f:
            next(f)  # skip header
            for line in f:
                p = line.split()
                score = int(p[2])
                if score >= 700:
                    edges.append((p[0], p[1], score))

        print(f"  {len(edges):,} high-confidence edges (score>=700)")

        # Build gene-level graph
        ppi_gene_graph = nx.Graph()
        skipped = 0
        for p1, p2, score in edges:
            g1 = id2gene.get(p1)
            g2 = id2gene.get(p2)
            if g1 and g2:
                if ppi_gene_graph.has_edge(g1, g2):
                    if ppi_gene_graph[g1][g2]['score'] < score:
                        ppi_gene_graph[g1][g2]['score'] = score
                else:
                    ppi_gene_graph.add_edge(g1, g2, score=score)
            else:
                skipped += 1
        print(f"  PPI gene graph: {ppi_gene_graph.number_of_nodes():,} nodes, "
              f"{ppi_gene_graph.number_of_edges():,} edges  (skipped {skipped:,} unmapped)")
    except Exception as e:
        print(f"  [ERROR] loading STRING: {e}")
        STRING_AVAILABLE = False
        ppi_gene_graph = None

# ══════════════════════════════════════════════════════════════════════════════
# Load primary datasets
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Loading validation datasets ──")
depmap_df  = pd.read_csv(DATA_PROC / "depmap_coessentiality_results.csv")
norman_df  = pd.read_csv(DATA_PROC / "roes_validation_norman2019.csv")
grn_df     = pd.read_csv(DATA_PROC / "grn_tf_target_network.csv")

print(f"  DepMap: {len(depmap_df)} pairs")
print(f"  Norman: {len(norman_df)} pairs")
print(f"  GRN edges: {len(grn_df)}")

# Build GRN target sets per TF
grn_targets = {}
tf_col = [c for c in grn_df.columns if 'tf' in c.lower() or 'source' in c.lower()][0]
tg_col = [c for c in grn_df.columns if 'target' in c.lower() or 'gene' in c.lower()][0]
for _, row in grn_df.iterrows():
    tf = row[tf_col]
    tg = row[tg_col]
    grn_targets.setdefault(tf, set()).add(tg)

# ══════════════════════════════════════════════════════════════════════════════
# E2: Compute PPI features
# ══════════════════════════════════════════════════════════════════════════════
print("\n── E2: Computing PPI features ──")

def compute_ppi_features(tf_a, tf_b, ppi_graph, grn_targets):
    feats = {
        'ppi_direct': 0,
        'ppi_score': 0.0,
        'ppi_shared_partners': 0.0,
        'ppi_n_shared_partners': 0,
        'same_complex_proxy': 0,
    }
    if ppi_graph is None:
        return feats

    # Direct interaction
    if ppi_graph.has_edge(tf_a, tf_b):
        feats['ppi_direct'] = 1
        feats['ppi_score']  = ppi_graph[tf_a][tf_b]['score'] / 1000.0

    # Shared partners (Jaccard)
    nbr_a = set(ppi_graph.neighbors(tf_a)) if tf_a in ppi_graph else set()
    nbr_b = set(ppi_graph.neighbors(tf_b)) if tf_b in ppi_graph else set()
    union = nbr_a | nbr_b
    inter = nbr_a & nbr_b
    if union:
        feats['ppi_shared_partners']   = len(inter) / len(union)
        feats['ppi_n_shared_partners'] = len(inter)

    # Same-complex proxy
    shared_grn = len(grn_targets.get(tf_a, set()) & grn_targets.get(tf_b, set()))
    if feats['ppi_direct'] == 1 and shared_grn > 20:
        feats['same_complex_proxy'] = 1

    return feats

ppi_rows = []
for _, row in depmap_df.iterrows():
    f = compute_ppi_features(row['TF_A'], row['TF_B'], ppi_gene_graph, grn_targets)
    f.update({'TF_A': row['TF_A'], 'TF_B': row['TF_B']})
    ppi_rows.append(f)

ppi_df = pd.DataFrame(ppi_rows)
ppi_df.to_csv(RES_DIR / "ppi_features_all_pairs.csv", index=False)
print(f"  PPI features computed for {len(ppi_df)} pairs")
if STRING_AVAILABLE:
    print(f"  ppi_direct=1: {ppi_df['ppi_direct'].sum()} pairs")
    print(f"  same_complex_proxy=1: {ppi_df['same_complex_proxy'].sum()} pairs")

# ══════════════════════════════════════════════════════════════════════════════
# E3: Benchmark
# ══════════════════════════════════════════════════════════════════════════════
print("\n── E3: Benchmark ──")

def auc_binary(y_true, y_pred):
    """ROC-AUC for binary classification (positive vs negative)."""
    from sklearn.metrics import roc_auc_score
    labels = (y_true > 0).astype(int)
    if len(np.unique(labels)) < 2:
        return np.nan
    return roc_auc_score(labels, y_pred)

def bootstrap_ci(y_true, y_pred, n_boot=1000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    rs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(y_true[idx]) < 1e-10 or np.std(y_pred[idx]) < 1e-10:
            continue
        rs.append(stats.pearsonr(y_true[idx], y_pred[idx])[0])
    if not rs:
        return np.nan, np.nan
    lo = np.percentile(rs, (100 - ci) / 2)
    hi = np.percentile(rs, 100 - (100 - ci) / 2)
    return lo, hi

def loo_predict(X, y, model_cls=Ridge, **model_kwargs):
    """LOO cross-validated predictions."""
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    scaler = StandardScaler()
    for train_idx, test_idx in loo.split(X):
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        m = model_cls(alpha=1.0, **model_kwargs) if model_cls == Ridge else model_cls(**model_kwargs)
        m.fit(X_tr, y[train_idx])
        preds[test_idx] = m.predict(X_te)
    return preds

def evaluate(y_true, y_pred, label, dataset):
    r, rp   = stats.pearsonr(y_true, y_pred)
    rho, sp = stats.spearmanr(y_true, y_pred)
    rmse    = np.sqrt(mean_squared_error(y_true, y_pred))
    auc     = auc_binary(y_true, y_pred)
    ci_lo, ci_hi = bootstrap_ci(y_true, y_pred)
    return {
        'method': label, 'dataset': dataset,
        'pearson_r': r, 'pearson_p': rp,
        'spearman_rho': rho, 'spearman_p': sp,
        'rmse': rmse, 'auc': auc,
        'pearson_r_ci_lo': ci_lo, 'pearson_r_ci_hi': ci_hi,
        'n': len(y_true),
    }

def run_benchmark(df, target_col, dataset_name, ppi_feat_df=None):
    results = []
    preds_store = {}

    y = df[target_col].values.astype(float)

    # ── Baselines ──────────────────────────────────────────────────────────
    # Random
    rng = np.random.default_rng(0)
    y_rand = rng.standard_normal(len(y)) * np.std(y) + np.mean(y)
    results.append(evaluate(y, y_rand, 'random_baseline', dataset_name))
    preds_store['random_baseline'] = y_rand

    # Mean
    y_mean = np.full(len(y), np.mean(y))
    results.append(evaluate(y, y_mean, 'mean_baseline', dataset_name))
    preds_store['mean_baseline'] = y_mean

    # jaccard only
    X_jac = df[['jaccard_overlap']].values
    y_jac = loo_predict(X_jac, y)
    results.append(evaluate(y, y_jac, 'jaccard_only', dataset_name))
    preds_store['jaccard_only'] = y_jac

    # n_shared only
    X_ns = df[['n_shared_targets']].values
    y_ns = loo_predict(X_ns, y)
    results.append(evaluate(y, y_ns, 'n_shared_only', dataset_name))
    preds_store['n_shared_only'] = y_ns

    # ── ROES_v1 ────────────────────────────────────────────────────────────
    feat_v1 = ['jaccard_overlap', 'n_shared_targets', 'parallel_fraction',
               'antagonism_score']
    if 'serial_score' in df.columns:
        feat_v1.append('serial_score')
    X_v1 = df[feat_v1].values
    y_v1 = loo_predict(X_v1, y)
    results.append(evaluate(y, y_v1, 'roes_v1', dataset_name))
    preds_store['roes_v1'] = y_v1

    # ── ROES_v2 (add direction_bias) ───────────────────────────────────────
    df2 = df.copy()
    df2['direction_bias'] = df2['parallel_fraction'] - df2['antagonism_score']
    feat_v2 = feat_v1 + ['direction_bias']
    X_v2 = df2[feat_v2].values
    y_v2 = loo_predict(X_v2, y)
    results.append(evaluate(y, y_v2, 'roes_v2', dataset_name))
    preds_store['roes_v2'] = y_v2

    # ── PPI features ───────────────────────────────────────────────────────
    if ppi_feat_df is not None and STRING_AVAILABLE:
        merged = df.merge(ppi_feat_df[['TF_A','TF_B','ppi_direct',
                                        'ppi_shared_partners','same_complex_proxy']],
                          on=['TF_A','TF_B'], how='left').fillna(0)

        # ppi_only
        ppi_cols = ['ppi_direct', 'ppi_shared_partners', 'same_complex_proxy']
        X_ppi = merged[ppi_cols].values
        y_ppi = loo_predict(X_ppi, y)
        results.append(evaluate(y, y_ppi, 'ppi_only', dataset_name))
        preds_store['ppi_only'] = y_ppi

        # roes + ppi
        X_rp = merged[feat_v1 + ppi_cols].values
        y_rp = loo_predict(X_rp, y)
        results.append(evaluate(y, y_rp, 'roes_ppi', dataset_name))
        preds_store['roes_ppi'] = y_rp

    return pd.DataFrame(results), preds_store, y

# ── Run on DepMap ──────────────────────────────────────────────────────────────
res_depmap, preds_depmap, y_depmap = run_benchmark(
    depmap_df, 'depmap_r', 'DepMap', ppi_feat_df=ppi_df)
print("\n  DepMap results:")
print(res_depmap[['method','pearson_r','spearman_rho','rmse','auc']].to_string(index=False))

# ── Run on Norman ──────────────────────────────────────────────────────────────
res_norman, preds_norman, y_norman = run_benchmark(
    norman_df, 'eps_l2', 'Norman')
print("\n  Norman results:")
print(res_norman[['method','pearson_r','spearman_rho','rmse']].to_string(index=False))

# Save
all_results = pd.concat([res_depmap, res_norman], ignore_index=True)
all_results.to_csv(RES_DIR / "benchmark_results.csv", index=False)
print(f"\n  Saved benchmark_results.csv")

# ══════════════════════════════════════════════════════════════════════════════
# E4: Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
print("\n── E4: Feature Importance ──")

feat_v1 = ['jaccard_overlap', 'n_shared_targets', 'parallel_fraction',
           'antagonism_score']
if 'serial_score' in depmap_df.columns:
    feat_v1.append('serial_score')

X_full = depmap_df[feat_v1].values
y_full = depmap_df['depmap_r'].values

scaler = StandardScaler()
X_sc = scaler.fit_transform(X_full)
ridge = Ridge(alpha=1.0)
ridge.fit(X_sc, y_full)

coef_df = pd.DataFrame({'feature': feat_v1, 'coefficient': ridge.coef_})
coef_df['abs_coef'] = coef_df['coefficient'].abs()
coef_df = coef_df.sort_values('abs_coef', ascending=False)
print("\n  Ridge coefficients (standardized features):")
print(coef_df.to_string(index=False))

# Permutation importance
perm = permutation_importance(ridge, X_sc, y_full, n_repeats=200, random_state=42)
perm_df = pd.DataFrame({
    'feature': feat_v1,
    'perm_importance_mean': perm.importances_mean,
    'perm_importance_std':  perm.importances_std,
})
perm_df = perm_df.sort_values('perm_importance_mean', ascending=False)
print("\n  Permutation importances:")
print(perm_df.to_string(index=False))

feat_imp_df = coef_df.merge(perm_df, on='feature')
feat_imp_df.to_csv(RES_DIR / "feature_importance.csv", index=False)

# Try SHAP
try:
    import shap
    explainer = shap.LinearExplainer(ridge, X_sc)
    shap_values = explainer.shap_values(X_sc)
    shap_mean = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({'feature': feat_v1, 'shap_mean_abs': shap_mean})
    shap_df = shap_df.sort_values('shap_mean_abs', ascending=False)
    print("\n  SHAP mean |values|:")
    print(shap_df.to_string(index=False))
    feat_imp_df = feat_imp_df.merge(shap_df, on='feature', how='left')
    feat_imp_df.to_csv(RES_DIR / "feature_importance.csv", index=False)
    SHAP_OK = True
except Exception as e:
    print(f"  SHAP skipped: {e}")
    SHAP_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# E5: Visualization
# ══════════════════════════════════════════════════════════════════════════════
print("\n── E5: Visualization ──")

# ── Figure 1: Method Comparison Bar Chart ─────────────────────────────────────
methods_order = ['random_baseline','mean_baseline','jaccard_only','n_shared_only',
                 'roes_v1','roes_v2']
if STRING_AVAILABLE:
    methods_order += ['ppi_only','roes_ppi']

dep_plot = res_depmap[res_depmap['method'].isin(methods_order)].copy()
dep_plot['method'] = pd.Categorical(dep_plot['method'], categories=methods_order, ordered=True)
dep_plot = dep_plot.sort_values('method')

colors = ['#aaaaaa','#bbbbbb','#6baed6','#3182bd',
          '#fd8d3c','#e6550d','#74c476','#238b45']

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(dep_plot))
bars = ax.bar(x, dep_plot['pearson_r'], color=colors[:len(dep_plot)],
              edgecolor='white', linewidth=0.5, zorder=3)
# CI error bars
yerr_lo = dep_plot['pearson_r'].values - dep_plot['pearson_r_ci_lo'].values
yerr_hi = dep_plot['pearson_r_ci_hi'].values - dep_plot['pearson_r'].values
ax.errorbar(x, dep_plot['pearson_r'], yerr=[yerr_lo, yerr_hi],
            fmt='none', color='black', capsize=4, linewidth=1.2, zorder=4)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xticks(x)
ax.set_xticklabels(dep_plot['method'], rotation=35, ha='right', fontsize=9)
ax.set_ylabel("Pearson r (LOO-CV)", fontsize=11)
ax.set_title("Method Comparison – DepMap Co-essentiality (n=136)", fontsize=12, fontweight='bold')
ax.set_ylim(min(-0.1, dep_plot['pearson_r_ci_lo'].min()-0.05),
            max(0.5, dep_plot['pearson_r_ci_hi'].max()+0.05))
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar, r in zip(bars, dep_plot['pearson_r']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{r:.3f}', ha='center', va='bottom', fontsize=7.5)

plt.tight_layout()
plt.savefig(FIG_DIR / "method_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved method_comparison.png")

# ── Figure 2: Scatter Matrix ───────────────────────────────────────────────────
show_methods = ['jaccard_only', 'roes_v1', 'roes_v2']
if STRING_AVAILABLE and 'roes_ppi' in preds_depmap:
    show_methods.append('roes_ppi')
else:
    show_methods.append('n_shared_only')

fig, axes = plt.subplots(1, len(show_methods), figsize=(4*len(show_methods), 4), sharey=True)
for ax, meth in zip(axes, show_methods):
    yp = preds_depmap.get(meth, np.zeros_like(y_depmap))
    r  = stats.pearsonr(y_depmap, yp)[0]
    ax.scatter(y_depmap, yp, alpha=0.55, s=22, color='steelblue', edgecolors='none')
    m, b = np.polyfit(y_depmap, yp, 1)
    xl = np.linspace(y_depmap.min(), y_depmap.max(), 100)
    ax.plot(xl, m*xl+b, 'r-', linewidth=1.2)
    ax.set_xlabel("Actual depmap_r", fontsize=9)
    ax.set_title(f"{meth}\nr={r:.3f}", fontsize=9, fontweight='bold')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
axes[0].set_ylabel("Predicted", fontsize=9)
fig.suptitle("Predicted vs Actual depmap_r (LOO-CV, n=136)", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / "scatter_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved scatter_matrix.png")

# ── Figure 3: Feature Importance ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Ridge coefficients
ax = axes[0]
feat_imp_df_s = feat_imp_df.sort_values('coefficient')
colors_fi = ['#d62728' if c < 0 else '#1f77b4' for c in feat_imp_df_s['coefficient']]
ax.barh(feat_imp_df_s['feature'], feat_imp_df_s['coefficient'], color=colors_fi)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title("Ridge Coefficients\n(standardized)", fontsize=10, fontweight='bold')
ax.set_xlabel("Coefficient", fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Permutation importance
ax = axes[1]
perm_s = feat_imp_df.sort_values('perm_importance_mean', ascending=True)
ax.barh(perm_s['feature'], perm_s['perm_importance_mean'],
        xerr=perm_s['perm_importance_std'], color='#ff7f0e',
        error_kw={'capsize': 4})
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title("Permutation Importance\n(R² decrease)", fontsize=10, fontweight='bold')
ax.set_xlabel("Mean importance", fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(FIG_DIR / "feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved feature_importance.png")

# ── Figure 4: PPI + GRN Network Viz (CEBPA / SPI1) ───────────────────────────
print("  Building network viz for CEBPA / SPI1 ...")

tf_a_viz, tf_b_viz = 'CEBPA', 'SPI1'
tgt_a = grn_targets.get(tf_a_viz, set())
tgt_b = grn_targets.get(tf_b_viz, set())
shared = tgt_a & tgt_b

# Subsample for clarity
MAX_NODES = 30
tgt_a_sub   = set(list(tgt_a - tgt_b)[:MAX_NODES])
tgt_b_sub   = set(list(tgt_b - tgt_a)[:MAX_NODES])
shared_sub  = set(list(shared)[:MAX_NODES])

G_viz = nx.Graph()
for tf in [tf_a_viz, tf_b_viz]:
    G_viz.add_node(tf, ntype='TF')
for g in tgt_a_sub:
    G_viz.add_node(g, ntype='target_A')
    G_viz.add_edge(tf_a_viz, g, etype='GRN')
for g in tgt_b_sub:
    G_viz.add_node(g, ntype='target_B')
    G_viz.add_edge(tf_b_viz, g, etype='GRN')
for g in shared_sub:
    G_viz.add_node(g, ntype='shared')
    G_viz.add_edge(tf_a_viz, g, etype='GRN')
    G_viz.add_edge(tf_b_viz, g, etype='GRN')

# PPI edges
if ppi_gene_graph is not None:
    all_nodes_viz = set(G_viz.nodes())
    for n1 in all_nodes_viz:
        for n2 in all_nodes_viz:
            if n1 < n2 and ppi_gene_graph.has_edge(n1, n2):
                G_viz.add_edge(n1, n2, etype='PPI',
                               score=ppi_gene_graph[n1][n2]['score'])
    # Direct TF-TF PPI
    if ppi_gene_graph.has_edge(tf_a_viz, tf_b_viz):
        G_viz.add_edge(tf_a_viz, tf_b_viz, etype='PPI',
                       score=ppi_gene_graph[tf_a_viz][tf_b_viz]['score'])
        print(f"  Direct PPI between {tf_a_viz}-{tf_b_viz}: "
              f"score={ppi_gene_graph[tf_a_viz][tf_b_viz]['score']}")
    else:
        print(f"  No direct PPI between {tf_a_viz}-{tf_b_viz} at score>=700")

pos = nx.spring_layout(G_viz, seed=42, k=0.8)

fig, ax = plt.subplots(figsize=(12, 9))
node_colors = []
node_sizes  = []
for node in G_viz.nodes():
    nt = G_viz.nodes[node].get('ntype', 'target_A')
    if node in [tf_a_viz, tf_b_viz]:
        node_colors.append('#d62728'); node_sizes.append(600)
    elif nt == 'shared':
        node_colors.append('#9467bd'); node_sizes.append(120)
    elif nt == 'target_A':
        node_colors.append('#aec7e8'); node_sizes.append(60)
    else:
        node_colors.append('#ffbb78'); node_sizes.append(60)

grn_edges = [(u,v) for u,v,d in G_viz.edges(data=True) if d.get('etype')=='GRN']
ppi_edges = [(u,v) for u,v,d in G_viz.edges(data=True) if d.get('etype')=='PPI']

nx.draw_networkx_nodes(G_viz, pos, node_color=node_colors,
                       node_size=node_sizes, ax=ax, alpha=0.85)
nx.draw_networkx_edges(G_viz, pos, edgelist=grn_edges,
                       edge_color='#1f77b4', alpha=0.4, width=0.8, ax=ax)
nx.draw_networkx_edges(G_viz, pos, edgelist=ppi_edges,
                       edge_color='#ff7f0e', alpha=0.7, width=2.0, ax=ax)

# Label only TFs + shared
label_nodes = {tf_a_viz, tf_b_viz} | shared_sub
labels = {n: n for n in G_viz.nodes() if n in label_nodes}
nx.draw_networkx_labels(G_viz, pos, labels=labels, font_size=7, ax=ax)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor='#d62728', label='TF (CEBPA / SPI1)'),
    Patch(facecolor='#9467bd', label='Shared GRN targets'),
    Patch(facecolor='#aec7e8', label=f'{tf_a_viz}-only targets'),
    Patch(facecolor='#ffbb78', label=f'{tf_b_viz}-only targets'),
    Line2D([0],[0], color='#1f77b4', linewidth=1.5, label='GRN edge'),
    Line2D([0],[0], color='#ff7f0e', linewidth=2.5, label='PPI edge'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
          framealpha=0.9, edgecolor='gray')
ax.set_title(f"GRN + PPI Network: {tf_a_viz} / {tf_b_viz}\n"
             f"(GRN shared targets: {len(shared)}, subsample shown: {len(shared_sub)})",
             fontsize=11, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig(FIG_DIR / "ppi_grn_network_viz.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved ppi_grn_network_viz.png")

# ══════════════════════════════════════════════════════════════════════════════
# Key findings summary
# ══════════════════════════════════════════════════════════════════════════════
best_row = res_depmap.sort_values('pearson_r', ascending=False).iloc[0]
roes_v1  = res_depmap[res_depmap['method']=='roes_v1'].iloc[0]
roes_v2  = res_depmap[res_depmap['method']=='roes_v2'].iloc[0]

findings = f"""ROES Benchmark Key Findings
===========================
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Dataset: DepMap co-essentiality (n=136 pairs)

Best method: {best_row['method']}
  Pearson r  = {best_row['pearson_r']:.4f}  (95% CI: {best_row['pearson_r_ci_lo']:.4f} – {best_row['pearson_r_ci_hi']:.4f})
  Spearman ρ = {best_row['spearman_rho']:.4f}
  RMSE       = {best_row['rmse']:.4f}
  AUC        = {best_row['auc']:.4f}

ROES_v1:
  Pearson r  = {roes_v1['pearson_r']:.4f}  (95% CI: {roes_v1['pearson_r_ci_lo']:.4f} – {roes_v1['pearson_r_ci_hi']:.4f})
  Spearman ρ = {roes_v1['spearman_rho']:.4f}
  AUC        = {roes_v1['auc']:.4f}

ROES_v2 (+ direction_bias):
  Pearson r  = {roes_v2['pearson_r']:.4f}
  Spearman ρ = {roes_v2['spearman_rho']:.4f}

STRING PPI available: {STRING_AVAILABLE}
{"PPI-enhanced results included above." if STRING_AVAILABLE else "PPI tasks skipped (download failed)."}

Top feature (Ridge coef): {feat_imp_df.iloc[0]['feature']} (coef={feat_imp_df.iloc[0]['coefficient']:.4f})

All methods (DepMap):
{res_depmap[['method','pearson_r','spearman_rho','rmse','auc']].to_string(index=False)}

Norman 2019 (n=12):
{res_norman[['method','pearson_r','spearman_rho','rmse']].to_string(index=False)}
"""

with open(RES_DIR / "key_findings.txt", 'w') as f:
    f.write(findings)
print("\n" + findings)
print("  Saved key_findings.txt")

print("\n" + "=" * 70)
print("DONE. All outputs in:")
print(f"  Results : {RES_DIR}")
print(f"  Figures : {FIG_DIR}")
print("=" * 70)

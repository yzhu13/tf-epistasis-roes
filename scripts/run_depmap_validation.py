"""
DepMap 25Q2 co-essentiality validation of ROES predictions.

Hypothesis: TF pairs predicted as synthetic essential (high ROES) should show
HIGH CO-ESSENTIALITY in DepMap — i.e., correlated CERES/Chronos scores across
cancer cell lines. This is a fully independent validation from Norman 2019.

Logic: if TF_A and TF_B are functionally redundant, cancer cell lines that happen
to have TF_A disrupted become more dependent on TF_B (and vice versa), creating a
positive correlation in their dependency scores across the cell line panel.

Run: python scripts/run_depmap_validation.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
DATA_RAW  = BASE / "data" / "raw"
DATA_PROC = BASE / "data" / "processed"
FIGURES   = BASE / "figures"
FIGURES.mkdir(exist_ok=True)

# ── 1. Load DepMap CERES scores ──────────────────────────────────
print("Loading DepMap 25Q2 CRISPRGeneEffect.csv ...")
ceres_path = DATA_RAW / "DepMap" / "CRISPRGeneEffect.csv"
ceres = pd.read_csv(ceres_path, index_col=0)
print(f"  Shape: {ceres.shape}  (cell lines x genes)")

# Column format: "GENENAME (ENTREZID)"  ->  extract gene name
ceres.columns = [c.split(" (")[0] for c in ceres.columns]
print(f"  Unique genes: {ceres.shape[1]}")
print(f"  Unique cell lines: {ceres.shape[0]}")

# ── 2. Load ROES predictions ─────────────────────────────────────
roes_path = DATA_PROC / "roes_norman_tfs_ranked.csv"
roes = pd.read_csv(roes_path)
print(f"\nROES pairs loaded: {len(roes)}")

# ── 3. Compute co-essentiality for each ROES TF pair ─────────────
genes_depmap = set(ceres.columns)
print(f"\nComputing Pearson co-essentiality for each TF pair ...")

results = []
skipped = 0
for _, row in roes.iterrows():
    tf_a, tf_b = row["TF_A"], row["TF_B"]
    if tf_a not in genes_depmap or tf_b not in genes_depmap:
        skipped += 1
        continue

    a = ceres[tf_a].dropna()
    b = ceres[tf_b].dropna()
    common = a.index.intersection(b.index)
    if len(common) < 100:
        skipped += 1
        continue

    r, p = pearsonr(a[common].values, b[common].values)
    rho, p2 = spearmanr(a[common].values, b[common].values)

    results.append({
        "TF_A": tf_a, "TF_B": tf_b,
        "depmap_r": r, "depmap_p": p,
        "depmap_rho": rho,
        "n_cell_lines": len(common),
        # carry over ROES features
        "jaccard_overlap":    row["jaccard_overlap"],
        "n_shared_targets":   row["n_shared_targets"],
        "parallel_fraction":  row["parallel_fraction"],
        "antagonism_score":   row["antagonism_score"],
        "synth_score":        row["synth_score"],
    })

df = pd.DataFrame(results)
print(f"  Computed: {len(df)} pairs  |  skipped (not in DepMap): {skipped}")

# ── 4. Key statistics ────────────────────────────────────────────
print("\n=== ROES vs DepMap co-essentiality ===")
for feat in ["jaccard_overlap", "n_shared_targets", "parallel_fraction",
             "antagonism_score", "synth_score"]:
    v = df.dropna(subset=[feat, "depmap_r"])
    if len(v) > 3:
        r, p = pearsonr(v[feat], v["depmap_r"])
        rho, p2 = spearmanr(v[feat], v["depmap_r"])
        print(f"  {feat:28s}: Pearson r={r:+.3f} (p={p:.3f})  "
              f"Spearman rho={rho:+.3f} (p={p2:.3f})")

# High vs low ROES: do high-ROES pairs have higher co-essentiality?
median_score = df["synth_score"].median()
high = df[df["synth_score"] > median_score]["depmap_r"].dropna()
low  = df[df["synth_score"] <= median_score]["depmap_r"].dropna()
stat, p_mw = mannwhitneyu(high, low, alternative="greater")
print(f"\n  Mann-Whitney (high ROES > low ROES co-essentiality): p={p_mw:.4f}")
print(f"  Mean depmap_r  high-ROES={high.mean():.4f}  low-ROES={low.mean():.4f}")

# ── 5. Result table ──────────────────────────────────────────────
df_sorted = df.sort_values("depmap_r", ascending=False)
print("\nAll TF pairs ranked by DepMap co-essentiality:")
print(df_sorted[["TF_A","TF_B","depmap_r","depmap_p",
                  "n_shared_targets","synth_score"]].to_string(index=False))

# Save
out_path = DATA_PROC / "depmap_coessentiality_results.csv"
df_sorted.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# ── 6. Figures ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("DepMap 25Q2 Independent Validation of ROES Predictions", fontsize=13, fontweight="bold")

# Panel 1: n_shared_targets vs depmap_r
ax = axes[0]
ax.scatter(df["n_shared_targets"], df["depmap_r"], s=80, alpha=0.8, color="steelblue")
for _, row in df.iterrows():
    ax.annotate(f"{row['TF_A']}/{row['TF_B']}",
                (row["n_shared_targets"], row["depmap_r"]),
                fontsize=6.5, ha="left", va="bottom")
v = df.dropna(subset=["n_shared_targets","depmap_r"])
if len(v) > 2:
    r, p = pearsonr(v["n_shared_targets"], v["depmap_r"])
    m, b = np.polyfit(v["n_shared_targets"], v["depmap_r"], 1)
    xr = np.linspace(v["n_shared_targets"].min(), v["n_shared_targets"].max(), 100)
    ax.plot(xr, m*xr+b, "r--", alpha=0.6)
    ax.text(0.05, 0.95, f"Pearson r={r:.3f}\np={p:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("n_shared_targets (ROES)", fontsize=11)
ax.set_ylabel("DepMap co-essentiality (Pearson r)", fontsize=11)
ax.set_title("Shared Targets vs Co-essentiality", fontsize=10)

# Panel 2: synth_score vs depmap_r
ax = axes[1]
sc = ax.scatter(df["synth_score"], df["depmap_r"],
                c=df["parallel_fraction"], cmap="RdYlBu", s=80, alpha=0.9)
plt.colorbar(sc, ax=ax, label="parallel_fraction")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("Synthetic essential score (n_shared x parallel_frac)", fontsize=11)
ax.set_ylabel("DepMap co-essentiality (Pearson r)", fontsize=11)
ax.set_title("ROES Composite Score vs Co-essentiality", fontsize=10)

# Panel 3: High vs low ROES distribution of depmap_r
ax = axes[2]
ax.hist(low.values,  bins=12, alpha=0.6, color="steelblue",
        label=f"Low ROES (n={len(low)})", density=True)
ax.hist(high.values, bins=12, alpha=0.6, color="darkorange",
        label=f"High ROES (n={len(high)})", density=True)
ax.axvline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("DepMap co-essentiality (Pearson r)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title(f"High-ROES pairs more co-essential?\nMann-Whitney p={p_mw:.4f}", fontsize=10)
ax.legend(fontsize=9)

plt.tight_layout()
fig_path = FIGURES / "depmap_validation.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved: {fig_path}")

# ── 7. Combined ROES + DepMap top hits ───────────────────────────
print("\n=== TOP SYNTHETIC ESSENTIAL TF PAIRS (ROES + DepMap concordant) ===")
df["composite_rank"] = (df["synth_score"].rank(pct=True) +
                        df["depmap_r"].rank(pct=True)) / 2
top = df.nlargest(10, "composite_rank")
print(top[["TF_A","TF_B","n_shared_targets","synth_score",
           "depmap_r","depmap_p","n_cell_lines"]].to_string(index=False))
top.to_csv(DATA_PROC / "top_synthetic_essential_concordant.csv", index=False)

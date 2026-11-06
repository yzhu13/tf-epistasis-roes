import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

plt.rcParams.update({"font.size": 11})

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "results", "L2_antagonism_depmap")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load & filter ─────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, "results", "J_depmap_revalidation_132", "depmap_coessentiality_132pairs.csv"))
df = df.dropna(subset=["depmap_r"])
df = df[df["n_cell_lines"] >= 100].reset_index(drop=True)
print(f"Working n = {len(df)}")

# Save backup
df.to_csv(os.path.join(OUT_DIR, "depmap_filtered_121pairs.csv"), index=False)

# ── 1. Correlations ───────────────────────────────────────────────────────────
def partial_corr_residualize(x, y, z):
    """Partial correlation of x~y controlling for z via residualization."""
    def residuals(a, b):
        slope, intercept, *_ = stats.linregress(b, a)
        return a - (slope * b + intercept)
    rx = residuals(x.values, z.values)
    ry = residuals(y.values, z.values)
    r, p = pearsonr(rx, ry)
    return r, p

xv = df["antagonism_score"]
yv = df["depmap_r"]
zv = df["n_shared_targets"].astype(float)

r_p, p_p = pearsonr(xv, yv)
r_s, p_s = spearmanr(xv, yv)
r_partial, p_partial = partial_corr_residualize(xv, yv, zv)

print(f"antagonism_score -> depmap_r: Pearson r={r_p:.3f} p={p_p:.4f}")
print(f"antagonism_score -> depmap_r: Spearman rho={r_s:.3f} p={p_s:.4f}")
print(f"antagonism_score -> depmap_r (partial, ctrl n_shared): r={r_partial:.3f} p={p_partial:.4f}")

# parallel_fraction
xv2 = df["parallel_fraction"]
r_p2, p_p2 = pearsonr(xv2, yv)
r_s2, p_s2 = spearmanr(xv2, yv)
r_partial2, p_partial2 = partial_corr_residualize(xv2, yv, zv)
print(f"parallel_fraction -> depmap_r: Pearson r={r_p2:.3f} p={p_p2:.4f}")
print(f"parallel_fraction -> depmap_r (partial): r={r_partial2:.3f} p={p_partial2:.4f}")

corr_rows = [
    {"predictor": "antagonism_score", "outcome": "depmap_r", "type": "pearson",
     "r": r_p, "p": p_p, "n": len(df), "note": ""},
    {"predictor": "antagonism_score", "outcome": "depmap_r", "type": "spearman",
     "r": r_s, "p": p_s, "n": len(df), "note": ""},
    {"predictor": "antagonism_score", "outcome": "depmap_r", "type": "partial_pearson",
     "r": r_partial, "p": p_partial, "n": len(df), "note": "controlling n_shared_targets"},
    {"predictor": "parallel_fraction", "outcome": "depmap_r", "type": "pearson",
     "r": r_p2, "p": p_p2, "n": len(df), "note": ""},
    {"predictor": "parallel_fraction", "outcome": "depmap_r", "type": "partial_pearson",
     "r": r_partial2, "p": p_partial2, "n": len(df), "note": "controlling n_shared_targets"},
]
pd.DataFrame(corr_rows).to_csv(os.path.join(OUT_DIR, "correlation_stats.csv"), index=False)

# ── 2. Key pairs ──────────────────────────────────────────────────────────────
KEY_PAIRS = [("CDKN1A", "MAPK1"), ("PLK4", "STIL"), ("KIF18B", "KIF2C"), ("CEBPE", "SPI1")]

def find_pair(df_, a, b):
    row = df_[((df_["TF_A"] == a) & (df_["TF_B"] == b)) |
              ((df_["TF_A"] == b) & (df_["TF_B"] == a))]
    return row

# ── 3. Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Regulatory direction concordance predicts functional co-dependence (n=121 TF pairs)",
             fontsize=12, fontweight="bold")

norm = mcolors.Normalize(vmin=df["n_shared_targets"].min(), vmax=df["n_shared_targets"].max())
cmap = cm.viridis

def make_panel(ax, x_col, y_col, r_pearson, p_pearson, r_part, p_part, xlabel):
    sc = ax.scatter(df[x_col], df[y_col], c=df["n_shared_targets"],
                    cmap=cmap, norm=norm, alpha=0.7, s=45, edgecolors="none", zorder=3)
    # regression
    xv_, yv_ = df[x_col].values, df[y_col].values
    slope, intercept, *_ = stats.linregress(xv_, yv_)
    x_line = np.linspace(xv_.min(), xv_.max(), 200)
    ax.plot(x_line, slope * x_line + intercept, "k--", lw=1.5, zorder=4)
    # horizontal line at 0
    ax.axhline(0, color="gray", lw=1, ls=":", zorder=2)
    # stats box
    stats_text = (f"r = {r_pearson:.2f}, p = {p_pearson:.3f}\n"
                  f"partial-r = {r_part:.2f}, p = {p_part:.3f}\n"
                  f"(ctrl n_shared)")
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))
    # annotate key pairs
    for a, b in KEY_PAIRS:
        row = find_pair(df, a, b)
        if len(row) == 0:
            continue
        row = row.iloc[0]
        label = f"{a}/{b}"
        ax.annotate(label, xy=(row[x_col], row["depmap_r"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8,
                    arrowprops=dict(arrowstyle="-", lw=0.8, color="black"))
    ax.set_xlabel(xlabel); ax.set_ylabel("DepMap co-essentiality (r)")
    return sc

sc = make_panel(axes[0], "antagonism_score", "depmap_r",
                r_p, p_p, r_partial, p_partial, "Antagonism score")
axes[0].set_title("Antagonism score vs DepMap co-essentiality")

make_panel(axes[1], "parallel_fraction", "depmap_r",
           r_p2, p_p2, r_partial2, p_partial2, "Parallel fraction")
axes[1].set_title("Parallel fraction vs DepMap co-essentiality")

# Colorbar
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.8, pad=0.02)
cbar.set_label("n shared targets", fontsize=10)

out_fig = os.path.join(FIG_DIR, "figure_L2_antagonism_vs_depmap.png")
plt.savefig(out_fig, dpi=200, bbox_inches="tight")
plt.close()
print(f"Figure saved: {out_fig}")

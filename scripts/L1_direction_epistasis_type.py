import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr, spearmanr, f_oneway, kruskal

plt.rcParams.update({"font.size": 11})

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "results", "L1_direction_analysis")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load inputs ──────────────────────────────────────────────────────────────
roes = pd.read_csv(os.path.join(BASE, "results", "G_grn_roes_validation", "roes_GSE133344_132pairs.csv"))
eps  = pd.read_csv(os.path.join(BASE, "results", "H_recalibrated_eps_types", "epistasis_groundtruth_recalibrated.csv"))

# Merge: try (A,B) then (B,A)
merged_ab = roes.merge(eps, on=["TF_A", "TF_B"], how="inner")
swapped   = roes.copy().rename(columns={"TF_A": "TF_B", "TF_B": "TF_A"})
merged_ba = swapped.merge(eps, on=["TF_A", "TF_B"], how="inner")
# Rename back so TF_A/B reflect eps orientation
merged_ba = merged_ba.rename(columns={"TF_A": "TF_B_orig", "TF_B": "TF_A_orig"})
merged_ba.columns = [c.replace("_orig", "") for c in merged_ba.columns]

# Combine, drop duplicates by (TF_A, TF_B) from eps
merged = pd.concat([merged_ab, merged_ba], ignore_index=True).drop_duplicates(subset=["TF_A", "TF_B"])
print(f"Merged rows: {len(merged)}")

# Save backup
merged.to_csv(os.path.join(OUT_DIR, "merged_roes_eps_132pairs.csv"), index=False)

# ── 1. Correlation analysis ───────────────────────────────────────────────────
pairs = [
    ("antagonism_score", "cos_obs_expected"),
    ("parallel_fraction", "cos_obs_expected"),
    ("antagonism_score", "magnitude_ratio"),
    ("parallel_fraction", "magnitude_ratio"),
]

corr_rows = []
for x_col, y_col in pairs:
    sub = merged[[x_col, y_col]].dropna()
    r_p, p_p = pearsonr(sub[x_col], sub[y_col])
    r_s, p_s = spearmanr(sub[x_col], sub[y_col])
    print(f"  {x_col} -> {y_col}: Pearson r={r_p:.3f} p={p_p:.4f} | Spearman rho={r_s:.3f} p={p_s:.4f}")
    corr_rows.append({"x": x_col, "y": y_col, "pearson_r": r_p, "pearson_p": p_p,
                       "spearman_rho": r_s, "spearman_p": p_s, "n": len(sub)})

pd.DataFrame(corr_rows).to_csv(os.path.join(OUT_DIR, "correlation_stats.csv"), index=False)

# ── 2. ANOVA / Kruskal-Wallis ─────────────────────────────────────────────────
groups = merged["new_eps_type"].unique()
anova_lines = []
for feat in ["parallel_fraction", "antagonism_score"]:
    group_data = [merged.loc[merged["new_eps_type"] == g, feat].dropna().values for g in groups]
    f_stat, p_anova = f_oneway(*group_data)
    h_stat, p_kw    = kruskal(*group_data)
    group_means = ", ".join(f"{g}={merged.loc[merged['new_eps_type']==g, feat].mean():.3f}" for g in groups)
    line = (f"{feat}:\n"
            f"  ANOVA F={f_stat:.3f} p={p_anova:.4f}\n"
            f"  Kruskal-Wallis H={h_stat:.3f} p={p_kw:.4f}\n"
            f"  Group means: {group_means}\n")
    print(line)
    anova_lines.append(line)

with open(os.path.join(OUT_DIR, "anova_results.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(anova_lines))

# ── 3. Figure ─────────────────────────────────────────────────────────────────
TYPE_COLORS = {
    "buffering":   "#1f77b4",
    "synergistic": "#d62728",
    "additive":    "#2ca02c",
    "complex":     "#ff7f0e",
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Regulatory direction predicts epistasis type (n=132 TF pairs)", fontsize=13, fontweight="bold")

def scatter_with_reg(ax, x_col, y_col, title, xlabel, ylabel):
    sub = merged[[x_col, y_col, "new_eps_type"]].dropna()
    colors = sub["new_eps_type"].map(TYPE_COLORS).fillna("gray")
    ax.scatter(sub[x_col], sub[y_col], c=colors, alpha=0.65, s=40, edgecolors="none")
    # regression
    xv, yv = sub[x_col].values, sub[y_col].values
    slope, intercept, *_ = stats.linregress(xv, yv)
    x_line = np.linspace(xv.min(), xv.max(), 200)
    ax.plot(x_line, slope * x_line + intercept, "k--", lw=1.5)
    r_p, p_p = pearsonr(xv, yv)
    ax.text(0.05, 0.95, f"r={r_p:.2f}, p={p_p:.3f}", transform=ax.transAxes,
            va="top", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)

# Top-left: antagonism_score vs magnitude_ratio
scatter_with_reg(axes[0, 0], "antagonism_score", "magnitude_ratio",
                 "Antagonism vs Magnitude ratio",
                 "Antagonism score", "Magnitude ratio")

# Top-right: parallel_fraction vs magnitude_ratio
scatter_with_reg(axes[0, 1], "parallel_fraction", "magnitude_ratio",
                 "Parallel fraction vs Magnitude ratio",
                 "Parallel fraction", "Magnitude ratio")

# Bottom-left: antagonism_score vs cos_obs_expected
scatter_with_reg(axes[1, 0], "antagonism_score", "cos_obs_expected",
                 "Antagonism vs Cosine similarity",
                 "Antagonism score", "cos(obs, expected)")

# Bottom-right: violin/box of parallel_fraction by eps_type
ax = axes[1, 1]
ordered_types = ["synergistic", "additive", "buffering", "complex"]
plot_data = [merged.loc[merged["new_eps_type"] == t, "parallel_fraction"].dropna().values for t in ordered_types]
vp = ax.violinplot(plot_data, positions=range(len(ordered_types)), showmedians=True, showextrema=True)
for i, (body, t) in enumerate(zip(vp["bodies"], ordered_types)):
    body.set_facecolor(TYPE_COLORS[t]); body.set_alpha(0.7)
vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2)
ax.set_xticks(range(len(ordered_types))); ax.set_xticklabels(ordered_types, fontsize=10)
ax.set_xlabel("Epistasis type"); ax.set_ylabel("Parallel fraction")
ax.set_title("Parallel fraction by epistasis type")

# Legend (shared)
patches = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()]
fig.legend(handles=patches, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=10)

plt.tight_layout(rect=[0, 0.04, 1, 1])
out_fig = os.path.join(FIG_DIR, "figure_L1_direction_vs_epistasis_type.png")
plt.savefig(out_fig, dpi=200, bbox_inches="tight")
plt.close()
print(f"Figure saved: {out_fig}")

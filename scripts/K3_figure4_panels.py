"""
Generate Figure 4 (A/B/C panels + combined): CEBP family x SPI1 case study.
  A - CEBPA/SPI1 GRN network, regenerated with structured (non-spring) layout,
      larger fonts, fewer/clearer nodes for publication
  B - within-family (CEBPB/CEBPA, CEBPE/CEBPA: buffering) vs cross-family
      (CEBPE/SPI1: synergistic) epistasis-type comparison
  C - genome-wide ROES predictions (reuse C_gwps/grn_expansion_comparison.png)

Run: python scripts/K3_figure4_panels.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
FIG = BASE / "figures"
plt.rcParams.update({"font.size": 11})

# ─────────────────────────────────────────────────────────────────
# Panel A: CEBPA/SPI1 network, structured layout
# ─────────────────────────────────────────────────────────────────
print("Loading GRN ...")
grn = pd.read_csv(BASE / "data" / "processed" / "grn_tf_target_network.csv")

# target_gene is a row-index into the GSE133344 genes.tsv (0-indexed) -> map to symbols
genes = pd.read_csv(BASE / "data" / "raw" / "Norman2019_GSE133344" / "genes.tsv", sep="\t", header=None)
idx2symbol = genes[1].to_dict()

def to_symbol(g):
    try:
        return idx2symbol[int(g)]
    except (ValueError, KeyError):
        return str(g)

grn_targets = {}
for tf, tg in zip(grn["TF"], grn["target_gene"]):
    grn_targets.setdefault(tf, set()).add(to_symbol(tg))

tf_a, tf_b = "CEBPA", "SPI1"
tgt_a, tgt_b = grn_targets[tf_a], grn_targets[tf_b]
shared = tgt_a & tgt_b
a_only = tgt_a - tgt_b
b_only = tgt_b - tgt_a
print(f"  {tf_a}: {len(tgt_a)}  {tf_b}: {len(tgt_b)}  shared: {len(shared)}")

N_SHARED_SHOW = 10
N_SIDE_SHOW = 7
shared_sub = sorted(shared)[:N_SHARED_SHOW]
a_only_sub = sorted(a_only)[:N_SIDE_SHOW]
b_only_sub = sorted(b_only)[:N_SIDE_SHOW]


def make_panel_A(ax):
    # x positions: A-only (left col), TF-A (x=2), shared (middle col, x=4),
    # TF-B (x=6), B-only (right col)
    def col_y(n, spread):
        return (np.linspace(1, n, n) - (n + 1) / 2) * spread

    # Plot edges first
    ys_a = col_y(len(a_only_sub), 1.3)
    for i, g in enumerate(a_only_sub):
        y = ys_a[i]
        ax.plot([0, 2], [y, 0], color="#1f77b4", alpha=0.35, lw=1, zorder=1)
        ax.scatter(0, y, s=70, color="#aec7e8", zorder=2, edgecolor="white")
        ax.text(-0.2, y, g, fontsize=9, ha="right", va="center")

    ys_b = col_y(len(b_only_sub), 1.3)
    for i, g in enumerate(b_only_sub):
        y = ys_b[i]
        ax.plot([6, 8], [0, y], color="#ff7f0e", alpha=0.35, lw=1, zorder=1)
        ax.scatter(8, y, s=70, color="#ffbb78", zorder=2, edgecolor="white")
        ax.text(8.2, y, g, fontsize=9, ha="left", va="center")

    ys_s = col_y(len(shared_sub), 1.05)
    for i, g in enumerate(shared_sub):
        y = ys_s[i]
        ax.plot([2, 4], [0, y], color="#1f77b4", alpha=0.5, lw=1.2, zorder=1)
        ax.plot([4, 6], [y, 0], color="#ff7f0e", alpha=0.5, lw=1.2, zorder=1)
        ax.scatter(4, y, s=80, color="#9467bd", zorder=2, edgecolor="white")
        side = 1 if i % 2 == 0 else -1
        ax.text(4 + side * 0.35, y, g, fontsize=9, ha="left" if side > 0 else "right", va="center", fontweight="bold")

    # TF nodes
    ax.scatter([2, 6], [0, 0], s=1400, color="#d62728", zorder=3, edgecolor="black", linewidth=1.2)
    ax.text(2, 0, tf_a, fontsize=14, ha="center", va="center", fontweight="bold", color="white", zorder=4)
    ax.text(6, 0, tf_b, fontsize=14, ha="center", va="center", fontweight="bold", color="white", zorder=4)

    # Direct PPI edge (STRING combined_score=990, established in Phase 2)
    ax.annotate("", xy=(5.45, -0.85), xytext=(2.55, -0.85),
                 arrowprops=dict(arrowstyle="-", color="#2ca02c", lw=3))
    ax.text(4, -1.15, "Direct PPI (STRING combined_score=990)", fontsize=10,
            ha="center", color="#2ca02c", fontweight="bold")

    legend_elements = [
        Patch(facecolor="#d62728", label=f"TF ({tf_a} / {tf_b})"),
        Patch(facecolor="#9467bd", label=f"Shared GRN targets (n={len(shared)} total)"),
        Patch(facecolor="#aec7e8", label=f"{tf_a}-only targets (n={len(a_only)} total)"),
        Patch(facecolor="#ffbb78", label=f"{tf_b}-only targets (n={len(b_only)} total)"),
        Line2D([0], [0], color="#2ca02c", lw=3, label="Direct PPI edge"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.32),
              ncol=3, fontsize=10, frameon=False)

    ax.set_xlim(-2.0, 10.0)
    ax.set_ylim(-8, 8)
    ax.axis("off")
    ax.set_title(f"A. {tf_a}/{tf_b} regulatory network ({len(shared)} shared GRN targets, top {N_SHARED_SHOW} shown)",
                  fontsize=12, fontweight="bold", pad=15)


print("Generating Figure 4A ...")
fig, ax = plt.subplots(figsize=(12, 9.5))
make_panel_A(ax)
plt.tight_layout()
fig.savefig(FIG / "figure4A_cebpa_spi1_network.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────
# Panel B: within-family vs cross-family CEBP epistasis comparison
# ─────────────────────────────────────────────────────────────────
print("Generating Figure 4B ...")
recal = pd.read_csv(BASE / "results" / "H_recalibrated_eps_types" / "epistasis_groundtruth_recalibrated.csv")
roes = pd.read_csv(BASE / "results" / "G_grn_roes_validation" / "roes_GSE133344_132pairs.csv")
roes["total_targets"] = roes["n_targets_A"] + roes["n_targets_B"]

pairs = [("CEBPB", "CEBPA", "within-family\n(CEBP-CEBP)"),
         ("CEBPE", "CEBPA", "within-family\n(CEBP-CEBP)"),
         ("CEBPE", "SPI1", "cross-family\n(CEBP-SPI1)")]

rows = []
for a, b, grp in pairs:
    r = recal[((recal.TF_A == a) & (recal.TF_B == b)) | ((recal.TF_A == b) & (recal.TF_B == a))].iloc[0]
    rr = roes[((roes.TF_A == a) & (roes.TF_B == b)) | ((roes.TF_A == b) & (roes.TF_B == a))].iloc[0]
    rows.append({"pair": f"{a}/{b}", "group": grp, "eps_l2": r.eps_l2,
                  "magnitude_ratio": r.magnitude_ratio, "cos_obs_expected": r.cos_obs_expected,
                  "new_eps_type": r.new_eps_type, "n_shared": rr.n_shared_targets,
                  "jaccard": rr.jaccard_overlap})

dfB = pd.DataFrame(rows)
print(dfB)

type_colors = {"buffering": "#1f77b4", "synergistic": "#d62728", "additive": "#2ca02c", "complex": "#ff7f0e"}

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
colors = [type_colors[t] for t in dfB["new_eps_type"]]
bars = ax.bar(dfB["pair"], dfB["magnitude_ratio"], color=colors, alpha=0.85, edgecolor="black")
ax.axhline(1.0, color="gray", ls="--", lw=1)
for bar, t, eps in zip(bars, dfB["new_eps_type"], dfB["eps_l2"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{t}\neps_l2={eps:.2f}",
            ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("magnitude_ratio = ||δAB|| / ||δA+δB||", fontsize=10)
ax.set_title("Epistasis type: within- vs cross-family", fontsize=11, fontweight="bold")
ax.set_ylim(0, max(dfB["magnitude_ratio"]) * 1.35)

ax = axes[1]
ax.scatter(dfB["n_shared"], dfB["jaccard"], s=120, c=colors, edgecolor="black", zorder=3)
for _, row in dfB.iterrows():
    ax.annotate(row["pair"], (row["n_shared"], row["jaccard"]), fontsize=9, fontweight="bold",
                 xytext=(6, 6), textcoords="offset points")
ax.set_xlabel("n_shared_targets", fontsize=10)
ax.set_ylabel("jaccard_overlap", fontsize=10)
ax.set_title("ROES topology features", fontsize=11, fontweight="bold")
ax.set_xlim(0, max(dfB["n_shared"]) * 1.3)

handles = [Patch(facecolor=c, edgecolor="black", label=t) for t, c in type_colors.items() if t in dfB["new_eps_type"].values]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("B. CEBP family x SPI1: within-family pairs buffer, cross-family pair is synergistic",
              fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(FIG / "figure4B_within_vs_cross_family.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────
# Combined Figure 4
# ─────────────────────────────────────────────────────────────────
print("Generating combined Figure 4 ...")
imgA = mpimg.imread(FIG / "figure4A_cebpa_spi1_network.png")
imgB = mpimg.imread(FIG / "figure4B_within_vs_cross_family.png")
imgC = mpimg.imread(FIG / "C_gwps" / "grn_expansion_comparison.png")

fig = plt.figure(figsize=(14, 14))
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
axA = fig.add_subplot(gs[0, :])
axB = fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[1, 1])

for ax, img, label in [(axA, imgA, ""),
                        (axB, imgB, ""),
                        (axC, imgC, "C. Genome-wide expanded GRN: ROES consistency check")]:
    ax.imshow(img)
    ax.axis("off")
    if label:
        ax.set_title(label, fontsize=12, fontweight="bold", loc="left")

fig.suptitle("Figure 4. Translational case study: CEBP family x SPI1 epistasis (AML)",
              fontsize=14, fontweight="bold", y=0.995)
plt.tight_layout()
fig.savefig(FIG / "Figure4_combined.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("Done. Saved figure4A_*.png, figure4B_*.png, Figure4_combined.png")

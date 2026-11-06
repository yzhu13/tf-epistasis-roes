"""
Generate Figure 3 (A/B/C panels + combined):
  A - DepMap co-essentiality vs ROES (n=121, GSE133344-derived) [reuse J figure]
  B - CDKN1A/MAPK1 antagonism (shared-target direction discordance) [reuse B3_fig4]
  C - PLK4/STIL & KIF18B/KIF2C deep-dive: DepMap co-essentiality scatter (positive controls)

Run: python scripts/K2_figure3_panels.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import pearsonr
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
FIG = BASE / "figures"
DATA_RAW = BASE / "data" / "raw"

plt.rcParams.update({"font.size": 11})

# ─────────────────────────────────────────────────────────────────
# Panel C: PLK4/STIL & KIF18B/KIF2C DepMap deep-dive
# ─────────────────────────────────────────────────────────────────
print("Loading DepMap CRISPRGeneEffect.csv ...")
ceres = pd.read_csv(DATA_RAW / "DepMap" / "CRISPRGeneEffect.csv", index_col=0)
ceres.columns = [c.split(" (")[0] for c in ceres.columns]

top = pd.read_csv(BASE / "results" / "J_depmap_revalidation_132" / "top_concordant_pairs.csv")

pairs_info = []
for a, b in [("PLK4", "STIL"), ("KIF18B", "KIF2C")]:
    row = top[((top.TF_A == a) & (top.TF_B == b)) | ((top.TF_A == b) & (top.TF_B == a))].iloc[0]
    pairs_info.append((a, b, row))


def make_panel_C(axes):
    for ax, (a, b, row) in zip(axes, pairs_info):
        ca = ceres[a].dropna()
        cb = ceres[b].dropna()
        common = ca.index.intersection(cb.index)
        x, y = ca[common].values, cb[common].values
        r, p = pearsonr(x, y)
        ax.scatter(x, y, s=8, alpha=0.35, color="darkorange")
        m, c = np.polyfit(x, y, 1)
        xr = np.linspace(x.min(), x.max(), 100)
        ax.plot(xr, m * xr + c, "k--", lw=1.5, alpha=0.7)
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.axvline(0, color="gray", lw=0.5, ls=":")
        ax.set_xlabel(f"{a} CERES gene effect", fontsize=10)
        ax.set_ylabel(f"{b} CERES gene effect", fontsize=10)
        ax.set_title(f"{a} / {b}", fontsize=11, fontweight="bold")
        ax.text(0.05, 0.05,
                f"DepMap r={row['depmap_r']:.3f}\np={row['depmap_p']:.2e}\n"
                f"n_shared_targets={int(row['n_shared_targets'])}\n"
                f"eps_l2={row['eps_l2']:.2f} ({row['eps_type']})",
                transform=ax.transAxes, va="bottom", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))


print("Generating Figure 3C (PLK4/STIL & KIF18B/KIF2C) ...")
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
make_panel_C(axes)
fig.suptitle("C. Functional-module positive controls: DepMap co-essentiality\n"
              "(mitosis/centriole genes — strongest concordant pairs in 132-pair set)",
              fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG / "figure3C_plk4stil_kif18bkif2c.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────
# Combined Figure 3: A (existing) + B (existing) + C (new)
# ─────────────────────────────────────────────────────────────────
print("Generating combined Figure 3 ...")
imgA = mpimg.imread(FIG / "figure3a_depmap_validation_n121.png")
imgB = mpimg.imread(FIG / "figure3b_cdkn1a_mapk1_antagonism.png")
imgC = mpimg.imread(FIG / "figure3C_plk4stil_kif18bkif2c.png")

fig = plt.figure(figsize=(14, 13))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
axA = fig.add_subplot(gs[0, :])
axB = fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[1, 1])

for ax, img, label in [(axA, imgA, "A. DepMap co-essentiality vs ROES (n=121)"),
                        (axB, imgB, "B. CDKN1A/MAPK1: discordant shared-target regulation"),
                        (axC, imgC, "C. PLK4/STIL & KIF18B/KIF2C (positive controls)")]:
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(label, fontsize=12, fontweight="bold", loc="left")

fig.suptitle("Figure 3. Cross-modality validation: GRN-derived ROES vs cancer-cell-line co-essentiality",
              fontsize=14, fontweight="bold", y=0.995)
plt.tight_layout()
fig.savefig(FIG / "Figure3_combined.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("Done. Saved figure3C_*.png and Figure3_combined.png")

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({"font.size": 11})

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPMAP_FILE   = os.path.join(BASE, "data", "processed", "depmap_coessentiality_results.csv")
PPI_FILE      = os.path.join(BASE, "results", "E_benchmark", "ppi_features_all_pairs.csv")
DISEASE_FILE  = os.path.join(BASE, "results", "D_disease", "tf_disease_associations.csv")
OUT_DIR       = os.path.join(BASE, "results", "L3_evidence_convergence")
FIG_PATH      = os.path.join(BASE, "figures", "figure_L3_cebpa_spi1_evidence_convergence.png")
CSV_PATH      = os.path.join(OUT_DIR, "evidence_matrix_all_pairs.csv")
TXT_PATH      = os.path.join(OUT_DIR, "cebpa_spi1_summary.txt")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)

# ── Load Evidence 1 & 2 from DepMap file ──────────────────────────────────────
depmap = pd.read_csv(DEPMAP_FILE)
depmap = depmap[["TF_A", "TF_B", "n_shared_targets", "depmap_r"]].copy()

# ── Load Evidence 3: PPI score ─────────────────────────────────────────────────
ppi = pd.read_csv(PPI_FILE)
ppi = ppi[["TF_A", "TF_B", "ppi_score"]].copy()

# Merge PPI — try both orientations
def merge_ppi(df, ppi):
    merged = df.merge(ppi, on=["TF_A", "TF_B"], how="left")
    # For rows that didn't match, try reversed orientation
    missing = merged["ppi_score"].isna()
    if missing.any():
        ppi_rev = ppi.rename(columns={"TF_A": "TF_B", "TF_B": "TF_A"})
        filled = df[missing].merge(ppi_rev, on=["TF_A", "TF_B"], how="left")
        merged.loc[missing, "ppi_score"] = filled["ppi_score"].values
    merged["ppi_score"] = merged["ppi_score"].fillna(0.0)
    return merged

df = merge_ppi(depmap, ppi)

# ── Load Evidence 4: Shared disease associations ───────────────────────────────
disease = pd.read_csv(DISEASE_FILE)
# Keep diseases with ot_score >= 0.5
disease_strong = disease[disease["ot_score"] >= 0.5].copy()

# For each TF, collect the set of disease_ids
tf_diseases = disease_strong.groupby("TF")["disease_id"].apply(set).to_dict()

def shared_disease_score(row):
    set_a = tf_diseases.get(row["TF_A"], set())
    set_b = tf_diseases.get(row["TF_B"], set())
    return len(set_a & set_b)

df["shared_disease_score"] = df.apply(shared_disease_score, axis=1)

# ── Save evidence matrix ───────────────────────────────────────────────────────
df.to_csv(CSV_PATH, index=False)
print(f"Saved evidence matrix: {CSV_PATH}")
print(f"Total pairs: {len(df)}")

# ── Identify CEBPA/SPI1 ────────────────────────────────────────────────────────
mask_cs = (
    ((df["TF_A"] == "CEBPA") & (df["TF_B"] == "SPI1")) |
    ((df["TF_A"] == "SPI1")  & (df["TF_B"] == "CEBPA"))
)
assert mask_cs.sum() == 1, "CEBPA/SPI1 pair not found or duplicated!"
cs_idx = df[mask_cs].index[0]

# ── Compute ranks and percentiles ─────────────────────────────────────────────
N = len(df)
dims = [
    ("n_shared_targets",   "Evidence 1: GRN shared targets\n(n_shared_targets)"),
    ("depmap_r",           "Evidence 2: DepMap co-essentiality\n(Pearson r, 1,183 cell lines)"),
    ("ppi_score",          "Evidence 3: STRING protein–protein\ninteraction score"),
    ("shared_disease_score","Evidence 4: Shared disease associations\n(OpenTargets, score≥0.5)"),
]

summary_lines = ["CEBPA/SPI1 Evidence Convergence Summary", "=" * 50, ""]

cs_percentiles = {}
for col, _ in dims:
    vals = df[col].values
    cs_val = df.loc[cs_idx, col]
    # rank 1 = highest
    rank = int((vals > cs_val).sum()) + 1
    pct = 100.0 * (1 - rank / N)
    cs_percentiles[col] = (rank, pct, cs_val)
    line = f"{col}: value={cs_val:.4f}, rank={rank}/{N}, top {100-pct:.1f}% (percentile={pct:.1f}%)"
    summary_lines.append(line)
    print(line)

summary_lines.append("")
summary_lines.append("Other pairs in top 10% across ALL 4 dimensions simultaneously:")

# Compute percentile rank for every pair on all 4 dims
top10_masks = []
for col, _ in dims:
    vals = df[col].values
    # percentile of each value among all pairs (higher value = higher percentile)
    from scipy.stats import rankdata
    ranks = rankdata(vals, method='average')
    pct_rank = 100.0 * ranks / N
    top10_masks.append(pct_rank >= 90.0)

all_top10 = np.all(top10_masks, axis=0)
other_top10 = df[all_top10 & ~mask_cs.values]
if len(other_top10) == 0:
    msg = "  None — CEBPA/SPI1 is the ONLY pair in top 10% across all 4 dimensions."
else:
    msg = "\n".join(f"  {r['TF_A']}/{r['TF_B']}" for _, r in other_top10.iterrows())
summary_lines.append(msg)
print("")
print(summary_lines[-2])
print(msg)

with open(TXT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
print(f"\nSaved summary: {TXT_PATH}")

# ── Figure ─────────────────────────────────────────────────────────────────────
COLOR_HIGHLIGHT = "#d62728"
COLOR_OTHER     = "#cccccc"

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes_flat = axes.flatten()

for ax, (col, title) in zip(axes_flat, dims):
    # Sort all pairs by this dimension (ascending for bar chart from bottom)
    sorted_df = df.sort_values(col, ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(sorted_df))
    values = sorted_df[col].values

    # Color: highlight CEBPA/SPI1
    is_cs = (
        ((sorted_df["TF_A"] == "CEBPA") & (sorted_df["TF_B"] == "SPI1")) |
        ((sorted_df["TF_A"] == "SPI1")  & (sorted_df["TF_B"] == "CEBPA"))
    )
    colors = [COLOR_HIGHLIGHT if h else COLOR_OTHER for h in is_cs]

    ax.barh(y_pos, values, color=colors, height=1.0, linewidth=0)

    # Find CEBPA/SPI1 position
    cs_pos = int(np.where(is_cs.values)[0][0])
    cs_val = values[cs_pos]
    rank, pct, _ = cs_percentiles[col]

    # Annotation arrow + text
    ax.annotate(
        f"CEBPA/SPI1\nrank {rank}/{N}\ntop {100-pct:.1f}%",
        xy=(cs_val, cs_pos),
        xytext=(cs_val * 1.05 + (values.max() - values.min()) * 0.12, cs_pos),
        fontsize=9, color=COLOR_HIGHLIGHT, fontweight='bold',
        va='center',
        arrowprops=dict(arrowstyle='->', color=COLOR_HIGHLIGHT, lw=1.2),
    )

    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.set_xlabel(col.replace("_", " "), fontsize=9)
    ax.set_yticks([])
    ax.set_ylabel(f"All {N} pairs (ranked)", fontsize=9)
    ax.spines[['top', 'right', 'left']].set_visible(False)

    # Top-right corner annotation box
    ax.text(
        0.97, 0.97,
        f"CEBPA/SPI1:\nrank {rank}/{N} (top {100-pct:.1f}%)",
        transform=ax.transAxes,
        fontsize=8.5, color=COLOR_HIGHLIGHT, fontweight='bold',
        va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLOR_HIGHLIGHT, alpha=0.85)
    )

# Legend
legend_patches = [
    mpatches.Patch(color=COLOR_HIGHLIGHT, label='CEBPA / SPI1'),
    mpatches.Patch(color=COLOR_OTHER,     label='All other pairs'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=2, fontsize=10,
           frameon=False, bbox_to_anchor=(0.5, -0.01))

fig.suptitle(
    "CEBPA/SPI1 uniquely converges across 4 independent evidence dimensions",
    fontsize=13, fontweight='bold', y=1.01
)

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(FIG_PATH, dpi=200, bbox_inches='tight')
print(f"\nFigure saved: {FIG_PATH}")
plt.close()

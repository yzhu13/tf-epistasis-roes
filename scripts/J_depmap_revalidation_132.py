"""
Direction J: Re-run DepMap 25Q2 co-essentiality validation using the
GSE133344-derived ROES features for all 132 pairs (108 TFs) — much larger
than the original 17-TF/136-pair analysis (which used a different GRN).

For each of the 132 pairs, compute Pearson/Spearman co-essentiality of
CERES gene-effect scores across 1,183 cancer cell lines, then correlate
with n_shared_targets, jaccard_overlap, total_targets, synth_score, and
eps_l2 itself (does experimental epistasis predict cancer co-essentiality?).

Run: python scripts/J_depmap_revalidation_132.py
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
DATA_RAW = BASE / "data" / "raw"
PROC = BASE / "results" / "G_grn_roes_validation"
OUT = BASE / "results" / "J_depmap_revalidation_132"
OUT.mkdir(parents=True, exist_ok=True)
FIG = BASE / "figures" / "J_depmap_revalidation_132"
FIG.mkdir(parents=True, exist_ok=True)

# ── 1. Load DepMap ─────────────────────────────────────────────────
print("Loading DepMap CRISPRGeneEffect.csv ...")
ceres = pd.read_csv(DATA_RAW / "DepMap" / "CRISPRGeneEffect.csv", index_col=0)
ceres.columns = [c.split(" (")[0] for c in ceres.columns]
print(f"  Shape: {ceres.shape}")
genes_depmap = set(ceres.columns)

# ── 2. Load 132-pair ROES ───────────────────────────────────────────
roes = pd.read_csv(PROC / "roes_GSE133344_132pairs.csv")
roes["total_targets"] = roes["n_targets_A"] + roes["n_targets_B"]
print(f"\nROES pairs: {len(roes)}")

# ── 3. Compute co-essentiality ──────────────────────────────────────
print("\nComputing co-essentiality for each pair ...")
results = []
skipped = 0
for _, row in roes.iterrows():
    a, b = row["TF_A"], row["TF_B"]
    if a not in genes_depmap or b not in genes_depmap:
        skipped += 1
        continue
    ca = ceres[a].dropna()
    cb = ceres[b].dropna()
    common = ca.index.intersection(cb.index)
    if len(common) < 100:
        skipped += 1
        continue
    r, p = pearsonr(ca[common].values, cb[common].values)
    rho, p2 = spearmanr(ca[common].values, cb[common].values)

    rec = row.to_dict()
    rec.update({"depmap_r": r, "depmap_p": p, "depmap_rho": rho, "n_cell_lines": len(common)})
    results.append(rec)

df = pd.DataFrame(results)
print(f"  Computed: {len(df)}  Skipped: {skipped}")
df.to_csv(OUT / "depmap_coessentiality_132pairs.csv", index=False)

# ── 4. Correlations ────────────────────────────────────────────────
print("\n=== Feature -> depmap_r correlations (n={}) ===".format(len(df)))
corr_results = []
for feat in ["n_shared_targets", "jaccard_overlap", "parallel_fraction",
              "antagonism_score", "serial_score", "synth_score",
              "total_targets", "eps_l2", "eps_relative"]:
    v = df.dropna(subset=[feat, "depmap_r"])
    if v[feat].std() == 0:
        continue
    r, p = pearsonr(v[feat], v["depmap_r"])
    rho, p2 = spearmanr(v[feat], v["depmap_r"])
    corr_results.append({"feature": feat, "pearson_r": r, "pearson_p": p,
                          "spearman_rho": rho, "spearman_p": p2, "n": len(v)})
    print(f"  {feat:20s}: Pearson r={r:+.3f} (p={p:.3g})  Spearman rho={rho:+.3f} (p={p2:.3g})  n={len(v)}")

corr_df = pd.DataFrame(corr_results)
corr_df.to_csv(OUT / "correlation_results.csv", index=False)

# ── 5. High vs low n_shared_targets: Mann-Whitney ────────────────────
median_ns = df["n_shared_targets"].median()
high = df[df["n_shared_targets"] > median_ns]["depmap_r"].dropna()
low = df[df["n_shared_targets"] <= median_ns]["depmap_r"].dropna()
stat, p_mw = mannwhitneyu(high, low, alternative="greater")
print(f"\nMann-Whitney (high n_shared > low n_shared depmap_r): p={p_mw:.4g}")
print(f"  mean depmap_r: high={high.mean():.4f}  low={low.mean():.4f}")

# ── 6. Figure ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("DepMap Co-essentiality vs GSE133344-derived ROES (n={})".format(len(df)), fontsize=13, fontweight="bold")

ax = axes[0]
ax.scatter(df["n_shared_targets"], df["depmap_r"], s=25, alpha=0.6, color="steelblue")
v = df.dropna(subset=["n_shared_targets","depmap_r"])
r,p = pearsonr(v["n_shared_targets"], v["depmap_r"])
m,b = np.polyfit(v["n_shared_targets"], v["depmap_r"], 1)
xr = np.linspace(v["n_shared_targets"].min(), v["n_shared_targets"].max(), 100)
ax.plot(xr, m*xr+b, "r--", alpha=0.6)
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.text(0.05,0.95, f"r={r:.3f}\np={p:.2e}", transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round",facecolor="wheat",alpha=0.6))
ax.set_xlabel("n_shared_targets", fontsize=10)
ax.set_ylabel("DepMap co-essentiality (Pearson r)", fontsize=10)
ax.set_title("Shared Targets vs Co-essentiality", fontsize=10)

ax = axes[1]
ax.scatter(df["eps_l2"], df["depmap_r"], s=25, alpha=0.6, color="darkorange")
v = df.dropna(subset=["eps_l2","depmap_r"])
r,p = pearsonr(v["eps_l2"], v["depmap_r"])
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.text(0.05,0.95, f"r={r:.3f}\np={p:.2e}", transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round",facecolor="wheat",alpha=0.6))
ax.set_xlabel("eps_l2 (experimental epistasis magnitude)", fontsize=10)
ax.set_ylabel("DepMap co-essentiality (Pearson r)", fontsize=10)
ax.set_title("Experimental Epistasis vs Cancer Co-essentiality", fontsize=10)

ax = axes[2]
ax.hist(low.values, bins=15, alpha=0.6, color="steelblue", label=f"Low n_shared (n={len(low)})", density=True)
ax.hist(high.values, bins=15, alpha=0.6, color="darkorange", label=f"High n_shared (n={len(high)})", density=True)
ax.axvline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("DepMap co-essentiality (Pearson r)", fontsize=10)
ax.set_ylabel("density", fontsize=10)
ax.set_title(f"High vs Low n_shared (Mann-Whitney p={p_mw:.3g})", fontsize=10)
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(FIG / "J_depmap_132pairs.png", dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {FIG / 'J_depmap_132pairs.png'}")

# ── 7. Top concordant pairs ───────────────────────────────────────
df["composite_rank"] = (df["n_shared_targets"].rank(pct=True) + df["depmap_r"].rank(pct=True)) / 2
top = df.nlargest(15, "composite_rank")
print("\n=== Top 15 concordant pairs (high n_shared + high depmap_r) ===")
print(top[["TF_A","TF_B","n_shared_targets","jaccard_overlap","eps_l2","depmap_r","depmap_p","n_cell_lines"]].to_string(index=False))
top.to_csv(OUT / "top_concordant_pairs.csv", index=False)

with open(OUT / "summary.txt", "w") as f:
    f.write("="*60 + "\n")
    f.write("DIRECTION J: DEPMAP REVALIDATION (n={})\n".format(len(df)))
    f.write("="*60 + "\n\n")
    f.write(corr_df.to_string(index=False))
    f.write(f"\n\nMann-Whitney high vs low n_shared_targets: p={p_mw:.4g}\n")
    f.write(f"  mean depmap_r: high={high.mean():.4f}  low={low.mean():.4f}\n")
    f.write("\n\nTop 15 concordant pairs:\n")
    f.write(top[["TF_A","TF_B","n_shared_targets","jaccard_overlap","eps_l2","depmap_r","depmap_p","n_cell_lines"]].to_string(index=False))

print("\nDone.")

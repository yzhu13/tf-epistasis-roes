"""
Direction G: Build a GRN directly from the GSE133344 single-perturbation
pseudobulk profiles (108 TFs, same dataset/cells as the 132-pair ground
truth from Direction F), then compute ROES features for all 132 pairs and
re-validate against eps_l2 / eps_relative.

Why a new GRN: the original GRN (Norman subset 17 TFs + Replogle K562) only
overlaps 29/73 TFs appearing in the GSE133344 ground truth. Building edges
directly from the 108 GSE133344 singles gives full coverage AND removes any
cross-dataset batch effects (same cells, same normalization).

Edge definition: for TF X, target gene G is an edge if
  z = (delta_X[G] - mean(delta_all[G])) / std(delta_all[G])   |z| > Z_THRESH
where delta_X = E_singleX - E_ctrl (log1p CP10K pseudobulk).

ROES features per pair (A,B):
  n_shared_targets, jaccard_overlap, parallel_fraction, antagonism_score,
  serial_score (A->B or B->A direct edges), synth_score = n_shared*parallel_fraction

Run: python scripts/G_grn_from_GSE133344_and_roes_validation.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
PROC = BASE / "results" / "F_full_groundtruth"
OUT = BASE / "results" / "G_grn_roes_validation"
OUT.mkdir(parents=True, exist_ok=True)
FIG = BASE / "figures" / "G_grn_roes_validation"
FIG.mkdir(parents=True, exist_ok=True)

Z_THRESH = 2.5

# ── 1. Load pseudobulk + ground truth ────────────────────────────
print("Loading pseudobulk profiles ...")
pb = pd.read_csv(PROC / "pseudobulk_all_conditions.csv.gz", index_col=0)
print(f"  Pseudobulk: {pb.shape} (genes x conditions)")

gt = pd.read_csv(PROC / "epistasis_groundtruth_GSE133344.csv")
print(f"  Ground truth pairs: {len(gt)}")

ctrl_col = "NegCtrl0_NegCtrl0__NegCtrl0_NegCtrl0"
E_ctrl = pb[ctrl_col]

# Identify single-TF columns
def parse_guide(g):
    parts = g.split("__")[0].split("_")
    a, b = parts[0], "_".join(parts[1:]) if len(parts) > 2 else parts[1]
    return a, b

singles = {}
for c in pb.columns:
    a, b = parse_guide(c)
    if a == "NegCtrl0" and b == "NegCtrl0":
        continue
    elif a == "NegCtrl0":
        singles[b] = c
    elif b == "NegCtrl0":
        singles[a] = c

print(f"  Single-TF conditions: {len(singles)}")

# ── 2. Compute delta matrix (TF x gene) ──────────────────────────
print("\nComputing delta (single - ctrl) matrix ...")
tf_list = sorted(singles.keys())
delta = pd.DataFrame({tf: pb[singles[tf]] - E_ctrl for tf in tf_list})  # genes x TFs
print(f"  Delta matrix: {delta.shape} (genes x TFs)")

# ── 3. Z-score each gene's delta across TFs, build edges ─────────
print(f"\nBuilding GRN edges (|z| > {Z_THRESH}) ...")
mu = delta.mean(axis=1)
sigma = delta.std(axis=1).replace(0, np.nan)
z = delta.sub(mu, axis=0).div(sigma, axis=0)

edges = []
for tf in tf_list:
    col = z[tf].dropna()
    hits = col[col.abs() > Z_THRESH]
    for gene, zval in hits.items():
        if gene == tf:
            continue  # skip self-edges
        edges.append({
            "TF": tf, "target_gene": gene,
            "edge_type": "activation" if zval > 0 else "repression",
            "z": zval,
        })

grn = pd.DataFrame(edges)
print(f"  Total edges: {len(grn)}  across {grn['TF'].nunique()} TFs")
grn.to_csv(OUT / "grn_GSE133344_singles.csv", index=False)

# ── 4. Build per-TF target sets (signed) ─────────────────────────
tf_targets = {}      # TF -> set of target genes
tf_targets_signed = {}  # TF -> {gene: sign}
for tf, sub in grn.groupby("TF"):
    tf_targets[tf] = set(sub["target_gene"])
    tf_targets_signed[tf] = dict(zip(sub["target_gene"], np.sign(sub["z"])))

# direct edges between TFs (for serial_score)
direct_edge = set(zip(grn["TF"], grn["target_gene"]))

# ── 5. Compute ROES features for each ground-truth pair ──────────
print("\nComputing ROES features for 132 GT pairs ...")
roes_rows = []
for _, row in gt.iterrows():
    a, b = row["TF_A"], row["TF_B"]
    targets_a = tf_targets.get(a, set())
    targets_b = tf_targets.get(b, set())
    shared = targets_a & targets_b
    union = targets_a | targets_b

    n_shared = len(shared)
    jaccard = n_shared / len(union) if union else 0.0

    if n_shared > 0:
        signs_a = tf_targets_signed[a]
        signs_b = tf_targets_signed[b]
        same_dir = sum(1 for g in shared if signs_a[g] == signs_b[g])
        parallel_fraction = same_dir / n_shared
        antagonism_score = 1 - parallel_fraction
    else:
        parallel_fraction = 0.0
        antagonism_score = 0.0

    serial_ab = 1 if (a, b) in direct_edge else 0
    serial_ba = 1 if (b, a) in direct_edge else 0
    serial_score = serial_ab + serial_ba

    synth_score = n_shared * parallel_fraction

    roes_rows.append({
        "TF_A": a, "TF_B": b,
        "n_targets_A": len(targets_a), "n_targets_B": len(targets_b),
        "n_shared_targets": n_shared, "jaccard_overlap": jaccard,
        "parallel_fraction": parallel_fraction, "antagonism_score": antagonism_score,
        "serial_score": serial_score, "synth_score": synth_score,
        "eps_l2": row["eps_l2"], "eps_relative": row["eps_relative"],
        "eps_type": row["eps_type"],
    })

roes = pd.DataFrame(roes_rows)
roes.to_csv(OUT / "roes_GSE133344_132pairs.csv", index=False)
print(f"  Saved: {OUT / 'roes_GSE133344_132pairs.csv'}")
print(f"  Pairs with n_shared_targets > 0: {(roes['n_shared_targets']>0).sum()} / {len(roes)}")

# ── 6. Correlations: ROES features vs eps_l2 / eps_relative ──────
print("\n=== ROES features vs eps_l2 (n=132) ===")
corr_results = []
for feat in ["n_shared_targets", "jaccard_overlap", "parallel_fraction",
              "antagonism_score", "serial_score", "synth_score"]:
    for target in ["eps_l2", "eps_relative"]:
        v = roes.dropna(subset=[feat, target])
        if v[feat].std() == 0:
            continue
        r, p = pearsonr(v[feat], v[target])
        rho, p2 = spearmanr(v[feat], v[target])
        corr_results.append({"feature": feat, "target": target,
                              "pearson_r": r, "pearson_p": p,
                              "spearman_rho": rho, "spearman_p": p2, "n": len(v)})
        print(f"  {feat:20s} -> {target:14s}: Pearson r={r:+.3f} (p={p:.4g})  "
              f"Spearman rho={rho:+.3f} (p={p2:.4g})  n={len(v)}")

corr_df = pd.DataFrame(corr_results)
corr_df.to_csv(OUT / "correlation_results.csv", index=False)

# ── 7. Figure ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("ROES (from GSE133344-derived GRN) vs Epistasis Ground Truth (n=132)",
              fontsize=13, fontweight="bold")

ax = axes[0]
ax.scatter(roes["n_shared_targets"], roes["eps_l2"], s=30, alpha=0.6, color="steelblue")
v = roes.dropna(subset=["n_shared_targets","eps_l2"])
r, p = pearsonr(v["n_shared_targets"], v["eps_l2"])
m, b = np.polyfit(v["n_shared_targets"], v["eps_l2"], 1)
xr = np.linspace(v["n_shared_targets"].min(), v["n_shared_targets"].max(), 100)
ax.plot(xr, m*xr+b, "r--", alpha=0.6)
ax.text(0.05, 0.95, f"Pearson r={r:.3f}\np={p:.2e}\nn={len(v)}",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
ax.set_xlabel("n_shared_targets", fontsize=11)
ax.set_ylabel("eps_l2", fontsize=11)
ax.set_title("Shared Targets vs Epistasis Magnitude", fontsize=10)

ax = axes[1]
ax.scatter(roes["jaccard_overlap"], roes["eps_l2"], s=30, alpha=0.6, color="darkorange")
v = roes.dropna(subset=["jaccard_overlap","eps_l2"])
r, p = pearsonr(v["jaccard_overlap"], v["eps_l2"])
ax.text(0.05, 0.95, f"Pearson r={r:.3f}\np={p:.2e}\nn={len(v)}",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
ax.set_xlabel("jaccard_overlap", fontsize=11)
ax.set_ylabel("eps_l2", fontsize=11)
ax.set_title("Jaccard Overlap vs Epistasis Magnitude", fontsize=10)

ax = axes[2]
sc = ax.scatter(roes["synth_score"], roes["eps_l2"], c=roes["parallel_fraction"],
                cmap="RdYlBu", s=30, alpha=0.8)
plt.colorbar(sc, ax=ax, label="parallel_fraction")
v = roes.dropna(subset=["synth_score","eps_l2"])
r, p = pearsonr(v["synth_score"], v["eps_l2"])
ax.text(0.05, 0.95, f"Pearson r={r:.3f}\np={p:.2e}",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
ax.set_xlabel("synth_score (n_shared x parallel_frac)", fontsize=11)
ax.set_ylabel("eps_l2", fontsize=11)
ax.set_title("Composite Score vs Epistasis Magnitude", fontsize=10)

plt.tight_layout()
fig.savefig(FIG / "G_roes_validation_132pairs.png", dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {FIG / 'G_roes_validation_132pairs.png'}")

# ── 8. Top pairs ───────────────────────────────────────────────────
print("\n=== Top 15 pairs by eps_l2 (with ROES features) ===")
print(roes.nlargest(15, "eps_l2")[["TF_A","TF_B","eps_l2","n_shared_targets",
                                     "jaccard_overlap","parallel_fraction","synth_score","eps_type"]].to_string(index=False))

print("\n=== Top 15 pairs by synth_score ===")
print(roes.nlargest(15, "synth_score")[["TF_A","TF_B","synth_score","n_shared_targets",
                                          "parallel_fraction","eps_l2","eps_type"]].to_string(index=False))

# ── 9. Summary ──────────────────────────────────────────────────────
with open(OUT / "summary.txt", "w") as f:
    f.write("="*60 + "\n")
    f.write("DIRECTION G: GSE133344-DERIVED GRN + ROES VALIDATION (n=132)\n")
    f.write("="*60 + "\n\n")
    f.write(f"GRN: {len(grn)} edges across {grn['TF'].nunique()} TFs (Z_THRESH={Z_THRESH})\n")
    f.write(f"Ground truth pairs: {len(roes)}\n")
    f.write(f"Pairs with shared targets > 0: {(roes['n_shared_targets']>0).sum()}\n\n")
    f.write("Correlations (ROES feature -> eps_l2 / eps_relative):\n")
    f.write(corr_df.to_string(index=False))
    f.write("\n\nTop 15 by eps_l2:\n")
    f.write(roes.nlargest(15, "eps_l2")[["TF_A","TF_B","eps_l2","n_shared_targets","jaccard_overlap","parallel_fraction","synth_score","eps_type"]].to_string(index=False))
    f.write("\n\nTop 15 by synth_score:\n")
    f.write(roes.nlargest(15, "synth_score")[["TF_A","TF_B","synth_score","n_shared_targets","parallel_fraction","eps_l2","eps_type"]].to_string(index=False))

print("\nDone.")

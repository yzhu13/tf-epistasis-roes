"""
Direction I: Mechanism behind the jaccard_overlap sign-reversal.

Direction G found (n=132, GSE133344):
  n_shared_targets -> eps_l2 : r=+0.513, p=3.3e-10
  jaccard_overlap  -> eps_l2 : r=-0.539, p=2.6e-11

Hypothesis: jaccard = n_shared / (|A| + |B| - n_shared). "Hub" TFs (large
|A|, |B|, e.g. CEBP family with ~400 targets) mechanically have LOW jaccard
even when n_shared is large, and these hub pairs are exactly the ones with
the strongest epistasis. So jaccard's negative correlation with eps_l2 may
be largely a proxy for "total regulatory promiscuity" (|A|+|B|), not a
genuine independent signal.

Tests:
  1. total_targets (|A|+|B|) vs eps_l2 and vs jaccard_overlap
  2. partial correlation: jaccard vs eps_l2 controlling for n_shared_targets
     and for total_targets
  3. multiple regression eps_l2 ~ n_shared_targets + jaccard_overlap
     (standardized coefficients)

Run: python scripts/I_jaccard_mechanism.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
PROC = BASE / "results" / "G_grn_roes_validation"
OUT = BASE / "results" / "I_jaccard_mechanism"
OUT.mkdir(parents=True, exist_ok=True)
FIG = BASE / "figures" / "I_jaccard_mechanism"
FIG.mkdir(parents=True, exist_ok=True)

roes = pd.read_csv(PROC / "roes_GSE133344_132pairs.csv")
roes["total_targets"] = roes["n_targets_A"] + roes["n_targets_B"]
print(f"Loaded {len(roes)} pairs")

# ── 1. total_targets correlations ─────────────────────────────────
print("\n=== total_targets (|A|+|B|) correlations ===")
for target in ["eps_l2", "jaccard_overlap", "n_shared_targets"]:
    r, p = pearsonr(roes["total_targets"], roes[target])
    print(f"  total_targets -> {target:18s}: r={r:+.3f}  p={p:.3g}")

# ── 2. Partial correlations via residualization ────────────────────
def partial_corr(x, y, z):
    """Partial correlation between x and y, controlling for z (linear)."""
    bx = np.polyfit(z, x, 1)
    by = np.polyfit(z, y, 1)
    res_x = x - np.polyval(bx, z)
    res_y = y - np.polyval(by, z)
    return pearsonr(res_x, res_y)

print("\n=== Partial correlations: jaccard_overlap vs eps_l2 ===")
r0, p0 = pearsonr(roes["jaccard_overlap"], roes["eps_l2"])
print(f"  raw                              : r={r0:+.3f}  p={p0:.3g}")

r1, p1 = partial_corr(roes["jaccard_overlap"], roes["eps_l2"], roes["n_shared_targets"])
print(f"  controlling for n_shared_targets : r={r1:+.3f}  p={p1:.3g}")

r2, p2 = partial_corr(roes["jaccard_overlap"], roes["eps_l2"], roes["total_targets"])
print(f"  controlling for total_targets    : r={r2:+.3f}  p={p2:.3g}")

print("\n=== Partial correlations: n_shared_targets vs eps_l2 ===")
r3, p3 = pearsonr(roes["n_shared_targets"], roes["eps_l2"])
print(f"  raw                          : r={r3:+.3f}  p={p3:.3g}")
r4, p4 = partial_corr(roes["n_shared_targets"], roes["eps_l2"], roes["jaccard_overlap"])
print(f"  controlling for jaccard      : r={r4:+.3f}  p={p4:.3g}")
r5, p5 = partial_corr(roes["n_shared_targets"], roes["eps_l2"], roes["total_targets"])
print(f"  controlling for total_targets: r={r5:+.3f}  p={p5:.3g}")

# ── 3. Multiple regression (standardized) ───────────────────────────
print("\n=== Multiple linear regression: eps_l2 ~ z(n_shared) + z(jaccard) ===")
from numpy.linalg import lstsq
def zscore(x):
    return (x - x.mean()) / x.std()

X = np.column_stack([zscore(roes["n_shared_targets"]), zscore(roes["jaccard_overlap"]), np.ones(len(roes))])
y = zscore(roes["eps_l2"])
coef, _, _, _ = lstsq(X, y, rcond=None)
print(f"  beta(n_shared_targets) = {coef[0]:+.3f}")
print(f"  beta(jaccard_overlap)  = {coef[1]:+.3f}")
print(f"  intercept              = {coef[2]:+.3f}")
pred = X @ coef
r_model, p_model = pearsonr(pred, y)
print(f"  Model fit: r={r_model:.3f}  (R^2={r_model**2:.3f})")

# Compare to single-feature R^2
print(f"\n  R^2 (n_shared alone)  = {r3**2:.3f}")
print(f"  R^2 (jaccard alone)   = {r0**2:.3f}")
print(f"  R^2 (both, model)     = {r_model**2:.3f}")

# ── 4. Figure ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Mechanism of jaccard_overlap sign-reversal (n=132)", fontsize=13, fontweight="bold")

ax = axes[0]
sc = ax.scatter(roes["total_targets"], roes["jaccard_overlap"], c=roes["n_shared_targets"], cmap="viridis", s=25, alpha=0.8)
plt.colorbar(sc, ax=ax, label="n_shared_targets")
r,p = pearsonr(roes["total_targets"], roes["jaccard_overlap"])
ax.text(0.05,0.95, f"r={r:.3f}\np={p:.2e}", transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round",facecolor="wheat",alpha=0.6))
ax.set_xlabel("total_targets (|A|+|B|)", fontsize=10)
ax.set_ylabel("jaccard_overlap", fontsize=10)
ax.set_title("Hub TFs mechanically have low jaccard", fontsize=10)

ax = axes[1]
sc = ax.scatter(roes["total_targets"], roes["eps_l2"], c=roes["jaccard_overlap"], cmap="RdYlBu_r", s=25, alpha=0.8)
plt.colorbar(sc, ax=ax, label="jaccard_overlap")
r,p = pearsonr(roes["total_targets"], roes["eps_l2"])
ax.text(0.05,0.95, f"r={r:.3f}\np={p:.2e}", transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round",facecolor="wheat",alpha=0.6))
ax.set_xlabel("total_targets (|A|+|B|)", fontsize=10)
ax.set_ylabel("eps_l2", fontsize=10)
ax.set_title("Hub TFs (high total_targets) -> high epistasis", fontsize=10)

ax = axes[2]
labels = ["n_shared\n(raw)", "n_shared\n(partial,\nctrl jaccard)", "jaccard\n(raw)", "jaccard\n(partial,\nctrl n_shared)"]
vals = [r3, r4, r0, r1]
colors_bar = ["steelblue","steelblue","darkorange","darkorange"]
ax.bar(labels, vals, color=colors_bar, alpha=0.8)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Pearson r with eps_l2", fontsize=10)
ax.set_title("Raw vs partial correlations", fontsize=10)
for i,v in enumerate(vals):
    ax.text(i, v + (0.02 if v>=0 else -0.05), f"{v:.2f}", ha="center", fontsize=9)

plt.tight_layout()
fig.savefig(FIG / "I_jaccard_mechanism.png", dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {FIG / 'I_jaccard_mechanism.png'}")

# ── 5. Summary ─────────────────────────────────────────────────────
with open(OUT / "summary.txt", "w") as f:
    f.write("="*60 + "\n")
    f.write("DIRECTION I: JACCARD SIGN-REVERSAL MECHANISM (n=132)\n")
    f.write("="*60 + "\n\n")
    f.write("total_targets (|A|+|B|) correlations:\n")
    for target in ["eps_l2", "jaccard_overlap", "n_shared_targets"]:
        r, p = pearsonr(roes["total_targets"], roes[target])
        f.write(f"  total_targets -> {target:18s}: r={r:+.3f}  p={p:.3g}\n")
    f.write("\nPartial correlations (jaccard vs eps_l2):\n")
    f.write(f"  raw                              : r={r0:+.3f}  p={p0:.3g}\n")
    f.write(f"  controlling for n_shared_targets : r={r1:+.3f}  p={p1:.3g}\n")
    f.write(f"  controlling for total_targets    : r={r2:+.3f}  p={p2:.3g}\n")
    f.write("\nPartial correlations (n_shared_targets vs eps_l2):\n")
    f.write(f"  raw                          : r={r3:+.3f}  p={p3:.3g}\n")
    f.write(f"  controlling for jaccard      : r={r4:+.3f}  p={p4:.3g}\n")
    f.write(f"  controlling for total_targets: r={r5:+.3f}  p={p5:.3g}\n")
    f.write(f"\nMultiple regression (standardized): eps_l2 ~ n_shared + jaccard\n")
    f.write(f"  beta(n_shared_targets) = {coef[0]:+.3f}\n")
    f.write(f"  beta(jaccard_overlap)  = {coef[1]:+.3f}\n")
    f.write(f"  Model R^2 = {r_model**2:.3f}  (n_shared alone R^2={r3**2:.3f}, jaccard alone R^2={r0**2:.3f})\n")

print("\nDone.")

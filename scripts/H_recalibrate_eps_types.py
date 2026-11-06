"""
Direction H: Recalibrate epistasis-type classification for the 132-pair
GSE133344 ground truth.

The original classifier (used for the n=12/27 pairs) used a fixed
eps_relative < 0.1 threshold for "additive". On GSE133344, eps_relative
ranges 0.44-1.41 (much higher noise floor from single-cell pseudobulk at
larger scale), so that threshold yields zero "additive" calls.

New scheme, data-driven from the 132-pair distribution, using two axes:
  - magnitude_ratio = ||delta_AB|| / ||expected_additive||
      >1 => observed double-KO effect LARGER than additive expectation (synergy-like)
      <1 => observed double-KO effect SMALLER than additive expectation (buffering-like)
  - cos_obs_expected = cosine(delta_AB, expected_additive)
      high (~1) => same direction as additive expectation (concordant)
      low/negative => observed effect points in a different/opposite direction (complex)

Classification (thresholds = tertiles of the 132-pair distributions):
  - synergistic : magnitude_ratio > q66  AND cos_obs_expected > q33
  - buffering   : magnitude_ratio < q33  AND cos_obs_expected > q33
  - additive    : q33 <= magnitude_ratio <= q66 AND cos_obs_expected > q66
  - complex     : cos_obs_expected <= q33 (direction mismatch, regardless of magnitude)

Run: python scripts/H_recalibrate_eps_types.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
PROC = BASE / "results" / "F_full_groundtruth"
OUT = BASE / "results" / "H_recalibrated_eps_types"
OUT.mkdir(parents=True, exist_ok=True)
FIG = BASE / "figures" / "H_recalibrated_eps_types"
FIG.mkdir(parents=True, exist_ok=True)

print("Loading pseudobulk + ground truth ...")
pb = pd.read_csv(PROC / "pseudobulk_all_conditions.csv.gz", index_col=0)
gt = pd.read_csv(PROC / "epistasis_groundtruth_GSE133344.csv")
ctrl_col = "NegCtrl0_NegCtrl0__NegCtrl0_NegCtrl0"
E_ctrl = pb[ctrl_col]

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

# ── Recompute magnitude_ratio and cos_obs_expected for all 132 pairs ──
print("\nRecomputing magnitude_ratio and cos_obs_expected ...")
rows = []
for _, row in gt.iterrows():
    a, b, cond_ab = row["TF_A"], row["TF_B"], row["perturbation_key"]
    E_A = pb[singles[a]].values
    E_B = pb[singles[b]].values
    E_AB = pb[cond_ab].values

    delta_A = E_A - E_ctrl.values
    delta_B = E_B - E_ctrl.values
    delta_AB = E_AB - E_ctrl.values
    expected = delta_A + delta_B

    norm_obs = np.linalg.norm(delta_AB)
    norm_exp = np.linalg.norm(expected)
    magnitude_ratio = norm_obs / norm_exp if norm_exp > 0 else np.nan

    nu, nv = np.linalg.norm(delta_AB), np.linalg.norm(expected)
    cos_obs_expected = np.dot(delta_AB, expected) / (nu*nv) if nu > 0 and nv > 0 else 0.0

    rows.append({
        "TF_A": a, "TF_B": b, "eps_l2": row["eps_l2"], "eps_relative": row["eps_relative"],
        "magnitude_ratio": magnitude_ratio, "cos_obs_expected": cos_obs_expected,
        "old_eps_type": row["eps_type"],
    })

df = pd.DataFrame(rows)

# ── Tertile thresholds ────────────────────────────────────────────
q33_mag, q66_mag = df["magnitude_ratio"].quantile([0.333, 0.667])
q33_cos, q66_cos = df["cos_obs_expected"].quantile([0.333, 0.667])
print(f"\nmagnitude_ratio tertiles: q33={q33_mag:.3f}  q66={q66_mag:.3f}")
print(f"cos_obs_expected tertiles: q33={q33_cos:.3f}  q66={q66_cos:.3f}")
print(f"\nmagnitude_ratio range: {df['magnitude_ratio'].min():.3f} - {df['magnitude_ratio'].max():.3f}")
print(f"cos_obs_expected range: {df['cos_obs_expected'].min():.3f} - {df['cos_obs_expected'].max():.3f}")

def classify(row):
    mr, cos = row["magnitude_ratio"], row["cos_obs_expected"]
    if cos <= q33_cos:
        return "complex"
    if mr > q66_mag:
        return "synergistic"
    if mr < q33_mag:
        return "buffering"
    return "additive"

df["new_eps_type"] = df.apply(classify, axis=1)

print("\nNew eps_type distribution:")
print(df["new_eps_type"].value_counts())
print("\nOld vs new crosstab:")
print(pd.crosstab(df["old_eps_type"], df["new_eps_type"]))

out_path = OUT / "epistasis_groundtruth_recalibrated.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# ── Figure: 2D scatter of the two axes, colored by new_eps_type ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Recalibrated Epistasis Classification (n=132, GSE133344)", fontsize=13, fontweight="bold")

ax = axes[0]
colors = {"synergistic":"crimson", "buffering":"steelblue", "additive":"seagreen", "complex":"goldenrod"}
for t, c in colors.items():
    sub = df[df["new_eps_type"] == t]
    ax.scatter(sub["magnitude_ratio"], sub["cos_obs_expected"], s=25, alpha=0.7, color=c, label=f"{t} (n={len(sub)})")
ax.axvline(q33_mag, color="gray", ls="--", lw=0.8)
ax.axvline(q66_mag, color="gray", ls="--", lw=0.8)
ax.axhline(q33_cos, color="gray", ls="--", lw=0.8)
ax.axhline(q66_cos, color="gray", ls="--", lw=0.8)
ax.axvline(1.0, color="black", ls=":", lw=1)
ax.set_xlabel("magnitude_ratio = ||delta_AB|| / ||delta_A + delta_B||", fontsize=10)
ax.set_ylabel("cos(delta_AB, delta_A + delta_B)", fontsize=10)
ax.set_title("Classification space (tertile thresholds)", fontsize=10)
ax.legend(fontsize=8)

ax = axes[1]
ax.hist(df["eps_l2"], bins=20, color="slateblue", alpha=0.7)
ax.set_xlabel("eps_l2", fontsize=10)
ax.set_ylabel("count", fontsize=10)
ax.set_title("eps_l2 distribution (n=132)", fontsize=10)

plt.tight_layout()
fig.savefig(FIG / "H_recalibrated_classification.png", dpi=150, bbox_inches="tight")
print(f"Figure saved: {FIG / 'H_recalibrated_classification.png'}")

# ── Top examples per type ─────────────────────────────────────────
with open(OUT / "summary.txt", "w") as f:
    f.write("="*60 + "\n")
    f.write("RECALIBRATED EPS_TYPE CLASSIFICATION (n=132)\n")
    f.write("="*60 + "\n\n")
    f.write(f"magnitude_ratio tertiles: q33={q33_mag:.3f}  q66={q66_mag:.3f}\n")
    f.write(f"cos_obs_expected tertiles: q33={q33_cos:.3f}  q66={q66_cos:.3f}\n\n")
    f.write("New eps_type distribution:\n")
    f.write(df["new_eps_type"].value_counts().to_string())
    f.write("\n\nOld vs new crosstab:\n")
    f.write(pd.crosstab(df["old_eps_type"], df["new_eps_type"]).to_string())
    for t in colors:
        f.write(f"\n\nTop 5 '{t}' pairs by eps_l2:\n")
        sub = df[df["new_eps_type"]==t].nlargest(5, "eps_l2")
        f.write(sub[["TF_A","TF_B","eps_l2","magnitude_ratio","cos_obs_expected"]].to_string(index=False))

print("\nDone.")

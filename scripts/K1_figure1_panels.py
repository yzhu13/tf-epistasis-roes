"""
Generate Figure 1 (A/B/C panels + combined):
  A - ROES framework schematic (conceptual diagram, no data)
  B - Norman 2019 (n=12): n_shared_targets vs eps_l2
  C - GSE133344 (n=132): n_shared_targets vs eps_l2

Run: python scripts/K1_figure1_panels.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, FancyArrowPatch
from scipy.stats import pearsonr
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
FIG = BASE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11})

# ─────────────────────────────────────────────────────────────────
# Panel A: ROES schematic
# ─────────────────────────────────────────────────────────────────
def make_panel_A(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Two overlapping circles representing target sets of TF-A and TF-B
    cA = Circle((3.6, 5.5), 2.4, facecolor="#aec7e8", edgecolor="#1f77b4", lw=2, alpha=0.6)
    cB = Circle((5.6, 5.5), 2.4, facecolor="#ffbb78", edgecolor="#ff7f0e", lw=2, alpha=0.6)
    ax.add_patch(cA)
    ax.add_patch(cB)

    ax.text(2.3, 8.2, "TF A", fontsize=14, fontweight="bold", color="#1f77b4", ha="center")
    ax.text(6.9, 8.2, "TF B", fontsize=14, fontweight="bold", color="#ff7f0e", ha="center")
    ax.text(2.0, 5.5, "targets(A)\n|A|", fontsize=10, ha="center", color="#1f77b4")
    ax.text(7.2, 5.5, "targets(B)\n|B|", fontsize=10, ha="center", color="#ff7f0e")
    ax.text(4.6, 5.5, "shared\nn_shared", fontsize=10, ha="center", fontweight="bold", color="#333333")

    # Arrow to ROES box
    arrow = FancyArrowPatch((8.3, 5.5), (9.7, 5.5), mutation_scale=18, color="black",
                             arrowstyle="-|>", lw=1.5)
    # keep arrow within bounds; redraw with adjusted coordinates below
    ax.annotate("", xy=(8.5, 8.7), xytext=(7.0, 7.6),
                 arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))

    # ROES formula box
    ax.text(8.7, 9.2,
            "ROES features:\n"
            "  n_shared_targets\n"
            "  jaccard = n_shared / |A∪B|\n"
            "  total_targets = |A|+|B|\n"
            "  parallel_fraction\n"
            "  synth_score = n_shared × parallel_frac",
            fontsize=8.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="gray"))

    # Bottom: predicted epistasis arrow
    ax.annotate("", xy=(4.6, 0.6), xytext=(4.6, 2.7),
                 arrowprops=dict(arrowstyle="->", lw=2, color="#d62728"))
    ax.text(4.6, 0.0, "Predicted epistasis magnitude (eps_l2)\n"
                       "high n_shared / high total_targets\n"
                       "→ strong epistasis (synergy or buffering)",
            fontsize=9, ha="center", va="top", color="#d62728")

    ax.set_title("A. ROES framework: GRN topology → predicted epistasis",
                  fontsize=11, fontweight="bold", loc="left")


# ─────────────────────────────────────────────────────────────────
# Panel B: Norman 2019 (n=12)
# ─────────────────────────────────────────────────────────────────
def make_panel_B(ax):
    df = pd.read_csv(BASE / "data" / "processed" / "roes_validation_norman2019.csv")
    r, p = pearsonr(df["n_shared_targets"], df["eps_l2"])
    m, b = np.polyfit(df["n_shared_targets"], df["eps_l2"], 1)
    xr = np.linspace(df["n_shared_targets"].min(), df["n_shared_targets"].max(), 100)

    colors = df["eps_type"].map({"buffering": "#1f77b4", "complex": "#ff7f0e"}).fillna("#888888")
    ax.scatter(df["n_shared_targets"], df["eps_l2"], s=70, c=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.plot(xr, m * xr + b, "r--", alpha=0.7, lw=1.5)

    for _, row in df.iterrows():
        if row["n_shared_targets"] > 100 or row["eps_l2"] > 6:
            ax.annotate(f"{row.TF_A}/{row.TF_B}", (row["n_shared_targets"], row["eps_l2"]),
                         fontsize=7.5, xytext=(4, 4), textcoords="offset points")

    ax.text(0.05, 0.95, f"Pearson r={r:.3f}\np={p:.3g}\nn=12", transform=ax.transAxes,
            va="top", fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    ax.set_xlabel("n_shared_targets (ROES)", fontsize=10)
    ax.set_ylabel("Epistasis magnitude (eps_l2)", fontsize=10)
    ax.set_title("B. Norman 2019 (n=12)", fontsize=11, fontweight="bold", loc="left")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#1f77b4", label="buffering"),
                        Patch(facecolor="#ff7f0e", label="complex")], fontsize=8, loc="lower right")


# ─────────────────────────────────────────────────────────────────
# Panel C: GSE133344 (n=132)
# ─────────────────────────────────────────────────────────────────
def make_panel_C(ax):
    df = pd.read_csv(BASE / "results" / "G_grn_roes_validation" / "roes_GSE133344_132pairs.csv")
    r, p = pearsonr(df["n_shared_targets"], df["eps_l2"])
    m, b = np.polyfit(df["n_shared_targets"], df["eps_l2"], 1)
    xr = np.linspace(df["n_shared_targets"].min(), df["n_shared_targets"].max(), 100)

    ax.scatter(df["n_shared_targets"], df["eps_l2"], s=35, alpha=0.55, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.plot(xr, m * xr + b, "r--", alpha=0.7, lw=1.5)

    # Annotate top pairs (CEBP family hubs)
    top = df.nlargest(3, "eps_l2")
    for _, row in top.iterrows():
        xytext = (-70, -12) if row["n_shared_targets"] < 100 else (5, 5)
        ax.annotate(f"{row.TF_A}/{row.TF_B}", (row["n_shared_targets"], row["eps_l2"]),
                     fontsize=8, xytext=xytext, textcoords="offset points", fontweight="bold")

    ax.text(0.97, 0.05, f"Pearson r={r:.3f}\np={p:.2e}\nn=132", transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    ax.set_xlabel("n_shared_targets (ROES, GSE133344-derived GRN)", fontsize=10)
    ax.set_ylabel("Epistasis magnitude (eps_l2)", fontsize=10)
    ax.set_title("C. GSE133344, full Norman 2019 (n=132)", fontsize=11, fontweight="bold", loc="left")


# ─────────────────────────────────────────────────────────────────
# Build individual panels + combined figure
# ─────────────────────────────────────────────────────────────────
print("Generating Figure 1A (schematic) ...")
fig, ax = plt.subplots(figsize=(8, 7))
make_panel_A(ax)
plt.tight_layout()
fig.savefig(FIG / "figure1A_roes_schematic.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("Generating Figure 1B (Norman n=12) ...")
fig, ax = plt.subplots(figsize=(6, 5.5))
make_panel_B(ax)
plt.tight_layout()
fig.savefig(FIG / "figure1B_roes_norman_n12.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("Generating Figure 1C (GSE133344 n=132) ...")
fig, ax = plt.subplots(figsize=(6, 5.5))
make_panel_C(ax)
plt.tight_layout()
fig.savefig(FIG / "figure1C_roes_gse133344_n132.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("Generating combined Figure 1 ...")
fig = plt.figure(figsize=(15, 6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1])
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])
make_panel_A(axA)
make_panel_B(axB)
make_panel_C(axC)
fig.suptitle("Figure 1. Network topology (ROES) predicts TF-pair epistasis magnitude",
              fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
fig.savefig(FIG / "Figure1_combined.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("Done. Saved to figures/figure1A_*.png, figure1B_*.png, figure1C_*.png, Figure1_combined.png")

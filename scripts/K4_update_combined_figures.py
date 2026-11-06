"""
K4: Rebuild Figure2, Figure3, Figure4 combined panels with new narrative structure.

New Figure 2: M1 schematic (Panel A, full-width top) + existing hub-TF mechanism panels (B/C/D)
New Figure 3: CDKN1A/MAPK1 (Panel A, mechanistic) + PLK4/STIL etc (Panel B) + null overview (Panel C)
New Figure 4: L3 4-evidence convergence (Panel A) + network (Panel B) + L4 family (Panel C) + genome-wide (Panel D)

Run: python scripts/K4_update_combined_figures.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from pathlib import Path

BASE = Path(__file__).parent.parent
FIG = BASE / "figures"
plt.rcParams.update({"font.size": 11})

def load(name):
    return mpimg.imread(FIG / name)

def show(ax, img, label="", label_size=13):
    ax.imshow(img)
    ax.axis("off")
    if label:
        ax.text(0.01, 0.99, label, transform=ax.transAxes,
                fontsize=label_size, fontweight="bold", va="top", ha="left",
                color="black")

# ─────────────────────────────────────────────────────────────────
# Figure 2 — Hub TF Effect (4 panels: M1 schematic + 3 mechanism panels)
# Layout: Row 1 = M1 schematic (full width, Panel A)
#         Row 2 = 3 existing mechanism panels side-by-side (Panels B/C/D)
# ─────────────────────────────────────────────────────────────────
print("Building new Figure 2 ...")
img_M1  = load("figure_M1_hub_tf_schematic.png")
img_mech = load("figure2_hub_tf_mechanism.png")   # already a 3-panel combined

fig = plt.figure(figsize=(14, 11))
gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.05, 1], hspace=0.04)

ax_A = fig.add_subplot(gs[0])
ax_B = fig.add_subplot(gs[1])

show(ax_A, img_M1,  "A")
show(ax_B, img_mech, "B")

fig.suptitle("Figure 2. The hub TF effect: regulatory reach drives epistasis strength",
             fontsize=13, fontweight="bold", y=0.995)
out2 = FIG / "Figure2_combined.png"
fig.savefig(out2, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out2.name}  ({out2.stat().st_size//1024} KB)")

# ─────────────────────────────────────────────────────────────────
# Figure 3 — Cross-modality validation (reordered for narrative)
# NEW order: A = CDKN1A/MAPK1 antagonism (mechanism)
#            B = PLK4/STIL + KIF18B/KIF2C (positive modules)
#            C = DepMap null overview (boundary/context, smaller)
# Layout: Row 1 = A (left) + B (right)
#         Row 2 = C (full width, 60% height — contextual)
# ─────────────────────────────────────────────────────────────────
print("Building new Figure 3 ...")
img_mech3 = load("figure3B_cdkn1a_mapk1_antagonism.png")   # → new Panel A
img_mod   = load("figure3C_plk4stil_kif18bkif2c.png")       # → new Panel B
img_null  = load("figure3A_depmap_validation_n121.png")      # → new Panel C

fig = plt.figure(figsize=(14, 13))
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.1, 0.85],
                       hspace=0.06, wspace=0.04)

ax_A = fig.add_subplot(gs[0, 0])
ax_B = fig.add_subplot(gs[0, 1])
ax_C = fig.add_subplot(gs[1, :])

show(ax_A, img_mech3, "A")
show(ax_B, img_mod,   "B")
show(ax_C, img_null,  "C")

fig.suptitle("Figure 3. Cross-modality validation: regulatory direction and functional co-dependence",
             fontsize=13, fontweight="bold", y=0.998)
out3 = FIG / "Figure3_combined.png"
fig.savefig(out3, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out3.name}  ({out3.stat().st_size//1024} KB)")

# ─────────────────────────────────────────────────────────────────
# Figure 4 — Translational case study (expanded with L3 + L4)
# Layout: Row 1 = L3 evidence convergence (Panel A, full width)
#         Row 2 = network (Panel B, left) + L4 family analysis (Panel C, right)
#         Row 3 = genome-wide GRN (Panel D, full width, smaller)
# ─────────────────────────────────────────────────────────────────
print("Building new Figure 4 ...")
img_L3     = load("figure_L3_cebpa_spi1_evidence_convergence.png")  # Panel A
img_net    = load("figure4A_cebpa_spi1_network.png")                  # Panel B
img_family = load("figure_L4_family_epistasis_type.png")              # Panel C
img_gwps   = load(str(Path("C_gwps") / "grn_expansion_comparison.png")
                  .replace("\\", "/"))                                 # Panel D

fig = plt.figure(figsize=(14, 18))
gs = gridspec.GridSpec(3, 2, figure=fig,
                       height_ratios=[0.9, 1.1, 0.75],
                       hspace=0.06, wspace=0.04)

ax_A  = fig.add_subplot(gs[0, :])   # L3 full width
ax_B  = fig.add_subplot(gs[1, 0])   # network
ax_C  = fig.add_subplot(gs[1, 1])   # family
ax_D  = fig.add_subplot(gs[2, :])   # genome-wide

show(ax_A, img_L3,     "A")
show(ax_B, img_net,    "B")
show(ax_C, img_family, "C")
show(ax_D, img_gwps,   "D")

fig.suptitle("Figure 4. Translational case study: CEBPA–SPI1 as a convergent synthetic-vulnerability candidate in AML",
             fontsize=13, fontweight="bold", y=0.999)
out4 = FIG / "Figure4_combined.png"
fig.savefig(out4, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out4.name}  ({out4.stat().st_size//1024} KB)")

print("\nAll combined figures updated.")

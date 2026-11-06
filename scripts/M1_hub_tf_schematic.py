import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import numpy as np
import os

# ── Output path ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(BASE, "figures", "figure_M1_hub_tf_schematic.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ── Colors ───────────────────────────────────────────────────────────────────
C_TF     = "#d62728"
C_AONLY  = "#aec7e8"
C_BONLY  = "#ffbb78"
C_SHARED = "#9467bd"
C_EDGE   = "#888888"
C_BG_L   = "#f8f8f8"
C_BG_R   = "#fffaed"

fig, axes = plt.subplots(1, 2, figsize=(12, 7))
fig.patch.set_facecolor("white")

# ─────────────────────────────────────────────────────────────────────────────
def draw_panel(ax, title, tf_a_pos, tf_b_pos,
               aonly_positions, bonly_positions, shared_positions,
               stats_text, bg_color, node_r_tf=0.07, node_r_gene=0.035,
               strong=False):
    """Draw one panel (non-hub or hub)."""

    ax.set_facecolor(bg_color)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── TF node A ────────────────────────────────────────────────────────────
    for pos, label in [(tf_a_pos, "TF-A"), (tf_b_pos, "TF-B")]:
        circ = Circle(pos, node_r_tf, color=C_TF, zorder=5, linewidth=1.5,
                      ec="white")
        ax.add_patch(circ)
        ax.text(pos[0], pos[1], label, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white', zorder=6)

    # ── Gene nodes ───────────────────────────────────────────────────────────
    def add_genes(positions, color, tf_sources):
        for gp in positions:
            c = Circle(gp, node_r_gene, color=color, zorder=4,
                       ec='white', linewidth=0.8)
            ax.add_patch(c)
            for src in tf_sources:
                ax.annotate("",
                    xy=gp, xytext=src,
                    arrowprops=dict(arrowstyle="-",
                                   color=C_EDGE, alpha=0.45, lw=0.9),
                    zorder=2)

    add_genes(aonly_positions,  C_AONLY,  [tf_a_pos])
    add_genes(bonly_positions,  C_BONLY,  [tf_b_pos])
    add_genes(shared_positions, C_SHARED, [tf_a_pos, tf_b_pos])

    # ── Stats annotation box ─────────────────────────────────────────────────
    ep_color = "#c0392b" if strong else "#27ae60"
    ep_label = "Epistasis: STRONG" if strong else "Epistasis: WEAK"

    box_y = 0.06
    ax.text(0.5, box_y + 0.105, stats_text,
            ha='center', va='bottom', fontsize=8.5,
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', fc='white',
                      ec='#cccccc', lw=1.0),
            transform=ax.transAxes, zorder=7)

    ax.text(0.5, box_y - 0.01, ep_label,
            ha='center', va='bottom', fontsize=9.5, fontweight='bold',
            color=ep_color,
            transform=ax.transAxes, zorder=7)

    # ── Column title ─────────────────────────────────────────────────────────
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8, color='#333333')

    # ── Shared-region bracket label ───────────────────────────────────────────
    if shared_positions:
        xs = [p[0] for p in shared_positions]
        ys = [p[1] for p in shared_positions]
        cx = np.mean(xs)
        cy = np.mean(ys) - 0.06
        ax.text(cx, cy, "Shared\ntargets", ha='center', va='top',
                fontsize=7.5, color=C_SHARED, fontstyle='italic',
                bbox=dict(fc='white', ec=C_SHARED, lw=0.7,
                          boxstyle='round,pad=0.2'), zorder=8)


# ─────────────────────────────────────────────────────────────────────────────
#  NON-HUB PANEL  (left)
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[0]

tf_a = (0.30, 0.84)
tf_b = (0.70, 0.84)

# 4 A-only genes — left arc
aonly = [(0.12 + i*0.075, 0.60 - i*0.05) for i in range(4)]
# 4 B-only genes — right arc
bonly = [(0.88 - i*0.075, 0.60 - i*0.05) for i in range(4)]
# 3 shared genes — center column
shared = [(0.50, 0.65), (0.50, 0.55), (0.50, 0.45)]

stats = ("total_A  = 20 genes\n"
         "total_B  = 20 genes\n"
         "n_shared =  5\n"
         "union    = 35\n"
         "Jaccard  = 5/35 = 0.14")

draw_panel(ax,
           title="Non-Hub TF Pair",
           tf_a_pos=tf_a, tf_b_pos=tf_b,
           aonly_positions=aonly,
           bonly_positions=bonly,
           shared_positions=shared,
           stats_text=stats,
           bg_color=C_BG_L,
           strong=False)

# ─────────────────────────────────────────────────────────────────────────────
#  HUB PANEL  (right)
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[1]

tf_a = (0.30, 0.84)
tf_b = (0.70, 0.84)

# 8 A-only — spread left
aonly_h = []
for i in range(4):
    aonly_h.append((0.07 + i*0.055, 0.63 - i*0.04))
for i in range(4):
    aonly_h.append((0.07 + i*0.055, 0.44 - i*0.03))

# 8 B-only — spread right
bonly_h = []
for i in range(4):
    bonly_h.append((0.93 - i*0.055, 0.63 - i*0.04))
for i in range(4):
    bonly_h.append((0.93 - i*0.055, 0.44 - i*0.03))

# 6 shared — center, denser
shared_h = [
    (0.45, 0.70), (0.50, 0.70), (0.55, 0.70),
    (0.45, 0.58), (0.50, 0.58), (0.55, 0.58),
    (0.45, 0.46), (0.50, 0.46), (0.55, 0.46),
    (0.45, 0.34), (0.50, 0.34), (0.55, 0.34),
]

stats_h = ("total_A  = 400 genes\n"
           "total_B  = 400 genes\n"
           "n_shared = 100\n"
           "union    = 700\n"
           "Jaccard  = 100/700 = 0.14")

draw_panel(ax,
           title="Hub TF Pair",
           tf_a_pos=tf_a, tf_b_pos=tf_b,
           aonly_positions=aonly_h,
           bonly_positions=bonly_h,
           shared_positions=shared_h,
           stats_text=stats_h,
           bg_color=C_BG_R,
           node_r_tf=0.07, node_r_gene=0.025,
           strong=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Legend
# ─────────────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=C_TF,     label='TF (transcription factor)'),
    mpatches.Patch(color=C_AONLY,  label='TF-A-only targets'),
    mpatches.Patch(color=C_BONLY,  label='TF-B-only targets'),
    mpatches.Patch(color=C_SHARED, label='Shared targets (epistasis loci)'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=4,
           fontsize=8.5, frameon=True, framealpha=0.9,
           bbox_to_anchor=(0.5, 0.00))

# ─────────────────────────────────────────────────────────────────────────────
#  Global title + key-insight annotation
# ─────────────────────────────────────────────────────────────────────────────
fig.suptitle(
    "Hub TF Effect: Absolute Regulatory Overlap Drives Epistasis Strength",
    fontsize=13, fontweight='bold', y=0.99, color='#222222')

# Key-insight box spanning both panels
fig.text(
    0.5, 0.915,
    "Same Jaccard (0.14) — yet Hub pair has 20× more shared targets → far stronger non-additive effects",
    ha='center', va='center', fontsize=9.5, color='#555555',
    style='italic',
    bbox=dict(boxstyle='round,pad=0.35', fc='#eaf4fb', ec='#3498db', lw=1.2))

# Arrow between panels hinting at the comparison
fig.text(0.5, 0.50, "⟵  same Jaccard  ⟶",
         ha='center', va='center', fontsize=9, color='#888888',
         rotation=90)

plt.tight_layout(rect=[0, 0.07, 1, 0.91])
fig.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUT}")

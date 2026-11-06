import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import chi2_contingency, mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

# ─── Paths ─────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT_ANNOT   = os.path.join(BASE, "data", "external", "tf_family_annotations.csv")
ROES_CSV    = os.path.join(BASE, "results", "G_grn_roes_validation", "roes_GSE133344_132pairs.csv")
EPS_CSV     = os.path.join(BASE, "results", "H_recalibrated_eps_types", "epistasis_groundtruth_recalibrated.csv")
OUT_DIR     = os.path.join(BASE, "results", "L4_tf_family_analysis")
OUT_PAIRS   = os.path.join(OUT_DIR, "pairs_with_family_annotation.csv")
OUT_STATS   = os.path.join(OUT_DIR, "family_vs_epistasis_stats.txt")
FIG_OUT     = os.path.join(BASE, "figures", "figure_L4_family_epistasis_type.png")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — TF family annotations
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Obtaining TF family annotations")
print("=" * 60)

annot_source = None

# --- Approach A: Lambert et al. 2018 supplementary table --------------------
try:
    import urllib.request, tempfile
    url = ("https://www.cell.com/cms/10.1016/j.cell.2018.01.029/attachment/"
           "ede37821-fd6f-41b7-9a0e-9d5410855ae8/mmc2.xlsx")
    print(f"Approach A: Downloading Lambert 2018 mmc2.xlsx from Cell ...")
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name
    urllib.request.urlretrieve(url, tmp_path)
    lambert = pd.read_excel(tmp_path, header=0)
    print(f"  Downloaded. Columns: {lambert.columns.tolist()[:10]}")
    # Find HGNC symbol column and DBD column
    sym_col = None
    dbd_col = None
    for c in lambert.columns:
        cl = str(c).lower()
        if "hgnc" in cl or "symbol" in cl or "gene name" in cl or "name" in cl:
            sym_col = c
        if "dbd" in cl or "family" in cl or "domain" in cl:
            dbd_col = c
    if sym_col is None:
        sym_col = lambert.columns[1]   # fallback: second column often has gene names
    if dbd_col is None:
        # look for column index with "DBD" in header or use column named "DBD"
        for c in lambert.columns:
            if str(c).strip().upper() == "DBD":
                dbd_col = c
                break
        if dbd_col is None:
            dbd_col = lambert.columns[3]  # common position
    print(f"  Using symbol col='{sym_col}', family col='{dbd_col}'")
    lambert_annot = lambert[[sym_col, dbd_col]].dropna()
    lambert_annot.columns = ["gene_symbol", "family"]
    lambert_annot["gene_symbol"] = lambert_annot["gene_symbol"].astype(str).str.strip().str.upper()
    lambert_annot["family"] = lambert_annot["family"].astype(str).str.strip()
    lambert_annot["source"] = "Lambert2018"
    annot_df = lambert_annot
    annot_source = "Lambert2018"
    print(f"  Lambert annotation loaded: {len(annot_df)} TFs")
    os.unlink(tmp_path)
except Exception as e:
    print(f"  Approach A failed: {e}")

# --- Approach B: HGNC REST API ----------------------------------------------
if annot_source is None:
    try:
        import urllib.request, json
        print("Approach B: HGNC REST API ...")
        url = "https://rest.genenames.org/fetch/gene_group/transcription%20factors"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode())
        docs = data.get("response", {}).get("docs", [])
        rows = []
        for d in docs:
            sym = d.get("symbol", "")
            grp = d.get("gene_group", ["Unknown"])
            if isinstance(grp, list):
                grp = grp[0]
            rows.append({"gene_symbol": sym.upper(), "family": grp, "source": "HGNC"})
        annot_df = pd.DataFrame(rows)
        annot_source = "HGNC"
        print(f"  HGNC annotation loaded: {len(annot_df)} TFs")
    except Exception as e:
        print(f"  Approach B failed: {e}")

# --- Approach C: Manual assignment from gene name patterns ------------------
if annot_source is None:
    print("Approach C: Manual family assignment from gene name patterns ...")
    # Read our TF list
    roes_tmp = pd.read_csv(ROES_CSV)
    all_tfs_tmp = sorted(set(roes_tmp["TF_A"].tolist() + roes_tmp["TF_B"].tolist()))

    def manual_family(gene):
        g = gene.upper()
        if g.startswith("CEBP"):
            return "CEBP"
        if g.startswith("CDKN"):
            return "CDKN"
        if g.startswith("KLF"):
            return "KLF"
        if g.startswith("FOX"):
            return "FOX"
        if g in ("JUN", "JUNB", "JUND"):
            return "JUN"
        if g in ("FOS", "FOSB", "FOSL1", "FOSL2"):
            return "FOS"
        if g in ("ETS1", "ETS2", "SPI1", "FEV"):
            return "ETS"
        if g.startswith("KIF"):
            return "KIF"
        if g.startswith("PLK"):
            return "PLK"
        if g.startswith("ZNF"):
            return "ZNF"
        if g.startswith("TBX"):
            return "TBX"
        if g.startswith("HOX"):
            return "HOX"
        if g.startswith("MAP2K"):
            return "MAP2K"
        if g.startswith("MAPK"):
            return "MAPK"
        if g.startswith("PTPN"):
            return "PTPN"
        if g.startswith("UBASH"):
            return "UBASH"
        if g.startswith("ZBTB"):
            return "ZBTB"
        if g.startswith("MEIS") or g.startswith("HOXB") or g.startswith("HOXC"):
            return "HOX"
        if g.startswith("IRF"):
            return "IRF"
        if g.startswith("LHX") or g.startswith("ISL"):
            return "LIM_HD"
        if g.startswith("POU"):
            return "POU"
        if g.startswith("RUNX") or g.startswith("CBFA"):
            return "RUNX"
        if g.startswith("MAP7"):
            return "MAP7"
        return g  # singleton

    rows = [{"gene_symbol": tf.upper(), "family": manual_family(tf), "source": "manual"}
            for tf in all_tfs_tmp]
    annot_df = pd.DataFrame(rows)
    annot_source = "manual"
    print(f"  Manual annotation created: {len(annot_df)} TFs")

# Deduplicate (keep first hit per gene_symbol)
annot_df = annot_df.drop_duplicates(subset="gene_symbol", keep="first")
annot_df.to_csv(EXT_ANNOT, index=False)
print(f"\nAnnotation saved to: {EXT_ANNOT}")
print(f"Source: {annot_source}, total unique genes annotated: {len(annot_df)}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Load our TFs and map to families
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Mapping our TFs to families")
print("=" * 60)

roes_df = pd.read_csv(ROES_CSV)
all_tfs = sorted(set(roes_df["TF_A"].tolist() + roes_df["TF_B"].tolist()))
print(f"Unique TFs in dataset: {len(all_tfs)}")

# Build lookup: gene_symbol (upper) -> family
family_lookup = dict(zip(annot_df["gene_symbol"].str.upper(), annot_df["family"]))

# Map each TF
tf_family = {}
for tf in all_tfs:
    fam = family_lookup.get(tf.upper(), None)
    if fam is None:
        # Apply manual fallback inline
        g = tf.upper()
        if g.startswith("CEBP"):     fam = "CEBP"
        elif g.startswith("CDKN"):   fam = "CDKN"
        elif g.startswith("KLF"):    fam = "KLF"
        elif g.startswith("FOX"):    fam = "FOX"
        elif g in ("JUN","JUNB","JUND"): fam = "JUN"
        elif g in ("FOS","FOSB","FOSL1","FOSL2"): fam = "FOS"
        elif g in ("ETS1","ETS2","SPI1","FEV"): fam = "ETS"
        elif g.startswith("KIF"):    fam = "KIF"
        elif g.startswith("PLK"):    fam = "PLK"
        elif g.startswith("ZNF"):    fam = "ZNF"
        elif g.startswith("TBX"):    fam = "TBX"
        elif g.startswith("HOX") or g.startswith("HOXB") or g.startswith("HOXC"): fam = "HOX"
        elif g.startswith("MAP2K"):  fam = "MAP2K"
        elif g.startswith("MAPK"):   fam = "MAPK"
        elif g.startswith("PTPN"):   fam = "PTPN"
        elif g.startswith("UBASH"):  fam = "UBASH"
        elif g.startswith("ZBTB"):   fam = "ZBTB"
        elif g.startswith("IRF"):    fam = "IRF"
        elif g.startswith("LHX") or g.startswith("ISL"): fam = "LIM_HD"
        elif g.startswith("POU"):    fam = "POU"
        elif g.startswith("RUNX") or g.startswith("CBFA"): fam = "RUNX"
        elif g.startswith("MEIS"):   fam = "HOX"
        else:                        fam = tf   # singleton
    tf_family[tf] = fam

# Count families with >1 member
from collections import Counter
fam_counts = Counter(tf_family.values())
multi_member_families = {f for f, c in fam_counts.items() if c > 1}
tfs_in_multi = [tf for tf in all_tfs if tf_family[tf] in multi_member_families]
print(f"TFs mapped to families with >1 member: {len(tfs_in_multi)} / {len(all_tfs)}")
print(f"Multi-member families: {sorted(multi_member_families)}")
for fam in sorted(multi_member_families):
    members = [tf for tf in all_tfs if tf_family[tf] == fam]
    print(f"  {fam}: {members}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Classify pairs as within-family or cross-family
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Classifying pairs")
print("=" * 60)

eps_df = pd.read_csv(EPS_CSV)

# Merge ROES features
merged = eps_df.merge(
    roes_df[["TF_A", "TF_B", "n_targets_A", "n_targets_B", "n_shared_targets",
              "jaccard_overlap", "parallel_fraction", "antagonism_score",
              "serial_score", "synth_score"]],
    on=["TF_A", "TF_B"],
    how="left"
)

def classify_pair(row):
    fam_a = tf_family.get(row["TF_A"], row["TF_A"])
    fam_b = tf_family.get(row["TF_B"], row["TF_B"])
    if fam_a == fam_b and fam_a not in ("Unknown",) and fam_a != row["TF_A"]:
        # Make sure both are in a family with >1 member
        return "within_family"
    return "cross_family"

merged["family_A"] = merged["TF_A"].map(tf_family)
merged["family_B"] = merged["TF_B"].map(tf_family)
merged["pair_type"] = merged.apply(classify_pair, axis=1)

merged.to_csv(OUT_PAIRS, index=False)
print(f"Merged pairs saved to: {OUT_PAIRS}")
print(f"Total pairs: {len(merged)}")
print(f"Within-family pairs: {(merged['pair_type']=='within_family').sum()}")
print(f"Cross-family pairs:  {(merged['pair_type']=='cross_family').sum()}")

within_df = merged[merged["pair_type"] == "within_family"]
cross_df  = merged[merged["pair_type"] == "cross_family"]

print("\nWithin-family pairs:")
for _, r in within_df.iterrows():
    print(f"  {r['TF_A']} ({r['family_A']}) x {r['TF_B']} ({r['family_B']})  ->  {r['new_eps_type']}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Statistical analysis
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Statistical analysis")
print("=" * 60)

stats_lines = []
stats_lines.append("TF Family vs Epistasis Type — Statistical Analysis")
stats_lines.append("=" * 60)
stats_lines.append(f"Annotation source: {annot_source}")
stats_lines.append(f"Total pairs: {len(merged)}")
stats_lines.append(f"Within-family: {len(within_df)}")
stats_lines.append(f"Cross-family:  {len(cross_df)}")
stats_lines.append("")

# 4a — Contingency table + chi-square
eps_types = sorted(merged["new_eps_type"].unique())
stats_lines.append("=== Epistasis Type Distribution ===")
ctable = pd.crosstab(merged["pair_type"], merged["new_eps_type"])
stats_lines.append(str(ctable))
stats_lines.append("")

# Proportions
for pt in ["within_family", "cross_family"]:
    sub = merged[merged["pair_type"] == pt]["new_eps_type"].value_counts(normalize=True)
    stats_lines.append(f"  {pt} proportions:")
    for et, p in sub.items():
        n = (merged[merged["pair_type"] == pt]["new_eps_type"] == et).sum()
        stats_lines.append(f"    {et}: {p:.3f} (n={n})")
stats_lines.append("")

# Chi-square test
try:
    chi2, p_chi2, dof, exp = chi2_contingency(ctable)
    stats_lines.append(f"Chi-square test: chi2={chi2:.4f}, df={dof}, p={p_chi2:.4f}")
    print(f"Chi-square: chi2={chi2:.4f}, p={p_chi2:.4f}")
except Exception as e:
    p_chi2 = np.nan
    stats_lines.append(f"Chi-square test failed: {e}")
    print(f"Chi-square failed: {e}")
stats_lines.append("")

# 4b — Magnitude ratio: Mann-Whitney U
within_mr = within_df["magnitude_ratio"].dropna()
cross_mr  = cross_df["magnitude_ratio"].dropna()
try:
    stat_mr, p_mr = mannwhitneyu(within_mr, cross_mr, alternative="two-sided")
    stats_lines.append(f"=== Magnitude Ratio (Mann-Whitney U) ===")
    stats_lines.append(f"  Within-family: median={within_mr.median():.4f}, n={len(within_mr)}")
    stats_lines.append(f"  Cross-family:  median={cross_mr.median():.4f}, n={len(cross_mr)}")
    stats_lines.append(f"  U={stat_mr:.1f}, p={p_mr:.4f}")
    print(f"Magnitude ratio MWU: U={stat_mr:.1f}, p={p_mr:.4f}")
except Exception as e:
    p_mr = np.nan
    stats_lines.append(f"Magnitude ratio MWU failed: {e}")
stats_lines.append("")

# 4c — eps_l2: Mann-Whitney U
within_l2 = within_df["eps_l2"].dropna()
cross_l2  = cross_df["eps_l2"].dropna()
try:
    stat_l2, p_l2 = mannwhitneyu(within_l2, cross_l2, alternative="two-sided")
    stats_lines.append(f"=== eps_l2 (Mann-Whitney U) ===")
    stats_lines.append(f"  Within-family: median={within_l2.median():.4f}, n={len(within_l2)}")
    stats_lines.append(f"  Cross-family:  median={cross_l2.median():.4f}, n={len(cross_l2)}")
    stats_lines.append(f"  U={stat_l2:.1f}, p={p_l2:.4f}")
    print(f"eps_l2 MWU: U={stat_l2:.1f}, p={p_l2:.4f}")
except Exception as e:
    p_l2 = np.nan
    stats_lines.append(f"eps_l2 MWU failed: {e}")
stats_lines.append("")

# Buffering fraction summary
n_within = len(within_df)
n_within_buf = (within_df["new_eps_type"] == "buffering").sum()
n_cross_buf  = (cross_df["new_eps_type"] == "buffering").sum()
frac_within_buf = n_within_buf / n_within if n_within > 0 else np.nan
frac_cross_buf  = n_cross_buf / len(cross_df) if len(cross_df) > 0 else np.nan
stats_lines.append(f"=== Summary ===")
stats_lines.append(f"Within-family pairs: {n_within}, buffering fraction: {frac_within_buf:.3f} ({n_within_buf}/{n_within})")
stats_lines.append(f"Cross-family pairs: {len(cross_df)}, buffering fraction: {frac_cross_buf:.3f} ({n_cross_buf}/{len(cross_df)})")

with open(OUT_STATS, "w", encoding="utf-8") as f:
    f.write("\n".join(stats_lines))
print(f"\nStats saved to: {OUT_STATS}")
print(f"\n--- Summary ---")
print(f"Within-family pairs: {n_within}")
print(f"Within-family buffering fraction: {frac_within_buf:.3f} ({n_within_buf}/{n_within})")
print(f"Cross-family buffering fraction: {frac_cross_buf:.3f} ({n_cross_buf}/{len(cross_df)})")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — Figure
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Generating figure")
print("=" * 60)

EPS_COLORS = {
    "buffering":   "#1f77b4",
    "synergistic": "#d62728",
    "additive":    "#2ca02c",
    "complex":     "#ff7f0e",
}

fig, axes = plt.subplots(1, 2, figsize=(12, 8))
fig.suptitle("Within-family TF pairs preferentially show buffering epistasis",
             fontsize=14, fontweight="bold", y=0.98)

# ── LEFT PANEL: Stacked bar chart ────────────────────────────────────────
ax = axes[0]
groups = ["Within family", "Cross family"]
group_dfs = [within_df, cross_df]
group_ns  = [len(within_df), len(cross_df)]

# All eps types present
all_eps = list(EPS_COLORS.keys())
# Add any additional types
for et in merged["new_eps_type"].unique():
    if et not in all_eps:
        all_eps.append(et)

bar_data = {}
for et in all_eps:
    bar_data[et] = []
    for gdf in group_dfs:
        n = (gdf["new_eps_type"] == et).sum()
        total = len(gdf)
        bar_data[et].append(n / total if total > 0 else 0)

x = np.arange(len(groups))
bottoms = np.zeros(len(groups))
patches_list = []
for et in all_eps:
    vals = np.array(bar_data[et])
    color = EPS_COLORS.get(et, "#888888")
    bars = ax.bar(x, vals, bottom=bottoms, color=color, label=et, width=0.5, edgecolor="white", linewidth=0.5)
    patches_list.append(mpatches.Patch(color=color, label=et))
    bottoms += vals

ax.set_xticks(x)
ax.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(groups, group_ns)], fontsize=12)
ax.set_ylabel("Proportion", fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_title("Epistasis type distribution", fontsize=12)
ax.legend(handles=patches_list, loc="upper right", fontsize=9, title="Epistasis type")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Chi-square p-value annotation
if not np.isnan(p_chi2):
    p_txt = f"χ² p = {p_chi2:.3f}" if p_chi2 >= 0.001 else f"χ² p = {p_chi2:.2e}"
else:
    p_txt = "χ² p = N/A"
ax.text(0.5, 1.08, p_txt, ha="center", va="top", transform=ax.transAxes,
        fontsize=10, color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="gray", alpha=0.8))

# ── RIGHT PANEL: Violin + strip plot ─────────────────────────────────────
ax2 = axes[1]

all_mr_groups = [within_df["magnitude_ratio"].dropna().values,
                 cross_df["magnitude_ratio"].dropna().values]

# Violin plot
parts = ax2.violinplot(all_mr_groups, positions=[0, 1],
                        showmedians=True, showextrema=True)
for pc in parts["bodies"]:
    pc.set_facecolor("#90CAF9")
    pc.set_alpha(0.5)
parts["cmedians"].set_color("navy")
parts["cmedians"].set_linewidth(2)
for key in ["cbars", "cmins", "cmaxes"]:
    if key in parts:
        parts[key].set_color("steelblue")
        parts[key].set_linewidth(1)

# Strip plot (jittered)
np.random.seed(42)
for i, (gdf, color) in enumerate(zip(group_dfs, ["#1565C0", "#B71C1C"])):
    vals = gdf["magnitude_ratio"].dropna().values
    jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
    ax2.scatter(np.full(len(vals), i) + jitter, vals,
                alpha=0.6, s=25, color=color, zorder=3)

# Reference line at 1.0
ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, zorder=1)
ax2.text(1.05, 1.02, "ratio=1.0", transform=ax2.get_yaxis_transform(),
         fontsize=8, color="gray", va="bottom")

ax2.set_xticks([0, 1])
ax2.set_xticklabels([f"Within family\n(n={len(within_df)})",
                      f"Cross family\n(n={len(cross_df)})"], fontsize=12)
ax2.set_ylabel("Magnitude ratio", fontsize=12)
ax2.set_title("Magnitude ratio distribution", fontsize=12)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Mann-Whitney p-value annotation
if not np.isnan(p_mr):
    p_txt2 = f"MWU p = {p_mr:.3f}" if p_mr >= 0.001 else f"MWU p = {p_mr:.2e}"
else:
    p_txt2 = "MWU p = N/A"
ax2.text(0.5, 0.97, p_txt2, ha="center", va="top", transform=ax2.transAxes,
         fontsize=10, color="black",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="gray", alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(FIG_OUT, dpi=200, bbox_inches="tight")
print(f"Figure saved to: {FIG_OUT}")

# ─── Final check ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("OUTPUT FILE VERIFICATION")
print("=" * 60)
for fpath in [EXT_ANNOT, OUT_PAIRS, OUT_STATS, FIG_OUT]:
    exists = os.path.isfile(fpath)
    size = os.path.getsize(fpath) if exists else 0
    status = "OK" if exists else "MISSING"
    print(f"  [{status}] {os.path.basename(fpath)}  ({size} bytes)")

print("\nDone.")

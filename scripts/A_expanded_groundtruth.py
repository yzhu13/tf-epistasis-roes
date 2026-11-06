"""
Script A: Download full Norman 2019 dataset and expand epistasis ground truth to ~131 pairs.

Steps:
  1. Download full Norman 2019 (4.2 GB) from Figshare
  2. Explore dataset structure
  3. Compute epistasis for all pairs with direct single-KO controls
  4. Linear decomposition to recover implied single-KO effects for additional pairs
  5. Save results and figures
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import urllib.request
import sys
import time
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_RAW   = BASE_DIR / "data" / "raw" / "Norman2019_full"
RESULTS    = BASE_DIR / "results" / "A_expanded_groundtruth"
FIGURES    = BASE_DIR / "figures"

DATA_RAW.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

H5AD_PATH  = DATA_RAW / "norman2019_full.h5ad"
FIGSHARE_URL = "https://ndownloader.figshare.com/files/43390776"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Download
# ══════════════════════════════════════════════════════════════════════════════

def download_full_dataset():
    if H5AD_PATH.exists():
        size_mb = H5AD_PATH.stat().st_size / 1e6
        print(f"[Step 1] File already exists ({size_mb:.0f} MB), skipping download.")
        return

    print(f"[Step 1] Downloading Norman 2019 full dataset (~4.2 GB)...")
    print(f"  URL : {FIGSHARE_URL}")
    print(f"  Dest: {H5AD_PATH}")

    start = time.time()
    last_reported = [0]

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = downloaded / total_size * 100 if total_size > 0 else 0
        mb  = downloaded / 1e6
        if pct - last_reported[0] >= 5:
            elapsed = time.time() - start
            speed   = mb / elapsed if elapsed > 0 else 0
            print(f"  {pct:5.1f}%  {mb:6.0f} MB  {speed:5.1f} MB/s", flush=True)
            last_reported[0] = pct

    urllib.request.urlretrieve(FIGSHARE_URL, H5AD_PATH, reporthook=progress)
    elapsed = time.time() - start
    size_mb = H5AD_PATH.stat().st_size / 1e6
    print(f"  Done! {size_mb:.0f} MB in {elapsed/60:.1f} min")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Explore dataset
# ══════════════════════════════════════════════════════════════════════════════

def explore_dataset(adata):
    print("\n[Step 2] Dataset structure")
    print(f"  Shape : {adata.shape}")
    print(f"  Obs columns: {list(adata.obs.columns)}")
    if adata.var is not None:
        print(f"  Var columns: {list(adata.var.columns)}")

    # Find perturbation column
    pert_col = detect_pert_col(adata)
    print(f"  Perturbation column: '{pert_col}'")

    labels = adata.obs[pert_col].astype(str).values
    unique = np.unique(labels)
    print(f"\n  Total unique labels: {len(unique)}")
    print(f"  First 50 labels:")
    for i, lab in enumerate(unique[:50]):
        n = np.sum(labels == lab)
        print(f"    [{i:3d}] {lab!r:<50}  n={n}")

    # Classify
    singles, doubles, ctrl = classify_labels(unique)
    print(f"\n  Control labels   : {ctrl}")
    print(f"  Single-gene perts: {len(singles)}")
    print(f"  Double-gene perts: {len(doubles)}")
    if doubles:
        print(f"  Double examples  : {doubles[:5]}")
    return pert_col, singles, doubles, ctrl


def detect_pert_col(adata):
    priority = ["guide_identity", "perturbation", "condition", "gene",
                "perturbation_name", "crispr_screen_1", "guide_ids", "name",
                "gene_identity", "perturbations"]
    for col in priority:
        if col in adata.obs.columns:
            return col
    # Fallback: pick column with most unique values containing gene-like strings
    for col in adata.obs.columns:
        vals = adata.obs[col].astype(str)
        if vals.str.contains(r'[A-Z]{2,}', regex=True).mean() > 0.5:
            return col
    raise ValueError(f"Cannot find perturbation column in {list(adata.obs.columns)}")


def _is_ctrl_part(s):
    """Return True if string s is a NegCtrl or similar control token."""
    ctrl_keywords = ["negctrl", "ctrl", "control", "non-targeting",
                     "nontargeting", "scramble", "safe_harbor", "untreated",
                     "unperturbed"]
    sl = s.lower()
    return any(kw in sl for kw in ctrl_keywords)


def _parse_norman_label(lab):
    """
    Parse Norman 2019 label format: 'TF_A_TF_B__TF_A_TF_B'
    The part before '__' encodes the pair as 'GENE1_GENE2'.
    Some gene names themselves contain underscores (e.g. MAP2K6),
    so we cannot simply split on first '_'.
    Strategy: take the prefix before '__', then try splitting at each '_'
    to find two valid tokens.

    Returns (tf_a, tf_b) where neither is a ctrl token, or
            (tf_a, None) if one side is ctrl (single + ctrl design),
            (None, None) on failure.
    """
    if '__' not in lab:
        return None, None

    prefix = lab.split('__')[0]   # e.g. "AHR_KLF1" or "CEBPE_NegCtrl0"

    # Try every split position
    parts_list = prefix.split('_')
    # Enumerate all ways to split into two non-empty parts
    candidates = []
    for i in range(1, len(parts_list)):
        left  = '_'.join(parts_list[:i])
        right = '_'.join(parts_list[i:])
        candidates.append((left, right))

    # Prefer splits where neither side is ctrl
    non_ctrl = [(l, r) for l, r in candidates
                if not _is_ctrl_part(l) and not _is_ctrl_part(r)]
    if non_ctrl:
        # Take the last one — if gene names have '_', longer right part is gene
        return non_ctrl[-1]

    # One side is ctrl — treat as single+ctrl
    mixed = [(l, r) for l, r in candidates
             if _is_ctrl_part(l) != _is_ctrl_part(r)]  # XOR
    if mixed:
        l, r = mixed[0]
        if _is_ctrl_part(l):
            return r, None   # single is r
        else:
            return l, None   # single is l

    return None, None


def classify_labels(unique_labels):
    """
    Classify labels into:
      ctrl   — pure NegCtrl × NegCtrl or all-ctrl tokens
      singles — single TF × NegCtrl (one real TF, one ctrl)
      doubles — two real TFs
    """
    ctrl, singles, doubles = [], [], []

    for lab in unique_labels:
        # Standard '+' / ',' separators (non-Norman format)
        if any(sep in lab for sep in ['+', ',']):
            parts = lab.replace(',', '+').split('+')
            parts = [p.strip() for p in parts if p.strip()]
            real = [p for p in parts if not _is_ctrl_part(p)]
            if len(real) == 0:
                ctrl.append(lab)
            elif len(real) == 1:
                singles.append(lab)
            else:
                doubles.append(lab)
            continue

        # Norman '__' format
        if '__' in lab:
            tf_a, tf_b = _parse_norman_label(lab)
            if tf_a is None:
                ctrl.append(lab)   # pure ctrl
            elif tf_b is None:
                singles.append(lab)
            else:
                doubles.append(lab)
            continue

        # Plain token — ctrl or single
        if _is_ctrl_part(lab):
            ctrl.append(lab)
        else:
            singles.append(lab)

    return singles, doubles, ctrl


def parse_double_label(lab):
    """
    Return (tf_a, tf_b) from a double-perturbation label.
    Both tf_a and tf_b are non-ctrl gene names.
    """
    # Standard separators
    for sep in ['+', ',']:
        if sep in lab:
            parts = [p.strip() for p in lab.split(sep, 1)]
            real = [p for p in parts if p and not _is_ctrl_part(p)]
            if len(real) == 2:
                return real[0], real[1]

    # Norman format
    if '__' in lab:
        tf_a, tf_b = _parse_norman_label(lab)
        if tf_a and tf_b:
            return tf_a, tf_b

    return None, None


def parse_single_label(lab):
    """
    Return the single TF name from a single+ctrl label, or None.
    """
    for sep in ['+', ',']:
        if sep in lab:
            parts = [p.strip() for p in lab.split(sep)]
            real = [p for p in parts if p and not _is_ctrl_part(p)]
            if len(real) == 1:
                return real[0]

    if '__' in lab:
        tf_a, tf_b = _parse_norman_label(lab)
        if tf_a and tf_b is None:
            return tf_a

    if not _is_ctrl_part(lab):
        return lab

    return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Compute mean expression profiles & direct epistasis
# ══════════════════════════════════════════════════════════════════════════════

def get_expression_matrix(adata):
    if adata.raw is not None:
        X    = adata.raw.X
        genes = np.array(adata.raw.var_names)
    else:
        X    = adata.X
        genes = np.array(adata.var_names)
    if scipy.sparse.issparse(X):
        X = X.toarray()
    return X.astype(np.float32), genes


def compute_mean_profiles(adata, pert_col):
    print("\n[Step 3] Computing mean expression profiles...")
    X, genes = get_expression_matrix(adata)
    labels   = adata.obs[pert_col].astype(str).values
    unique   = np.unique(labels)
    profiles = {}
    for lab in unique:
        mask = labels == lab
        profiles[lab] = X[mask].mean(axis=0)
    print(f"  {len(profiles)} profiles computed, {len(genes)} genes each")
    return profiles, genes


def compute_epistasis_score(e_ctrl, e_a, e_b, e_ab):
    """Return (eps_vector, eps_l2, eps_relative, eps_type, n_epistatic_genes)."""
    delta_a  = e_a  - e_ctrl
    delta_b  = e_b  - e_ctrl
    delta_ab = e_ab - e_ctrl

    e_expected = e_ctrl + delta_a + delta_b   # additive prediction
    eps_vector = e_ab - e_expected

    eps_l2  = float(np.linalg.norm(eps_vector))
    norm_a  = float(np.linalg.norm(delta_a))
    norm_b  = float(np.linalg.norm(delta_b))
    norm_ab = float(np.linalg.norm(delta_ab))

    eps_rel = eps_l2 / (norm_a + norm_b + 1e-6)

    # Classification (Norman 2019 / Ahlmann-Eltze 2025)
    if eps_rel < 0.10:
        eps_type = "additive"
    else:
        # Project eps onto single-KO directions
        cos_a = (np.dot(eps_vector, delta_a) /
                 (eps_l2 * norm_a + 1e-10)) if norm_a > 0 else 0.0
        cos_b = (np.dot(eps_vector, delta_b) /
                 (eps_l2 * norm_b + 1e-10)) if norm_b > 0 else 0.0

        if cos_a > 0.5 or cos_b > 0.5:
            eps_type = "synergistic"
        elif cos_a < -0.3 or cos_b < -0.3:
            eps_type = "buffering"
        elif norm_ab < 0.5 * min(norm_a, norm_b) + 1e-6:
            eps_type = "masking"
        else:
            eps_type = "complex"

    n_epistatic = int(np.sum(np.abs(eps_vector) > 0.5))
    return eps_vector, eps_l2, eps_rel, eps_type, n_epistatic


def compute_direct_epistasis(profiles, genes, ctrl_labels, doubles):
    print(f"\n  Computing direct epistasis for {len(doubles)} double perts...")

    # Merge all control cells into one profile
    e_ctrl = np.mean([profiles[c] for c in ctrl_labels], axis=0)

    results = []
    skipped_no_parse = 0
    skipped_no_single = 0

    for dbl in doubles:
        tf_a, tf_b = parse_double_label(dbl)
        if tf_a is None:
            skipped_no_parse += 1
            continue

        e_a  = profiles.get(tf_a)
        e_b  = profiles.get(tf_b)
        e_ab = profiles[dbl]

        if e_a is None or e_b is None:
            skipped_no_single += 1
            continue

        eps_vec, eps_l2, eps_rel, eps_type, n_ep = compute_epistasis_score(
            e_ctrl, e_a, e_b, e_ab)

        top_idx   = np.argsort(np.abs(eps_vec))[::-1][:10]
        top_genes = ";".join(genes[top_idx])

        results.append({
            "TF_A": tf_a, "TF_B": tf_b,
            "perturbation_key": dbl,
            "eps_l2": eps_l2, "eps_relative": eps_rel,
            "eps_type": eps_type,
            "n_epistatic_genes": n_ep,
            "top_epistatic_genes": top_genes,
            "source": "direct",
        })

    print(f"  Direct pairs computed : {len(results)}")
    print(f"  Skipped (parse error) : {skipped_no_parse}")
    print(f"  Skipped (no singles)  : {skipped_no_single}")
    return pd.DataFrame(results), e_ctrl


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Linear decomposition: recover implied single-KO effects
# ══════════════════════════════════════════════════════════════════════════════

def recover_implied_singles(profiles, genes, ctrl_labels, doubles, known_singles):
    """
    For TFs with no direct single-KO control, estimate delta via:
        delta_X_implied = mean over all (A, X) pairs of (E_AX - E_ctrl - delta_A)
    where delta_A is known.

    Returns:
        implied_singles: dict {tf: implied_delta_vector}
        coverage_info:   list of dicts for the summary CSV
    """
    print("\n[Step 4] Linear decomposition to recover implied single-KO effects...")

    e_ctrl = np.mean([profiles[c] for c in ctrl_labels], axis=0)

    # Build known deltas
    known_deltas = {}
    for tf in known_singles:
        if tf in profiles:
            known_deltas[tf] = profiles[tf] - e_ctrl

    print(f"  Known single-KO TFs: {len(known_deltas)}")

    # For each double, find TFs missing a single-KO
    missing_tfs = set()
    for dbl in doubles:
        tf_a, tf_b = parse_double_label(dbl)
        if tf_a is None:
            continue
        if tf_a not in known_deltas:
            missing_tfs.add(tf_a)
        if tf_b not in known_deltas:
            missing_tfs.add(tf_b)

    print(f"  Missing single-KO TFs: {len(missing_tfs)}")

    # For each missing TF, gather implied deltas from pairs where the partner is known
    implied_singles = {}
    coverage_info   = []

    for missing_tf in sorted(missing_tfs):
        implied_deltas = []
        supporting_pairs = []

        for dbl in doubles:
            tf_a, tf_b = parse_double_label(dbl)
            if tf_a is None:
                continue

            if tf_a == missing_tf and tf_b in known_deltas:
                # delta_missing = E_AB - E_ctrl - delta_B
                implied = profiles[dbl] - e_ctrl - known_deltas[tf_b]
                implied_deltas.append(implied)
                supporting_pairs.append(dbl)

            elif tf_b == missing_tf and tf_a in known_deltas:
                # delta_missing = E_AB - E_ctrl - delta_A
                implied = profiles[dbl] - e_ctrl - known_deltas[tf_a]
                implied_deltas.append(implied)
                supporting_pairs.append(dbl)

        if len(implied_deltas) >= 1:
            avg_implied = np.mean(implied_deltas, axis=0)
            implied_singles[missing_tf] = avg_implied
            coverage_info.append({
                "tf": missing_tf,
                "n_supporting_pairs": len(implied_deltas),
                "supporting_pairs": ";".join(supporting_pairs),
                "implied_delta_l2": float(np.linalg.norm(avg_implied)),
            })
            print(f"    {missing_tf}: implied from {len(implied_deltas)} pair(s)")

    print(f"  Implied single-KO TFs recovered: {len(implied_singles)}")
    return implied_singles, coverage_info, e_ctrl


def compute_implied_epistasis(profiles, genes, doubles,
                              known_deltas, implied_singles, e_ctrl):
    """Compute epistasis for pairs where at least one TF is implied."""
    print("\n  Computing epistasis for implied pairs...")

    all_deltas = dict(known_deltas)  # copy
    all_deltas.update(implied_singles)

    results = []
    for dbl in doubles:
        tf_a, tf_b = parse_double_label(dbl)
        if tf_a is None:
            continue

        # Skip if both singles were directly observed (already computed)
        has_a_direct = tf_a in known_deltas
        has_b_direct = tf_b in known_deltas
        if has_a_direct and has_b_direct:
            continue  # already in direct results

        delta_a = all_deltas.get(tf_a)
        delta_b = all_deltas.get(tf_b)
        if delta_a is None or delta_b is None:
            continue

        e_a  = e_ctrl + delta_a
        e_b  = e_ctrl + delta_b
        e_ab = profiles[dbl]

        eps_vec, eps_l2, eps_rel, eps_type, n_ep = compute_epistasis_score(
            e_ctrl, e_a, e_b, e_ab)

        top_idx   = np.argsort(np.abs(eps_vec))[::-1][:10]
        top_genes = ";".join(genes[top_idx])

        source_tag = []
        if not has_a_direct:
            source_tag.append(f"{tf_a}=implied")
        if not has_b_direct:
            source_tag.append(f"{tf_b}=implied")

        results.append({
            "TF_A": tf_a, "TF_B": tf_b,
            "perturbation_key": dbl,
            "eps_l2": eps_l2, "eps_relative": eps_rel,
            "eps_type": eps_type,
            "n_epistatic_genes": n_ep,
            "top_epistatic_genes": top_genes,
            "source": "implied(" + ",".join(source_tag) + ")",
        })

    print(f"  Additional implied pairs: {len(results)}")
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Save outputs & figure
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(df_direct, df_implied, implied_coverage, genes):
    # Combined ground truth
    df_all = pd.concat([df_direct, df_implied], ignore_index=True)
    gt_path = RESULTS / "epistasis_groundtruth_full.csv"
    df_all.to_csv(gt_path, index=False)
    print(f"\n  Saved: {gt_path}  ({len(df_all)} rows)")

    # Implied singles coverage
    df_cov = pd.DataFrame(implied_coverage)
    cov_path = RESULTS / "epistasis_implied_singles.csv"
    df_cov.to_csv(cov_path, index=False)
    print(f"  Saved: {cov_path}  ({len(df_cov)} rows)")

    # Summary
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("EXPANDED EPISTASIS GROUND TRUTH — SUMMARY")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Direct pairs (both singles observed)  : {len(df_direct)}")
    summary_lines.append(f"Implied pairs (>=1 single inferred)   : {len(df_implied)}")
    summary_lines.append(f"Total pairs                           : {len(df_all)}")
    summary_lines.append("")
    summary_lines.append("Epistasis type distribution (ALL pairs):")
    if len(df_all) > 0 and "eps_type" in df_all.columns:
        for t, n in df_all["eps_type"].value_counts().items():
            summary_lines.append(f"  {t:<15}: {n}")
    summary_lines.append("")
    summary_lines.append("Epistasis type distribution (direct only):")
    if len(df_direct) > 0 and "eps_type" in df_direct.columns:
        for t, n in df_direct["eps_type"].value_counts().items():
            summary_lines.append(f"  {t:<15}: {n}")
    summary_lines.append("")
    summary_lines.append(f"Implied single-KO TFs recovered: {len(implied_coverage)}")
    if len(df_all) > 0 and "eps_l2" in df_all.columns:
        summary_lines.append(f"eps_l2 mean  : {df_all['eps_l2'].mean():.4f}")
        summary_lines.append(f"eps_l2 median: {df_all['eps_l2'].median():.4f}")
        summary_lines.append(f"eps_l2 max   : {df_all['eps_l2'].max():.4f}")

    summary_text = "\n".join(summary_lines)
    sum_path = RESULTS / "summary.txt"
    sum_path.write_text(summary_text)
    print(f"  Saved: {sum_path}")
    print("\n" + summary_text)

    return df_all


def make_figure(df_all):
    if len(df_all) == 0:
        print("  No data to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. eps_l2 distribution by source
    ax = axes[0]
    sources = df_all["source"].unique()
    for src in sources:
        sub = df_all[df_all["source"] == src]["eps_l2"]
        ax.hist(sub, bins=30, alpha=0.6, label=src)
    ax.set_xlabel("eps_l2 (epistasis magnitude)")
    ax.set_ylabel("Count")
    ax.set_title("Epistasis Magnitude Distribution")
    ax.legend(fontsize=7)

    # 2. eps_type distribution
    ax = axes[1]
    type_counts = df_all["eps_type"].value_counts()
    colors = {"additive": "#4CAF50", "synergistic": "#F44336",
              "buffering": "#2196F3", "masking": "#FF9800", "complex": "#9C27B0"}
    bar_colors = [colors.get(t, "gray") for t in type_counts.index]
    ax.bar(type_counts.index, type_counts.values, color=bar_colors)
    ax.set_xlabel("Epistasis Type")
    ax.set_ylabel("Count")
    ax.set_title("Epistasis Type Distribution")
    ax.tick_params(axis='x', rotation=30)

    # 3. eps_relative scatter
    ax = axes[2]
    type_order = ["additive", "buffering", "synergistic", "masking", "complex"]
    palette = {t: colors.get(t, "gray") for t in type_order}
    for etype, grp in df_all.groupby("eps_type"):
        ax.scatter(grp["eps_l2"], grp["eps_relative"],
                   label=etype, alpha=0.7, s=20,
                   color=colors.get(etype, "gray"))
    ax.set_xlabel("eps_l2")
    ax.set_ylabel("eps_relative")
    ax.set_title("eps_l2 vs eps_relative")
    ax.legend(fontsize=7)

    plt.tight_layout()
    fig_path = FIGURES / "A_expanded_groundtruth_distribution.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved figure: {fig_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("A_expanded_groundtruth.py — Norman 2019 Full Dataset Analysis")
    print("=" * 60)

    # Step 1
    download_full_dataset()

    # Load
    print(f"\nLoading {H5AD_PATH} ...")
    adata = sc.read_h5ad(H5AD_PATH)
    print(f"  Loaded: {adata.shape}")

    # Normalize (log-normalize counts)
    X_check = adata.X
    if scipy.sparse.issparse(X_check):
        max_val = X_check.data.max() if X_check.nnz > 0 else 0
    else:
        max_val = X_check.max()
    if max_val > 100:
        print("  Normalizing (log1p)...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # Step 2
    pert_col, singles_labels, doubles_labels, ctrl_labels = explore_dataset(adata)

    if not ctrl_labels:
        raise ValueError("No control labels found! Check label classification.")
    if not doubles_labels:
        raise ValueError("No double perturbation labels found!")

    # Step 3 — compute mean profiles per label
    profiles, genes = compute_mean_profiles(adata, pert_col)

    # Build single-KO profiles: map TF_name -> mean expression
    # singles_labels are single+ctrl labels like 'CEBPA_NegCtrl0__CEBPA_NegCtrl0'
    e_ctrl = np.mean([profiles[c] for c in ctrl_labels], axis=0)

    single_tf_profiles = {}  # TF_name -> mean expression vector
    for sl in singles_labels:
        tf = parse_single_label(sl)
        if tf and sl in profiles:
            # Average if multiple ctrl replicates exist for same TF
            if tf not in single_tf_profiles:
                single_tf_profiles[tf] = []
            single_tf_profiles[tf].append(profiles[sl])
    # Average across replicates
    for tf in single_tf_profiles:
        single_tf_profiles[tf] = np.mean(single_tf_profiles[tf], axis=0)

    print(f"\n  Single-KO TF profiles available: {len(single_tf_profiles)}")
    print(f"  Example TFs: {list(single_tf_profiles.keys())[:10]}")

    # Inject single profiles into profiles dict so compute_direct_epistasis can look them up
    for tf, prof in single_tf_profiles.items():
        if tf not in profiles:
            profiles[tf] = prof

    df_direct, e_ctrl = compute_direct_epistasis(
        profiles, genes, ctrl_labels, doubles_labels)

    # Step 4
    known_deltas = {tf: single_tf_profiles[tf] - e_ctrl
                    for tf in single_tf_profiles}

    implied_singles, implied_coverage, _ = recover_implied_singles(
        profiles, genes, ctrl_labels, doubles_labels, set(single_tf_profiles.keys()))

    df_implied = compute_implied_epistasis(
        profiles, genes, doubles_labels,
        known_deltas, implied_singles, e_ctrl)

    # Step 5
    df_all = save_outputs(df_direct, df_implied, implied_coverage, genes)
    make_figure(df_all)

    print("\n[DONE] All outputs saved.")
    print(f"  Results dir : {RESULTS}")
    print(f"  Figures dir : {FIGURES}")


if __name__ == "__main__":
    main()

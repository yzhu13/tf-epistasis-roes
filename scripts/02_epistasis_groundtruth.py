"""
Script 02: Compute epistasis ground truth from Norman 2019 double perturbation data.

Method:
  For each double perturbation (TF_A + TF_B) in Norman 2019:
    1. Get mean expression profile: E_A (single KO A), E_B (single KO B), E_AB (double KO)
    2. Compute expected additive effect: E_expected = E_ctrl + (E_A - E_ctrl) + (E_B - E_ctrl)
    3. Epistasis vector: eps = E_AB - E_expected
    4. Epistasis score (scalar): S = ||eps||_2 with sign from direction analysis
    5. Classify: synergistic / buffering / masking (epistatic) / additive

Output:
  data/processed/epistasis_groundtruth_norman2019.csv
  Columns: TF_A, TF_B, eps_score, eps_type, n_genes_changed, top_affected_genes

References:
  Norman et al. 2019, Science - epistasis classification framework
  Ahlmann-Eltze et al. 2025, Nature Methods - epistasis metric definitions
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROC = BASE_DIR / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)

def load_norman2019():
    h5ad = DATA_RAW / "Norman2019" / "norman2019_perturbseq.h5ad"
    if not h5ad.exists():
        raise FileNotFoundError(f"Run 01_download_data.py first. Missing: {h5ad}")
    print(f"Loading Norman 2019: {h5ad}")
    adata = sc.read_h5ad(h5ad)
    print(f"  Shape: {adata.shape}")
    print(f"  Obs columns: {list(adata.obs.columns)}")
    return adata

def identify_perturbation_column(adata):
    """Find which obs column contains perturbation labels."""
    candidates = ["perturbation", "condition", "gene", "perturbation_name",
                  "crispr_screen_1", "guide_ids", "name"]
    for col in candidates:
        if col in adata.obs.columns:
            return col
    # Show all columns and let user identify
    print("Available obs columns:", list(adata.obs.columns))
    raise ValueError("Cannot identify perturbation column. Check obs columns above.")

def compute_mean_profiles(adata, pert_col):
    """Compute mean expression profile per perturbation."""
    print("Computing mean expression profiles per perturbation...")

    # Use raw counts or normalized
    if adata.raw is not None:
        X = adata.raw.X
        genes = adata.raw.var_names
    else:
        X = adata.X
        genes = adata.var_names

    # Convert to dense if sparse
    import scipy.sparse
    if scipy.sparse.issparse(X):
        X = X.toarray()

    profiles = {}
    pert_labels = adata.obs[pert_col].values
    unique_perts = np.unique(pert_labels)

    for pert in unique_perts:
        mask = pert_labels == pert
        profiles[pert] = X[mask].mean(axis=0)

    print(f"  {len(profiles)} unique perturbations found")
    return profiles, genes

def identify_control_key(profiles):
    """Find control perturbation key."""
    control_keys = ["control", "ctrl", "CTRL", "non-targeting", "NT",
                    "scramble", "safe_harbor", "nontargeting"]
    for key in profiles:
        for ck in control_keys:
            if ck.lower() in str(key).lower():
                print(f"  Control key: '{key}'")
                return key
    print("Available keys (first 20):", list(profiles.keys())[:20])
    raise ValueError("Cannot identify control. Check keys above.")

def identify_double_perturbations(profiles, control_key):
    """Identify which perturbations are double KOs."""
    singles = []
    doubles = []

    for key in profiles:
        if key == control_key:
            continue
        # Norman 2019 format: double perturbations have '+' or ',' separator
        key_str = str(key)
        if '+' in key_str or (key_str.count('_') > 1 and ',' in key_str):
            doubles.append(key)
        elif '+' in key_str:
            doubles.append(key)
        else:
            # Try splitting by common double-pert separators
            parts = key_str.replace('+', '|').replace(',', '|').split('|')
            if len(parts) == 2 and all(p.strip() for p in parts):
                doubles.append(key)
            else:
                singles.append(key)

    # If no doubles found, try another approach
    if len(doubles) == 0:
        print("  Trying alternative double detection...")
        for key in profiles:
            if key == control_key:
                continue
            # Check if any single pert name is a substring of this key
            key_str = str(key)
            sep_counts = key_str.count('+') + key_str.count(',')
            if sep_counts >= 1:
                doubles.append(key)
            else:
                singles.append(key)

    print(f"  Singles: {len(singles)}, Doubles: {len(doubles)}")
    return singles, doubles

def parse_double_pert(key):
    """Parse 'TF_A+TF_B' -> ('TF_A', 'TF_B')"""
    key_str = str(key)
    for sep in ['+', ',', '|']:
        if sep in key_str:
            parts = [p.strip() for p in key_str.split(sep)]
            if len(parts) == 2:
                return parts[0], parts[1]
    return None, None

def compute_epistasis_score(e_ctrl, e_a, e_b, e_ab):
    """
    Compute epistasis score for a TF pair.

    Epistasis = observed double KO - expected additive double KO
    E_expected = E_ctrl + (E_A - E_ctrl) + (E_B - E_ctrl)
               = E_A + E_B - E_ctrl

    Returns:
        eps_vector: gene-level epistasis (E_AB - E_expected)
        eps_l2: L2 norm of epistasis vector (magnitude)
        eps_type: classification of interaction type
        n_de_genes: number of significantly epistatic genes
    """
    e_expected = e_a + e_b - e_ctrl
    eps_vector = e_ab - e_expected

    eps_l2 = np.linalg.norm(eps_vector)

    # Classify interaction type using direction of deviation
    # Compare against single perturbation magnitudes
    delta_a = e_a - e_ctrl
    delta_b = e_b - e_ctrl
    delta_ab = e_ab - e_ctrl

    # Synergistic: double KO effect > sum of singles
    # Buffering: double KO effect < sum of singles (one rescues the other)
    # Masking/Epistatic: double KO looks like one of the singles

    # Norman 2019 framework: project E_AB onto E_A and E_B directions
    norm_a = np.linalg.norm(delta_a)
    norm_b = np.linalg.norm(delta_b)
    norm_ab = np.linalg.norm(delta_ab)

    eps_magnitude_relative = eps_l2 / (norm_a + norm_b + 1e-6)

    # Cosine similarity of eps with single perturbation effects
    if norm_a > 0:
        cos_eps_a = np.dot(eps_vector, delta_a) / (eps_l2 * norm_a + 1e-10)
    else:
        cos_eps_a = 0.0
    if norm_b > 0:
        cos_eps_b = np.dot(eps_vector, delta_b) / (eps_l2 * norm_b + 1e-10)
    else:
        cos_eps_b = 0.0

    # Count genes with substantial epistasis (|eps| > 0.5 log2FC-equivalent)
    n_de_genes = int(np.sum(np.abs(eps_vector) > 0.5))

    # Classification rules (from Norman 2019 supplementary)
    if eps_magnitude_relative < 0.1:
        eps_type = "additive"
    elif cos_eps_a > 0.5 or cos_eps_b > 0.5:
        eps_type = "synergistic"
    elif cos_eps_a < -0.3 or cos_eps_b < -0.3:
        eps_type = "buffering"
    elif norm_ab < 0.5 * min(norm_a, norm_b):
        eps_type = "masking"  # one masks the other
    else:
        eps_type = "complex"

    return eps_vector, eps_l2, eps_type, n_de_genes, eps_magnitude_relative

def run_epistasis_analysis():
    adata = load_norman2019()
    pert_col = identify_perturbation_column(adata)

    # Normalize if not already done
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    profiles, genes = compute_mean_profiles(adata, pert_col)
    ctrl_key = identify_control_key(profiles)
    singles, doubles = identify_double_perturbations(profiles, ctrl_key)

    e_ctrl = profiles[ctrl_key]

    results = []
    print(f"\nComputing epistasis for {len(doubles)} double perturbations...")

    skipped = 0
    for i, dbl_key in enumerate(doubles):
        tf_a, tf_b = parse_double_pert(dbl_key)
        if tf_a is None or tf_b is None:
            skipped += 1
            continue

        # Get single perturbation profiles
        e_a = profiles.get(tf_a, None)
        e_b = profiles.get(tf_b, None)
        e_ab = profiles[dbl_key]

        if e_a is None or e_b is None:
            skipped += 1
            continue

        eps_vec, eps_l2, eps_type, n_de, eps_rel = compute_epistasis_score(
            e_ctrl, e_a, e_b, e_ab
        )

        # Top affected genes (most epistatic)
        top_idx = np.argsort(np.abs(eps_vec))[::-1][:10]
        top_genes = ";".join([str(genes[i]) for i in top_idx])

        results.append({
            "TF_A": tf_a,
            "TF_B": tf_b,
            "perturbation_key": dbl_key,
            "eps_l2": eps_l2,
            "eps_relative": eps_rel,
            "eps_type": eps_type,
            "n_epistatic_genes": n_de,
            "top_epistatic_genes": top_genes,
        })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(doubles)} done")

    df = pd.DataFrame(results)
    out_path = DATA_PROC / "epistasis_groundtruth_norman2019.csv"
    df.to_csv(out_path, index=False)

    print(f"\n{'='*50}")
    print(f"Epistasis ground truth computed: {len(df)} pairs")
    print(f"Skipped (missing singles): {skipped}")
    print(f"\nEpistasis type distribution:")
    print(df["eps_type"].value_counts())
    print(f"\nSaved: {out_path}")

    # Summary stats
    print(f"\nTop 10 most epistatic pairs:")
    top10 = df.nlargest(10, "eps_l2")[["TF_A", "TF_B", "eps_l2", "eps_type"]]
    print(top10.to_string(index=False))

    return df

if __name__ == "__main__":
    df = run_epistasis_analysis()
    print("\nNext step: python scripts/03_build_grn.py")

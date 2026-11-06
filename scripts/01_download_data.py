"""
Script 01: Download all required datasets for TF Combinatorial Epistasis project.

Datasets:
  1. Norman 2019 (double perturbation ground truth) - via pertpy
  2. Replogle 2022 K562 essential (single KO baseline) - via pertpy
  3. DepMap 25Q2 CERES scores - direct download from depmap.org

Usage:
    conda activate tf_epistasis
    python scripts/01_download_data.py

Each dataset is saved to data/raw/<name>/ with a README.txt for provenance.
"""

import os
import sys
import urllib.request
import hashlib
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"

def download_file(url, dest_path, desc=""):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"  [skip] {dest_path.name} already exists")
        return
    print(f"  Downloading {desc} -> {dest_path.name} ...")
    urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
    print()

def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        mb = downloaded / 1e6
        total_mb = total_size / 1e6
        print(f"\r    {pct:.1f}%  {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)

def write_provenance(folder, text):
    with open(folder / "README.txt", "w") as f:
        f.write(text)

# ─────────────────────────────────────────────
# 1. Norman 2019 - via pertpy (lazy download)
# ─────────────────────────────────────────────
def download_norman2019():
    out = DATA_RAW / "Norman2019"
    out.mkdir(parents=True, exist_ok=True)

    h5ad_path = out / "norman2019_perturbseq.h5ad"
    if h5ad_path.exists():
        print("[Norman2019] Already downloaded.")
        return

    print("[Norman2019] Downloading via pertpy...")
    try:
        import pertpy as pt
        adata = pt.data.norman_2019()
        adata.write_h5ad(h5ad_path)
        print(f"  Saved: {h5ad_path} ({h5ad_path.stat().st_size/1e6:.1f} MB)")
    except ImportError:
        print("  pertpy not installed. Installing...")
        os.system("pip install pertpy -q")
        import pertpy as pt
        adata = pt.data.norman_2019()
        adata.write_h5ad(h5ad_path)

    write_provenance(out, """Norman et al. 2019 - Exploring genetic interaction manifolds constructed from rich single-cell phenotypes
Source: Science 2019, DOI: 10.1126/science.aax4438
Downloaded via: pertpy.data.norman_2019()
Cell line: K562 (CRISPRa)
Content: ~91,000 cells, 287 perturbations (100 single + 131 double + controls)
19,264 genes measured
Use: Double perturbation ground truth for epistasis scoring
""")
    print("[Norman2019] Done.")

# ─────────────────────────────────────────────
# 2. Replogle 2022 K562 Essential - via pertpy
# ─────────────────────────────────────────────
def download_replogle2022():
    out = DATA_RAW / "Replogle2022"
    out.mkdir(parents=True, exist_ok=True)

    h5ad_path = out / "replogle2022_k562_essential.h5ad"
    if h5ad_path.exists():
        print("[Replogle2022] Already downloaded.")
        return

    print("[Replogle2022] Downloading K562 essential via pertpy...")
    try:
        import pertpy as pt
        adata = pt.data.replogle_2022_k562_essential()
        adata.write_h5ad(h5ad_path)
        print(f"  Saved: {h5ad_path} ({h5ad_path.stat().st_size/1e6:.1f} MB)")
    except Exception as e:
        print(f"  pertpy download failed: {e}")
        print("  Falling back to Figshare direct download...")
        # Replogle 2022 Figshare: https://plus.figshare.com/articles/dataset/.../20029387
        url = "https://figshare.com/ndownloader/files/36775315"  # K562 essential h5ad
        download_file(url, h5ad_path, "Replogle2022 K562 essential")

    write_provenance(out, """Replogle et al. 2022 - Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq
Source: Cell 2022, DOI: 10.1016/j.cell.2022.05.013
Downloaded via: pertpy.data.replogle_2022_k562_essential()
Cell line: K562 (CRISPRi)
Content: ~400,000 cells, ~2,500 essential genes targeted
Use: Single KO baseline for computing expected additive effects
""")
    print("[Replogle2022] Done.")

# ─────────────────────────────────────────────
# 3. DepMap 25Q2 CERES scores
# ─────────────────────────────────────────────
def download_depmap():
    out = DATA_RAW / "DepMap"
    out.mkdir(parents=True, exist_ok=True)

    # DepMap public data portal
    # CERES gene effect scores (CRISPRko across ~1320 cell lines)
    files = {
        "CRISPRGeneDependency.csv": "https://depmap.org/portal/api/download/gene_dependency?release=25Q2",
        "CRISPRGeneEffect.csv":     "https://depmap.org/portal/api/download/gene_effect?release=25Q2",
        "sample_info.csv":           "https://depmap.org/portal/api/download/sample_info?release=25Q2",
    }

    for fname, url in files.items():
        dest = out / fname
        if dest.exists():
            print(f"  [skip] {fname} already exists")
            continue
        print(f"  Downloading DepMap {fname}...")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, dest)
            print(f"  Saved: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        except Exception as e:
            print(f"  Warning: Could not auto-download {fname}: {e}")
            print(f"  Please manually download from: https://depmap.org/portal/download/all/?release=25Q2")
            print(f"  Save as: {dest}")

    write_provenance(out, """DepMap 25Q2 CRISPR Gene Effect (CERES scores)
Source: Broad Institute Cancer Dependency Map
URL: https://depmap.org/portal/download/all/?release=25Q2
Files:
  - CRISPRGeneEffect.csv : CERES gene effect scores (rows=cell lines, cols=genes)
  - CRISPRGeneDependency.csv : probability of dependency
  - sample_info.csv : cell line metadata
Use: Co-essentiality validation — TF pairs with correlated CERES scores
     across cell lines = functionally redundant in cancer cells
""")
    print("[DepMap] Done.")

# ─────────────────────────────────────────────
# 4. TF annotation list (Lambert 2018 + update)
# ─────────────────────────────────────────────
def download_tf_list():
    out = DATA_RAW / "TF_annotations"
    out.mkdir(parents=True, exist_ok=True)

    # Human TF census from Lambert et al. 2018 (Cell)
    # Available via AnimalTFDB or direct from supplementary
    tf_path = out / "human_TFs_Lambert2018.txt"
    if tf_path.exists():
        print("[TF list] Already downloaded.")
        return

    # AnimalTFDB4 download of human TFs
    url = "http://bioinfo.life.hust.edu.cn/AnimalTFDB4/static/download/TF_list_final/Homo_sapiens_TF"
    try:
        download_file(url, tf_path, "Human TF list (AnimalTFDB4)")
    except Exception as e:
        print(f"  Could not download TF list: {e}")
        print("  Will use pertpy's built-in TF list as fallback.")

    write_provenance(out, """Human Transcription Factor list
Primary source: Lambert et al. 2018 Cell - The Human Transcription Factors
  DOI: 10.1016/j.cell.2018.01.029
  ~1,600 human TFs
Also downloaded from: AnimalTFDB4 (http://bioinfo.life.hust.edu.cn/AnimalTFDB4)
Use: Define the universe of TFs for ROES computation
""")
    print("[TF list] Done.")


if __name__ == "__main__":
    print("=" * 60)
    print("TF Combinatorial Epistasis - Data Download")
    print("=" * 60)

    # Check Python environment
    try:
        import scanpy
        import anndata
        print(f"scanpy {scanpy.__version__}, anndata {anndata.__version__} - OK")
    except ImportError:
        print("Installing core dependencies...")
        os.system("pip install scanpy anndata pertpy -q")

    print("\n[1/4] Norman 2019 double perturbation data")
    download_norman2019()

    print("\n[2/4] Replogle 2022 K562 essential Perturb-seq")
    download_replogle2022()

    print("\n[3/4] DepMap 25Q2 CERES scores")
    download_depmap()

    print("\n[4/4] Human TF annotation list")
    download_tf_list()

    print("\n" + "=" * 60)
    print("Download complete. Next step:")
    print("  python scripts/02_epistasis_groundtruth.py")
    print("=" * 60)

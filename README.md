# Shared regulatory targets predict the strength of combinatorial transcription-factor interactions

Analysis code for the manuscript (in preparation):

> Xiong Zhu, Yiteng Tang, Xin Qi, Yan Zhu. *Shared regulatory targets predict the strength of combinatorial transcription-factor interactions.*

## What this does

When two transcription factors (TFs) are perturbed together, the combined effect on gene expression is often larger (synergy) or smaller (buffering) than the sum of their single effects. This is called transcriptional epistasis. This repository asks whether a simple, interpretable quantity can predict how strong that interaction is. The quantity is the number of downstream targets two TFs share in a gene regulatory network (GRN). We call it the Regulatory Overlap Epistasis Score (ROES), and computing it needs no double-perturbation experiment.

Main results:

| Test | Data | Result |
|---|---|---|
| Original ground truth | Norman et al. 2019 Perturb-seq, 12 TF pairs | shared targets vs. epistasis magnitude, r = 0.76, p = 0.004 |
| Full release | GEO GSE133344, 111,668 cells, 132 TF pairs | r = 0.51, p = 3.3 × 10⁻¹⁰ |
| GRN robustness | network extended with Replogle et al. 2022 genome-wide Perturb-seq (K562) | ROES highly concordant with the original GRN (r = 0.98) |
| Cross-modality check | DepMap 25Q2 CRISPR co-essentiality, 1,183 cell lines, 121 pairs | no global correlation (\|r\| < 0.05); the strongest co-essential pairs lie within functional modules (PLK4/STIL r = 0.46; KIF18B/KIF2C r = 0.56) |
| Mechanism | regulatory direction of shared targets | CDKN1A/MAPK1 share targets but regulate 4 of 5 in opposite directions, consistent with their negative co-essentiality |
| Case study | GRN overlap, DepMap, STRING, OpenTargets | CEBPA–SPI1 ranks first on all four lines of evidence |

ROES is intended as a fast prior for ranking TF pairs within a pathway or process before running combinatorial screens. It complements deep perturbation models such as GEARS, which often fail to beat simple additive baselines.

## Repository layout

```
scripts/          analysis pipeline (Python)
data/processed/   small derived tables (GRN edges, ground truth, ROES ranks, DepMap results)
results/          per-analysis outputs (CSV tables and summary text)
figures/          figure panels and combined figures (see figures/FIGURES_STATUS.md)
```

Raw data are not included. `scripts/01_download_data.py` downloads them into `data/raw/`.

## Scripts

| Step | Script | Purpose |
|---|---|---|
| 1 | `01_download_data.py` | download Norman 2019, DepMap, STRING and other inputs |
| 2 | `02_epistasis_groundtruth.py` | epistasis ground truth from single and double perturbations |
| 3 | `03_build_grn.py` | TF–target gene regulatory network |
| 4 | `04_compute_roes.py` | ROES features for TF pairs |
| 5 | `05_validate_depmap.py`, `run_depmap_validation.py` | DepMap co-essentiality comparison |
| 6 | `06_disease_mapping.py`, `D_disease_mapping.py` | disease associations for TF pairs |
| A | `A_expanded_groundtruth.py` | expanded ground truth from the full Norman release |
| B | `B_mechanistic_analysis.py` | CDKN1A/MAPK1 case and leave-one-out benchmark |
| C | `C_gwps_grn_expansion.py` | GRN extension with Replogle 2022 genome-wide Perturb-seq |
| E | `E_benchmark_ppi.py` | comparison with STRING protein–protein interactions |
| F | `F_full_groundtruth_GSE133344.py` | pseudo-bulk ground truth from GSE133344 (132 pairs) |
| G | `G_grn_from_GSE133344_and_roes_validation.py` | ROES validation at n = 132 |
| H | `H_recalibrate_eps_types.py` | recalibrated synergy and buffering classes |
| I | `I_jaccard_mechanism.py` | absolute overlap vs. Jaccard and the hub-TF effect |
| J | `J_depmap_revalidation_132.py` | DepMap re-validation on the 132-pair set |
| K1–K4 | `K*_figure*_panels.py` | figure panels |
| L1–L4 | `L*_*.py` | regulatory direction, antagonism, CEBPA–SPI1 evidence, TF families |
| M1 | `M1_hub_tf_schematic.py` | schematic figure |

Run scripts from the repository root, for example `python scripts/04_compute_roes.py`. Paths are relative to the repository.

## Requirements

Python 3.10+; see `requirements.txt`.

## Data sources

- Norman et al. (2019) Perturb-seq: GEO [GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344)
- Replogle et al. (2022) genome-wide Perturb-seq (K562)
- DepMap 25Q2 CRISPR gene effect: [depmap.org](https://depmap.org/portal/download/)
- STRING v12.0: [string-db.org](https://string-db.org)
- OpenTargets TF–disease associations

## License

MIT

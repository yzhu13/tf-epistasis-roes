# Figure Status & Mapping (updated 2026-06-10)

All publication figures live in `figures/` root, named `figureNX_description.png`
(individual panels) and `FigureN_combined.png` (assembled multi-panel figure).
Generation scripts: `scripts/K1_figure1_panels.py`, `K2_figure3_panels.py`, `K3_figure4_panels.py`.

## Figure 1 — Core hypothesis: ROES predicts epistasis (3-tier validation) ✅ COMPLETE
- `figure1A_roes_schematic.png` — conceptual diagram: TF-A/TF-B target-set overlap → ROES features → predicted epistasis magnitude.
- `figure1B_roes_norman_n12.png` — Norman 2019 (n=12): n_shared_targets vs eps_l2, r=0.760, p=0.00412.
- `figure1C_roes_gse133344_n132.png` — GSE133344 (n=132): n_shared_targets vs eps_l2, r=0.513, p=3.28e-10.
- `Figure1_combined.png` — all 3 panels assembled.

## Figure 2 — "Hub TF effect" mechanism ✅ COMPLETE
- `figure2_hub_tf_mechanism.png` / `Figure2_combined.png` (same content) — 3 panels:
  total_targets vs jaccard, total_targets vs eps_l2, raw vs partial correlations
  (n_shared r: 0.51→0.18 partial; jaccard r: -0.54→-0.12 partial; joint R²=0.547).

## Figure 3 — Cross-modality validation ✅ COMPLETE
- `figure3A_depmap_validation_n121.png` — DepMap co-essentiality vs ROES, n=121, null overall (r≈-0.004 to -0.025).
- `figure3B_cdkn1a_mapk1_antagonism.png` — CDKN1A/MAPK1: 4/5 shared targets discordant direction (resolves overlap-but-negative-DepMap-r paradox).
- `figure3C_plk4stil_kif18bkif2c.png` — NEW: PLK4/STIL (depmap_r=0.455) & KIF18B/KIF2C (depmap_r=0.559) co-essentiality scatter, mitosis/centriole module positive controls.
- `Figure3_combined.png` — all 3 panels assembled (A on top row, B+C on bottom row).

## Figure 4 — Translational case study: CEBP family x SPI1 (AML) ✅ COMPLETE
- `figure4A_cebpa_spi1_network.png` — REGENERATED: CEBPA/SPI1 GRN network, structured
  layout (gene symbols mapped from GSE133344 gene index), 253 shared targets (top 10
  shown), direct PPI annotated (STRING combined_score=990).
- `figure4B_within_vs_cross_family.png` — NEW: CEBPB/CEBPA & CEBPE/CEBPA (within-family,
  buffering, eps_l2=10.18/9.02) vs CEBPE/SPI1 (cross-family, synergistic, eps_l2=6.60) —
  magnitude_ratio bar chart + n_shared/jaccard scatter.
- `figure4C` — reuses `C_gwps/grn_expansion_comparison.png` (genome-wide expanded GRN
  ROES consistency, r=0.979) as Panel C, embedded directly in `Figure4_combined.png`
  (no separate `figure4C_*.png` file — original is large/multi-panel itself).
- `Figure4_combined.png` — all 3 panels assembled.

## Supplementary
- `figure_supp_eps_type_classification.png` — recalibrated eps_type 2-axis classification (n=132, balanced 4-class).
- Other figures NOT mapped to main text (candidates for supplement): `A_expanded_groundtruth_distribution.png`,
  `depmap_validation.png` (original 17-TF/136-pair, superseded by Fig 3A),
  `B_mechanistic/B3_fig1-3*.png`, `B_mechanistic/B4_roes_v2_validation.png`,
  `D_disease/*.png`, `E_benchmark/feature_importance.png`, `E_benchmark/method_comparison.png`,
  `E_benchmark/scatter_matrix.png`.

## Status: ALL 4 MAIN FIGURES PUBLICATION-READY
Remaining optional polish (not blocking writing):
- Figure 4C could be replaced with a cleaner dedicated "top predictions" table/figure
  if reviewers find the embedded multi-panel C_gwps figure too dense.

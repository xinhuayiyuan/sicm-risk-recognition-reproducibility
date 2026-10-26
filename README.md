# Reproducibility Package

**Manuscript**: Left Ventricular Systolic Dysfunction in Sepsis without Prior Heart Failure: Early Risk Recognition and Mortality Risk

This package contains the analysis code, SQL extraction scripts, aggregated result files and figures underlying the manuscript. It is provided to editors, reviewers and readers to support evaluation and reproduction of the reported results.

## Contents

| Folder | Content |
|---|---|
| `code/` | Python and R analysis scripts (numbered roughly in execution order) |
| `sql/` | PostgreSQL scripts that build the analysis datasets from MIMIC-IV |
| `results/` | Aggregated result files (model performance, coefficients, score card, mediation, sensitivity analyses). **No record-level data.** |
| `figures/` | Figure image files as embedded in the manuscript |

## Analysis-set conventions

- Table 1 describes the 1,783-patient descriptive cohort. The new-onset cohort (n = 789; 215 SICM, 574 reference) is used for model development, mortality association and MIMIC mediation.
- The development/test partitions contain 552/150 and 237/65 patients/events, respectively. Five-fold OOF evaluation uses all 789 patients (215 events); its AUC CI is from 2,000-replicate stratified bootstrap. Subgroup recalibration uses twofold cross-fitting.
- The ICD-10-only HF-ascertainment robustness cohort contains 1,504 patients (676 SICM); its overall OOF AUC (0.738) must not be confused with the primary new-onset OOF AUC (0.677). The robustness battery used a separate sensitivity implementation (10 regularisation values, five-fold inner CV) and is not directly comparable with the primary n = 789 OOF estimate (20 values, 10-fold inner CV).
- The SHAP figure explains an exploratory XGBoost fit in all 789 new-onset patients and has no separate held-out AUC. The XGBoost AUC 0.663 belongs to the distinct 237-patient test-set comparator.
- The exploratory test-set operating point (0.297) was chosen to give sensitivity closest to 0.65, not by Youden's index. DCA is exploratory over evaluated thresholds of 5–50%.
- MIMIC mediation uses 2,000 bootstrap replicates; the eICU proxy-pathway analysis uses 300. The eICU analysis is not external validation of the imaging-defined phenotype or prediction model.

## Data access

The analyses use **MIMIC-IV v3.1**, **MIMIC-IV-Echo v1.0.1** and **eICU-CRD v2.0**. All are credentialed-access datasets available through [PhysioNet](https://physionet.org) subject to completion of the required training and data use agreements (DUAs). Echocardiographic LVEF values derive exclusively from the machine-extracted structured measurements of **MIMIC-IV-Echo** (Gow et al., 2026; doi:10.13026/307c-mr50; `structured-measurement.csv`, DICOM structured reports); no free-text report processing is involved. The DUAs prohibit redistribution of record-level data; derived stay-level extracts are therefore **not** included in this package. With credentialed access and a local PostgreSQL mirror (including the `mimiciv_derived` schema), the datasets can be rebuilt by running, in order:

0. LVEF import: filter `measurement = 'lvef'` rows from the MIMIC-IV-Echo `structured-measurement.csv` and load them into the analysis database (table `mimiciv_echo.echo_lvef_numeric`; the analysis used the 24,575 hospitalisation-linked records, reconciled 100% against the import — see `code/80_lvef_text_reconciliation.py` for the reconciliation procedure)
1. `sql/build_imaging_criteria.sql`, `sql/build_imaging_v2.sql`, `sql/build_clinical_criteria.sql` (upstream LVEF, troponin, comorbidity and vasoactive-agent tables)
2. `sql/extract_sicm_cohort_v2.sql` (analysis dataset)
3. `code/10_build_final_dataset.py`, then `code/100_strict_aki24h_derivation.py`

## Environment

- Python 3.12 with: pandas, numpy, scikit-learn 1.8, XGBoost 3.2, LightGBM 4.6, statsmodels, SHAP, psycopg2, matplotlib, patsy, scipy, Pillow
- R 4.6 with: cmprsk, survival
- PostgreSQL (local mirror of MIMIC-IV v3.1 with `mimiciv_hosp`, `mimiciv_icu`, `mimiciv_derived` schemas; eICU-CRD v2.0)
- Set the database password via the placeholder in each script (`<SET-IN-ENVIRONMENT>`) and adjust `<PROJECT_ROOT>` to the local working directory.

## Reproducing the reported results

The fixed random seed is **20260101** throughout. Key steps:

| Step | Script(s) | Principal outputs |
|---|---|---|
| Cohort build | `10_build_final_dataset.py`, `100_strict_aki24h_derivation.py` | analysis cohorts, strict 0-24 h AKI stage |
| Prediction model (test AUC, calibration, OOF, comparators, DCA) | `101_full_recompute_strict24h.py` (parts 1-2) | `v5_model_fulleval.csv`, `v5_oof.csv`, `v5_dca.csv` |
| Score card and score-to-risk mapping | `92_corrected_scorecard.py`, `101_...` (part 3), `102_score_recalibration_figures.py` | `v5_score_card_full.csv`, `v5_score_map_raw.csv` |
| Mortality association and IPW | `101_...` (part 4), `84_corrected_iptw.py` | `v5_adj_or.csv`, `v5_iptw.csv` |
| Mediation analysis | `108_mediation_boot2000.py`, `103_mediation_and_pathway_figure.py` | `v6_mediation_boot2000.csv`, `fig3_pathway.png` |
| Competing risks | `62_prepare_comprisk.py`, `62_comprisk.R`, `96_comprisk_corrected.R` | `v4_corrected_cif.csv`, `v4_corrected_finegray.csv` |
| Internal-validation battery (subgroups, eras, phenotypes, weighted AUC) | `101_...` (part 7) | `v5_subgroup_heterogeneity.csv`, `v5_temporal_era_cv.csv`, `v5_sensitivity_S1_definitions.csv`, `v5_weighted_auc.csv` |
| Landmark analysis | `101_...` (part 6), `106_landmark_and_sensitivity.py` | `v6_landmark24h_corrected.csv` |
| Eligibility sensitivity | `104_eligibility_sensitivity.py` | restriction sensitivity table (Supplement S1) |
| Coding-era sensitivity | `101_...` (part 4) | `table_S1_coding_sensitivity_en.csv` |
| eICU pathway consistency | `50_eicu_proxy.py`, `51_proxy_decompose.py` | `aim_eicu_V1_transport.csv`, `aim_eicu_V3_mediation.csv` |
| Baseline table | `82_table1.py`, `93_table1_pvalues.py`, `120_table1_extra.py` | `table1_era_spanning_p.csv`, `v6_table1_extra_rows.csv` |
| Independent aggregate checks | `121_analysis_set_metrics.py` | `v7_analysis_set_metrics.csv`, `v7_oof_fold_details.csv` |
| Figures | `103_...`, `105_shap_hires_figure.py`, `114_fig2_v6.py`, `116_fig1_v6.py`, `117_figS2_AJ.R`, `118a_figS3.py`, `118b_figS45.py`, `122_final_figure_labels.py` | `figures/` |

Earlier-numbered scripts (`06`-`98`) document the analysis lineage (model variants, leakage verification, temporal and subgroup analyses) that informed the final pre-specified pipeline.

## Notes

- Aggregated result files contain no record-level identifiers; files containing such identifiers are excluded by design and listed in the packager's exclusion log.
- `DATA_LOCK_STATEMENT.md` records the submission lock; SHA256 manifests cover code, results and figures.
- The pre-specified analysis plan and full computation logs are available from the corresponding author on reasonable request, as stated in the manuscript.
- Comments in some scripts are in Chinese; all outputs referenced by the manuscript are in English.

## License

Code is released under the MIT License (see `LICENSE`). The package contains no MIMIC-IV, MIMIC-IV-Echo or eICU-CRD record-level data; access to those datasets remains governed by the PhysioNet DUAs.

## Citation

If you use this code, please cite the manuscript (see `CITATION.cff`).

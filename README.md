# Reproducibility Package

**Manuscript**: Left Ventricular Systolic Dysfunction in Sepsis without Prior Heart Failure: Early Risk Recognition and Mortality Risk

This repository contains the analysis code, SQL extraction scripts, aggregated result files and figures underlying the manuscript. It is provided to editors, reviewers and readers to support evaluation and reproduction of the reported results.

## Data Lock

**All analyses were completed and results locked on 2026-10-20 14:01 UTC.** See `DATA_LOCK_STATEMENT.md` for database versions, code checksums (`code_checksums.txt`) and results checksums (`results_checksums.txt`).

## Repository structure

| Folder | Content |
|---|---|
| `code/` | Python and R analysis scripts (numbered roughly in execution order) |
| `sql/` | PostgreSQL scripts that build the analysis datasets from MIMIC-IV |
| `results/` | Aggregated result files (model performance, coefficients, score card, mediation, sensitivity analyses). **No record-level data.** |
| `figures/` | Figure image files as embedded in the manuscript |

## Data access

The analyses use **MIMIC-IV v3.1**, **MIMIC-IV-Echo v1.0.1** and **eICU-CRD v2.0**. All are credentialed-access datasets available through [PhysioNet](https://physionet.org) subject to completion of the required training and data use agreements (DUAs). Echocardiographic LVEF values derive from the `structured-measurement.csv` file of **MIMIC-IV-Echo** (Gow et al., 2026; doi:10.13026/307c-mr50), which contains cardiologist-recorded measurements from 206,488 echocardiogram studies in 91,372 patients; no free-text report processing is involved. The DUAs prohibit redistribution of record-level data; derived stay-level extracts are therefore **not** included in this package. With credentialed access and a local PostgreSQL mirror (including the `mimiciv_derived` schema), the datasets can be rebuilt by running, in order:

0. LVEF import: filter `measurement = 'lvef'` rows from the MIMIC-IV-Echo `structured-measurement.csv` and load them into the analysis database (table `mimiciv_echo.echo_lvef_numeric`; the analysis used the 24,575 hospitalisation-linked records, reconciled 100% against the import — see `code/80_lvef_text_reconciliation.py` for the reconciliation procedure)
1. `sql/build_imaging_criteria.sql`, `sql/build_imaging_v2.sql`, `sql/build_clinical_criteria.sql` (upstream LVEF, troponin, comorbidity and vasoactive-agent tables)
2. `sql/extract_sicm_cohort_v2.sql` (analysis dataset)
3. `code/10_build_final_dataset.py`, then `code/100_strict_aki24h_derivation.py`

## Environment

- Python 3.12 with: pandas, numpy, scikit-learn 1.8, XGBoost 3.2, LightGBM 4.6, statsmodels, SHAP, psycopg2, matplotlib, patsy, scipy (see `requirements.txt`)
- R 4.6 with: cmprsk, survival, mice
- PostgreSQL (local mirror of MIMIC-IV v3.1 with `mimiciv_hosp`, `mimiciv_icu`, `mimiciv_derived` schemas; eICU-CRD v2.0)
- Set the database password via the placeholder in each script (`<SET-IN-ENVIRONMENT>`) and adjust `<PROJECT_ROOT>` to the local working directory.

## Reproducing the reported analyses

The fixed random seed is **20260101** throughout. Key steps:

| Analysis | Script(s) | Output |
|---|---|---|
| Cohort build and primary model (Table 2), mortality association (Table 3), mediation (Table 4), landmark analysis | `code/101_full_recompute_strict24h.py` (parts 1–6) | `results/v5_*`, `results/v6_landmark24h_corrected.csv` |
| Score card and recalibration figures | `code/102_score_recalibration_figures.py` | `results/v5_score_card_full.csv`, `results/v5_recalibration.csv` |
| Mediation analysis (2,000 bootstrap replicates) | `code/103_mediation_and_pathway_figure.py`, `code/115_mediation_boot2000_fig.py` | `results/v6_mediation_boot2000.csv`, `figures/fig3_pathway.png` |
| Eligibility sensitivity (first-stay / ≥24 h) | `code/104_eligibility_sensitivity.py` | Supplement S1 table |
| Landmark derivation; COVID-era and decline-criterion exclusions | `code/106_landmark_and_sensitivity.py` | `results/v6_landmark24h_corrected.csv`, `results/v6_covid_era_sensitivity.csv`, `results/v6_excl_decline_only.csv` |
| IPW balance diagnostics (SMD, weights, ESS) | `code/107_ipw_recovery.py` | `results/v6_ipw_balance_smd.csv` |
| LVEF recovery; repeat-stay accounting | `code/109_recovery_readmission.py` | aggregated in `results/v6_recovery_readmission_log.txt` |
| Multiple imputation; Riley minimum sample size | `code/110_mi_riley.py`, `code/111_mi_sensitivity.R` | `results/v6_mi_sensitivity.csv` |
| Figures 1, 2, S2–S5 | `code/116_fig1_v6.py`, `code/114_fig2_v6.py`, `code/117_figS2_AJ.R`, `code/118a_figS3.py`, `code/118b_figS45.py` | `figures/` |
| Table 1 additional rows (peak troponin T, weight) | `code/120_table1_extra.py` | `results/v6_table1_extra_rows.csv` |
| Triage operating points (sensitivity/specificity by threshold) | `results/v6_triage_thresholds.csv` | — |

## Notes

- Aggregated result files contain no record-level identifiers; files containing such identifiers are excluded by design.
- The analysis plan (documented before modelling) is included in Supplement S1 of the manuscript; full computation logs are available from the corresponding author on reasonable request.
- Comments in some scripts are in Chinese; all outputs referenced by the manuscript are in English.

## License

Code is released under the MIT License (see `LICENSE`). The package contains no MIMIC-IV, MIMIC-IV-Echo or eICU-CRD record-level data; access to those datasets remains governed by the PhysioNet DUAs.

## Citation

If you use this code, please cite the manuscript (see `CITATION.cff`).

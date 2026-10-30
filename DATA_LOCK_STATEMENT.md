# Data Lock Statement

**Study**: Left Ventricular Systolic Dysfunction in Sepsis without Prior Heart Failure: 24-h Risk Recognition and 28-day Mortality: a retrospective development and internal validation study

**Database lock date**: 2026-10-20  
**Round-4 correction lock date**: 2026-10-30

## Database versions

| Database | Version | Source |
|:---|:---|:---|
| MIMIC-IV | v3.1 | PhysioNet |
| MIMIC-IV-Echo | v1.0.1 | PhysioNet |
| eICU-CRD | v2.0 | PhysioNet |

## Analysis code

**Primary analysis script**: `code/101_full_recompute_strict24h.py`  
**SHA256**: `26EC2AE60D599CFF5269EF633BCC9EE8A02F3522C87957EBEEF9AA498A32F40B`  
**Fixed random seed**: `20260101` for stochastic primary analyses.

Round-4 correction scripts (`round4_prepare_comprisk_fixed.py`, `round4_comprisk.R`, `round4_figS5.R`, `round4_landmark_clean.py`) are retained in the analysis working directory. They are not included in the public package because they read record-level MIMIC-IV extracts, which are excluded under the PhysioNet data-use agreements.

## Results

The package retains the aggregated v5/v6/v7/v8 result series and adds the Round-4 corrected outputs. Historical files (`results/v4_corrected_cif.csv`, `results/v4_corrected_finegray.csv`, `results/v6_landmark24h_corrected.csv`) are retained for provenance and must not be used for the corrected estimates.

Corrected files:

- `results/v10_comprisk_cif.csv` — 28-day Aalen–Johansen CIF 27.1%, naive 1−Kaplan–Meier 28.4% (SHA256 `593073672A5C3144C27E271BE7F8BEAA2B70E5A869380C899BF443C34BAAA1D8`)
- `results/v10_comprisk_finegray.csv` — Fine–Gray model with 214 timestamped SICM events (SHA256 `919E6CD426C81CEDFF67340E4E5DEFD8B46A4C9B322CC9783182522F07520277`)
- `results/v10_landmark_clean.csv` — landmark evaluation in the original held-out test set (n = 184, 22 events, AUC 0.658, 95% CI 0.528–0.779) and the exploratory 610-patient re-split (SHA256 `41118D9B60CF6C648D537FCF7914C2689FE1399B53BC2D88F8BDB50DF9902495`)
- `figures/figS2_cuminc_aj_km.png` — corrected cumulative-incidence figure (SHA256 `19E72A794382B5828AB7CD16EB55BFE3D0859089531421544D2C790D0258DC21`)

The primary penalised model was not refitted. `results/v8_primary_lasso_model.json` remains unchanged (SHA256 `5697DB9E10B34C9D13F3363CA029016B97E0E88639C3557D56519F1B34B14B82`).

## Round-4 corrections locked

1. Death time is taken from `admissions.deathtime` rather than the date-only `patients.dod`, removing 12 negative follow-up times and reclassifying 7 SICM cases that had been treated as competing deaths.
2. The 13 decline-only SICM cases are assigned the time of the confirmatory follow-up echocardiogram (52.1–569.5 h), so 214 of 215 SICM cases contribute a timestamped event; one case is censored at death.
3. The landmark analysis is reported in the original held-out test set (184 patients, 22 events, AUC 0.658). The 610-patient 70:30 re-split (AUC 0.655) is labelled exploratory because 115 of its 183 test patients were in the original development set.
4. Methods and limitations now document the troponin T assay (itemid 51003, upper reference limit 0.01 ng/mL, two-decimal reporting; >0.014 ng/mL is equivalent to ≥0.02 ng/mL in this dataset).

No record-level data or files containing patient/stay identifier columns are included; exclusions are recorded in `docs/EXCLUDED_FILES.txt`.

## Integrity verification

SHA256 manifests are provided as `code_checksums.txt`, `results_checksums.txt` and `figure_checksums.txt`. Run `sha256sum -c <manifest>` from the package root on Linux/Mac, or compare each listed hash with `Get-FileHash -Algorithm SHA256` in Windows PowerShell.

## Statement

The code, aggregated results and figures contained in this submission package were frozen on **2026-10-30**. This lock documents file integrity and provenance; it does not claim that every historical lineage script was re-executed on that date. Any subsequent change requires a new package version and lock statement.

**Locked by**: Study investigators  
**Date**: 2026-10-30

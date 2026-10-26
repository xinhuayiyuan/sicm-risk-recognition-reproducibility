# Data Lock Statement

**Study**: Left Ventricular Systolic Dysfunction in Sepsis without Prior Heart Failure: Early Risk Recognition and Mortality Risk

**Database lock date**: 2026-10-20  
**Submission package refresh/lock date**: 2026-10-26

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

## Results

The package retains the aggregated v5/v6 result series and adds v7 independent aggregate verification outputs. Key files include:

- `results/v6_landmark24h_corrected.csv` (landmark analysis, n = 610, 60 later events)
- `results/v6_mediation_boot2000.csv` (MIMIC mediation, 2,000 bootstrap replicates)
- `results/aim_eicu_V3_mediation.csv` (eICU proxy pathway, 300 bootstrap replicates)
- `results/v6_table1_extra_rows.csv` (additional Table 1 rows)
- `results/v7_analysis_set_metrics.csv` and `results/v7_oof_fold_details.csv`

No record-level data or files containing patient/stay identifier columns are included; exclusions are recorded in `docs/EXCLUDED_FILES.txt`.

## Integrity verification

SHA256 manifests are provided as `code_checksums.txt`, `results_checksums.txt` and `figure_checksums.txt`. Run `sha256sum -c <manifest>` from the package root on Linux/Mac, or compare each listed hash with `Get-FileHash -Algorithm SHA256` in Windows PowerShell.

## Statement

The code, aggregated results and figures contained in this submission package were frozen on **2026-10-26**. This lock documents file integrity and provenance; it does not claim that every historical lineage script was re-executed on that date. Any subsequent change requires a new package version and lock statement.

**Locked by**: Study investigators  
**Date**: 2026-10-26

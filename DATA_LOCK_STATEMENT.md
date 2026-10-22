# Data Lock Statement

**Study**: Left Ventricular Systolic Dysfunction in Sepsis without Prior Heart Failure: Early Risk Recognition and Mortality Risk

**Lock Date**: 2026-10-20 14:01 UTC

---

## Database Versions (Locked)

| Database | Version | Source | Lock Date |
|:---|:---|:---|:---:|
| MIMIC-IV | v3.1 | PhysioNet | 2026-10-20 |
| MIMIC-IV-Echo | v1.0.1 | PhysioNet | 2026-10-20 |
| eICU-CRD | v2.0 | PhysioNet | 2026-10-20 |

---

## Analysis Code (Locked)

**Primary analysis script**: `code/101_full_recompute_strict24h.py`  
**SHA256**: `6850A974B8EDCB817179ECDD7228A08D714388F540562C0C738DA0B8ACF15E9B`  
**Fixed random seed**: `20260101` (used throughout all scripts)

**Code checksums**: See `code_checksums.txt` for SHA256 hashes of all analysis scripts.

---

## Results (Locked)

**Final results version**: v6_*  
**Results checksums**: See `results_checksums.txt` for SHA256 hashes of all result files.

**Key result files**:
- `results/v6_landmark24h_corrected.csv` (landmark analysis, n = 610, 60 events)
- `results/v6_mediation_boot2000.csv` (mediation analysis, 2,000 bootstrap replicates)
- `results/v6_covid_era_sensitivity.csv` (COVID-era exclusion)
- `results/v6_excl_decline_only.csv` (decline-criterion exclusion)
- `results/v6_ipw_balance_smd.csv` (IPW balance diagnostics)
- `results/v6_mi_sensitivity.csv` (multiple imputation sensitivity)
- `results/v6_lvef_recovery_detail.csv` (LVEF recovery)
- `results/v6_table1_extra_rows.csv` (Table 1 additional rows)

---

## Lock Verification

To verify data integrity, regenerate checksums and compare with the provided files:

```bash
# On Linux/Mac
sha256sum -c code_checksums.txt
sha256sum -c results_checksums.txt

# On Windows (PowerShell)
Get-Content code_checksums.txt | ForEach-Object {
    $hash, $file = $_ -split '  ', 2
    $computed = (Get-FileHash $file -Algorithm SHA256).Hash
    if ($computed -eq $hash) { "OK: $file" } else { "MISMATCH: $file" }
}
```

---

## Statement

All analyses were completed and results locked on **2026-10-20 14:01 UTC**. No further modifications to data, code, or results were made after this date. Any subsequent changes will be documented in a new version with a corresponding lock statement.

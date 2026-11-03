# Round-7 corrected post-hoc analysis

Round-7 was conducted after the full data-chain audit on 2026-11-02. It is not represented as the original prespecified analysis. Earlier result files are retained for provenance.

## Implemented corrections

- ACS: ICD-9 410/411 and ICD-10 I21/I22.
- Qualifying low LVEF and decline follow-up at/after suspected infection.
- Treatment starts restricted to ICU 0–24 h; no pre-ICU start included.
- Corrected measured-selection source population and patient-clustered variance.
- Patient-grouped development/test and nested validation.
- Current-specification patient-cluster full-procedure IPW bootstrap (300 replicates).
- HF timing schemes, time-dependent Cox and 24-h landmark mortality analyses.
- Complete 44-variable model artifact, pure-rounding score, figures and manuscript rendering.

No separate cardiac-surgery/transplant/VAD exclusion was implemented because no auditable code list was available.

## Reproduction order

Run from the project root after the original materialised tables and strict AKI extract exist:

1. `python code/round6_prepare.py`
2. `python code/round7_prepare.py`
3. `python code/round7_full_recompute.py`
4. `python code/round7_models.py`
5. `python code/round7_metrics.py`
6. `python code/round7_time_mortality.py` and `Rscript code/round7_time_mortality.R`
7. `python code/round7_ipw_bootstrap.py`
8. `python code/round7_mediation_boot2000.py`
9. Table, sensitivity, figure and rendering scripts prefixed `round7_`.

`MIMIC_PASSWORD` is required in the environment. Record-level Round-7 files remain local and are not redistributed.

## Independently verified upstream definitions

- `troponin_sepsis`: itemid 51003, suspected infection ±24 h, 9,115 rows in the audited mirror. `sql/build_upstream_tables.sql` was executed and rolled back successfully.
- `hf_history`: index-admission ICD-10 I50 records.
- `build_imaging_v2.sql` and `build_imaging_criteria.sql`: both ACS eras; the stray incomplete SQL fragment was removed.

## Remaining reconstruction limitations

- The exact baseline/up-titration algorithm of legacy `vaso_new_upgrade` was not recoverable from the available lineage. It is not redefined or used in the locked model.
- The complete construction of `eicu_crd.temp_final_export` is unavailable. eICU remains a proxy analysis and is not claimed to be independently rebuildable from raw eICU-CRD.
- Round-7 uses the corrected cohort and must not be mixed with the earlier 1,783/789 analysis sets.

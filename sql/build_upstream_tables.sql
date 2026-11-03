-- Upstream tables whose definitions were independently verified against the local materialised tables.
-- Run after build_clinical_criteria.sql (sepsis_icu_cohort must exist).

DROP TABLE IF EXISTS mimiciv_echo.hf_history CASCADE;
CREATE TABLE mimiciv_echo.hf_history AS
SELECT DISTINCT i.stay_id
FROM mimiciv_icu.icustays i
JOIN mimiciv_hosp.diagnoses_icd d ON i.hadm_id = d.hadm_id
WHERE d.icd_version = 10 AND d.icd_code LIKE 'I50%';
CREATE INDEX idx_hf_history_stay ON mimiciv_echo.hf_history(stay_id);

DROP TABLE IF EXISTS mimiciv_echo.troponin_sepsis CASCADE;
CREATE TABLE mimiciv_echo.troponin_sepsis AS
SELECT s.stay_id, s.suspected_infection_time, l.itemid, l.charttime, l.valuenum,
       0.014 AS url_threshold,
       CASE WHEN l.valuenum > 0.014 THEN 1 ELSE 0 END AS trop_positive
FROM mimiciv_echo.sepsis_icu_cohort s
JOIN mimiciv_hosp.labevents l ON l.hadm_id = s.hadm_id
WHERE l.itemid = 51003
  AND l.valuenum IS NOT NULL
  AND l.charttime BETWEEN s.suspected_infection_time - INTERVAL '24 hours'
                      AND s.suspected_infection_time + INTERVAL '24 hours';
CREATE INDEX idx_troponin_sepsis_stay ON mimiciv_echo.troponin_sepsis(stay_id);

-- Validation target: 9,115 rows in the audited MIMIC-IV v3.1 mirror.
-- The legacy vaso_new_upgrade materialised table is not redefined here. Its exact
-- baseline/up-titration algorithm was not recoverable from the available lineage.
-- Round-7 therefore treats that component as legacy exploratory input and does not
-- use it in the locked prediction model. A separate documented derivation is still
-- required before claiming independent reconstruction of that mediator.

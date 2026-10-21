-- ============================================================
-- SICM 分析数据集提取 v2 (先建表, 再单独 \copy 导出)
-- SICM 定义: TnT>0.014 AND (任意EF<50% OR EF较基线下降>=10pp)
-- ============================================================
DROP TABLE IF EXISTS mimiciv_echo.sicm_analysis_dataset CASCADE;

CREATE TABLE mimiciv_echo.sicm_analysis_dataset AS
WITH lac AS (
    SELECT i.stay_id, MAX(l.valuenum) AS lactate_max_24h
    FROM mimiciv_hosp.labevents l
    INNER JOIN mimiciv_icu.icustays i ON l.hadm_id = i.hadm_id
    WHERE l.itemid IN (50813, 52442, 53154)
      AND l.charttime BETWEEN i.intime AND i.intime + INTERVAL '24 hours'
      AND l.valuenum IS NOT NULL
    GROUP BY i.stay_id
),
trop AS (
    SELECT stay_id, MAX(valuenum) AS trop_max
    FROM mimiciv_echo.troponin_sepsis
    WHERE valuenum IS NOT NULL
    GROUP BY stay_id
),
ef AS (
    SELECT stay_id, MIN(lvef_value) AS min_ef, COUNT(DISTINCT measurement_id) AS n_echo
    FROM mimiciv_echo.echo_lvef_numeric
    GROUP BY stay_id
),
efbase AS (
    SELECT stay_id, baseline_lvef FROM mimiciv_echo.echo_lvef_baseline
),
efdrop AS (
    SELECT stay_id, MAX(lvef_drop_absolute) AS max_ef_drop
    FROM mimiciv_echo.echo_lvef_followup
    GROUP BY stay_id
),
aki AS (
    SELECT stay_id, MAX(aki_stage) AS aki_stage_max
    FROM mimiciv_derived.kdigo_stages
    GROUP BY stay_id
),
vent AS (
    SELECT DISTINCT stay_id, 1 AS invasive_vent
    FROM mimiciv_derived.ventilation
    WHERE ventilation_status ILIKE '%Invasive%'
),
dobu AS (SELECT DISTINCT stay_id, 1 AS dobutamine FROM mimiciv_derived.dobutamine),
milr AS (SELECT DISTINCT stay_id, 1 AS milrinone FROM mimiciv_derived.milrinone),
rrtany AS (SELECT DISTINCT stay_id, 1 AS rrt_any FROM mimiciv_derived.rrt)
SELECT s.stay_id, s.subject_id, s.hadm_id, s.suspected_infection_time, s.sofa_score AS sepsis_sofa,
    d.gender, d.admission_age AS age, d.race, d.hospital_expire_flag AS hosp_mort,
    CASE WHEN d.dod IS NOT NULL AND d.dod <= d.icu_intime + INTERVAL '28 days' THEN 1 ELSE 0 END AS mort_28d,
    d.los_icu, d.los_hospital, d.first_icu_stay,
    fs.sofa AS sofa_24h, fs.respiration AS sofa_resp, fs.coagulation AS sofa_coag, fs.liver AS sofa_liver,
    fs.cardiovascular AS sofa_cardio, fs.cns AS sofa_cns, fs.renal AS sofa_renal,
    ch.charlson_comorbidity_index AS cci, ch.myocardial_infarct, ch.congestive_heart_failure,
    ch.peripheral_vascular_disease, ch.cerebrovascular_disease, ch.dementia, ch.chronic_pulmonary_disease,
    ch.rheumatic_disease, ch.peptic_ulcer_disease, ch.mild_liver_disease, ch.diabetes_without_cc,
    ch.diabetes_with_cc, ch.paraplegia, ch.renal_disease, ch.malignant_cancer, ch.severe_liver_disease,
    ch.metastatic_solid_tumor, ch.aids,
    CASE WHEN hf.stay_id IS NOT NULL THEN 1 ELSE 0 END AS has_hf,
    CASE WHEN acs.stay_id IS NOT NULL THEN 1 ELSE 0 END AS has_acs,
    lac.lactate_max_24h,
    fl.wbc_max, fl.platelets_min, fl.creatinine_max, fl.bun_max, fl.bilirubin_total_max, fl.albumin_min,
    fl.sodium_min, fl.sodium_max, fl.potassium_max, fl.glucose_max, fl.inr_max, fl.ptt_max,
    fl.bicarbonate_min, fl.aniongap_max, fl.hemoglobin_min,
    fv.heart_rate_max, fv.mbp_min, fv.resp_rate_max, fv.spo2_min, fv.temperature_max,
    COALESCE(vt.invasive_vent, 0) AS invasive_vent,
    vaso.vaso_new_upgrade, vaso.post_max_ned,
    COALESCE(db.dobutamine, 0) AS dobutamine, COALESCE(ml.milrinone, 0) AS milrinone,
    frrt.dialysis_present AS rrt_firstday, COALESCE(rt.rrt_any, 0) AS rrt_any,
    aki.aki_stage_max,
    t.trop_max, CASE WHEN t.trop_max > 0.014 THEN 1 ELSE 0 END AS trop_gt014,
    e.n_echo, CASE WHEN e.stay_id IS NOT NULL THEN 1 ELSE 0 END AS has_echo,
    eb.baseline_lvef AS baseline_ef, e.min_ef, ed.max_ef_drop,
    CASE WHEN e.min_ef < 50 OR ed.max_ef_drop >= 10 THEN 1 ELSE 0 END AS ef_criterion,
    CASE WHEN t.stay_id IS NOT NULL AND e.stay_id IS NOT NULL THEN 1 ELSE 0 END AS evaluable
FROM mimiciv_echo.sepsis_icu_cohort s
INNER JOIN mimiciv_derived.icustay_detail d ON s.stay_id = d.stay_id
LEFT JOIN mimiciv_derived.first_day_sofa fs ON s.stay_id = fs.stay_id
LEFT JOIN mimiciv_derived.charlson ch ON s.hadm_id = ch.hadm_id
LEFT JOIN mimiciv_echo.hf_history hf ON s.stay_id = hf.stay_id
LEFT JOIN mimiciv_echo.acs_exclusion acs ON s.stay_id = acs.stay_id
LEFT JOIN lac ON s.stay_id = lac.stay_id
LEFT JOIN mimiciv_derived.first_day_lab fl ON s.stay_id = fl.stay_id
LEFT JOIN mimiciv_derived.first_day_vitalsign fv ON s.stay_id = fv.stay_id
LEFT JOIN vent vt ON s.stay_id = vt.stay_id
LEFT JOIN mimiciv_echo.vaso_new_upgrade vaso ON s.stay_id = vaso.stay_id
LEFT JOIN dobu db ON s.stay_id = db.stay_id
LEFT JOIN milr ml ON s.stay_id = ml.stay_id
LEFT JOIN mimiciv_derived.first_day_rrt frrt ON s.stay_id = frrt.stay_id
LEFT JOIN rrtany rt ON s.stay_id = rt.stay_id
LEFT JOIN aki ON s.stay_id = aki.stay_id
LEFT JOIN trop t ON s.stay_id = t.stay_id
LEFT JOIN ef e ON s.stay_id = e.stay_id
LEFT JOIN efbase eb ON s.stay_id = eb.stay_id
LEFT JOIN efdrop ed ON s.stay_id = ed.stay_id;

CREATE INDEX idx_sicm_ds_stay ON mimiciv_echo.sicm_analysis_dataset(stay_id);

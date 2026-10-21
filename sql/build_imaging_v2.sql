-- ============================================================
-- MIMIC-IV Echo 影像学标准 (SICM) 构建脚本 v2 (优化版)
-- ============================================================

-- 1. 基线 EF (每 stay 最早的一次心超)
DROP TABLE IF EXISTS mimiciv_echo.echo_lvef_baseline CASCADE;
CREATE TABLE mimiciv_echo.echo_lvef_baseline AS
SELECT DISTINCT ON (stay_id)
    stay_id, hadm_id, subject_id, measurement_id, measurement_datetime AS baseline_datetime,
    lvef_value AS baseline_lvef, hours_from_icu_intime AS baseline_hours
FROM mimiciv_echo.echo_lvef_numeric
ORDER BY stay_id, measurement_datetime, measurement_id;

CREATE INDEX idx_lvef_base_stay ON mimiciv_echo.echo_lvef_baseline(stay_id);

-- 2. 后续 EF (非基线)
DROP TABLE IF EXISTS mimiciv_echo.echo_lvef_followup CASCADE;
CREATE TABLE mimiciv_echo.echo_lvef_followup AS
SELECT
    f.stay_id, f.measurement_id, f.measurement_datetime AS followup_datetime,
    f.lvef_value AS followup_lvef, f.hours_from_icu_intime AS followup_hours,
    b.baseline_lvef, b.baseline_datetime,
    b.baseline_lvef - f.lvef_value AS lvef_drop_absolute,
    CASE WHEN b.baseline_lvef > 0 THEN (b.baseline_lvef - f.lvef_value)/b.baseline_lvef * 100 ELSE NULL END AS lvef_drop_percent
FROM mimiciv_echo.echo_lvef_numeric f
INNER JOIN mimiciv_echo.echo_lvef_baseline b ON f.stay_id = b.stay_id
WHERE f.measurement_datetime > b.baseline_datetime;

CREATE INDEX idx_lvef_fu_stay ON mimiciv_echo.echo_lvef_followup(stay_id);

-- 3. 新发 EF 下降判定
DROP TABLE IF EXISTS mimiciv_echo.imaging_ef_drop CASCADE;
CREATE TABLE mimiciv_echo.imaging_ef_drop AS
SELECT DISTINCT ON (stay_id)
    stay_id, baseline_lvef, followup_lvef, lvef_drop_absolute,
    lvef_drop_percent, followup_datetime, followup_hours
FROM mimiciv_echo.echo_lvef_followup
WHERE baseline_lvef >= 50
  AND ( followup_lvef < 50 OR lvef_drop_absolute >= 10 )
ORDER BY stay_id, lvef_drop_absolute DESC;

-- 4. 室壁运动分类映射
DROP TABLE IF EXISTS mimiciv_echo.lvfunc_mapping CASCADE;
CREATE TABLE mimiciv_echo.lvfunc_mapping AS
SELECT stay_id, hadm_id, subject_id, measurement_id, measurement_datetime, result, hours_from_icu_intime,
    CASE
        WHEN result ILIKE '%Nl regional/global systolic function%' THEN 'normal'
        WHEN result ILIKE '%Nl regional LV WMA%' THEN 'normal'
        WHEN result ILIKE '%Suboptimal quality%' THEN 'indeterminate'
        WHEN result = '' OR result IS NULL THEN 'missing'
        WHEN result ILIKE '%focal dysfunction%' THEN 'abnormal'
        WHEN result ILIKE '%global hypo%' THEN 'abnormal'
        WHEN result ILIKE '%WMA%' THEN 'abnormal'
        WHEN result ILIKE '%Pseudodyskinesis%' THEN 'abnormal'
        WHEN result ILIKE '%Current focused - not full regional assess%' THEN 'indeterminate'
        ELSE 'abnormal'
    END AS func_category
FROM mimiciv_echo.echo_icu_link
WHERE measurement = 'lv_function1';

CREATE INDEX idx_lvfunc_map_stay ON mimiciv_echo.lvfunc_mapping(stay_id);
CREATE INDEX idx_lvfunc_map_cat ON mimiciv_echo.lvfunc_mapping(func_category);

-- 5. 新发室壁运动异常
DROP TABLE IF EXISTS mimiciv_echo.imaging_wma_new CASCADE;
CREATE TABLE mimiciv_echo.imaging_wma_new AS
WITH baseline_func AS (
    SELECT DISTINCT ON (stay_id) stay_id, measurement_datetime AS baseline_dt, func_category AS baseline_cat
    FROM mimiciv_echo.lvfunc_mapping WHERE func_category = 'normal'
    ORDER BY stay_id, measurement_datetime
),
followup_func AS (
    SELECT DISTINCT ON (stay_id) stay_id, measurement_datetime AS followup_dt, func_category AS followup_cat
    FROM mimiciv_echo.lvfunc_mapping WHERE func_category = 'abnormal'
    ORDER BY stay_id, measurement_datetime
)
SELECT b.stay_id, b.baseline_dt, b.baseline_cat, f.followup_dt, f.followup_cat
FROM baseline_func b
INNER JOIN followup_func f ON b.stay_id = f.stay_id AND f.followup_dt > b.baseline_dt;

-- 6. 影像学阳性合并 + ACS排除 + 最终结果
DROP TABLE IF EXISTS mimiciv_echo.acs_exclusion CASCADE;
CREATE TABLE mimiciv_echo.acs_exclusion AS
SELECT DISTINCT i.stay_id
FROM mimiciv_icu.icustays i
INNER JOIN mimiciv_hosp.diagnoses_icd d ON i.hadm_id = d.hadm_id
WHERE d.icd_code LIKE 'I21%' OR d.icd_code LIKE 'I22%';

CREATE INDEX idx_acs_stay ON mimiciv_echo.acs_exclusion(stay_id);

DROP TABLE IF EXISTS mimiciv_echo.sicm_imaging_criteria CASCADE;
CREATE TABLE mimiciv_echo.sicm_imaging_criteria AS
SELECT
    ic.stay_id,
    CASE WHEN ef.stay_id IS NOT NULL OR wma.stay_id IS NOT NULL THEN 1 ELSE 0 END AS imaging_positive,
    CASE WHEN acs.stay_id IS NOT NULL THEN 1 ELSE 0 END AS has_acs,
    CASE WHEN (ef.stay_id IS NOT NULL OR wma.stay_id IS NOT NULL) AND acs.stay_id IS NULL THEN 1 ELSE 0 END AS sicm_imaging
FROM mimiciv_icu.icustays ic
LEFT JOIN mimiciv_echo.imaging_ef_drop ef ON ic.stay_id = ef.stay_id
LEFT JOIN mimiciv_echo.imaging_wma_new wma ON ic.stay_id = wma.stay_id
LEFT JOIN mimiciv_echo.acs_exclusion acs ON ic.stay_id = acs.stay_id;

CREATE INDEX idx_sicm_img_stay ON mimiciv_echo.sicm_imaging_criteria(stay_id);

-- 7. 统计
SELECT 'Total ICU stays' AS metric, COUNT(*)::text AS value FROM mimiciv_icu.icustays
UNION ALL
SELECT 'ICU stays with echo (lvef)' AS metric, COUNT(DISTINCT stay_id)::text FROM mimiciv_echo.echo_lvef_numeric
UNION ALL
SELECT 'ICU stays with echo (lv_func)' AS metric, COUNT(DISTINCT stay_id)::text FROM mimiciv_echo.lvfunc_mapping
UNION ALL
SELECT 'Imaging: New EF drop' AS metric, COUNT(*)::text FROM mimiciv_echo.imaging_ef_drop
UNION ALL
SELECT 'Imaging: New WMA' AS metric, COUNT(*)::text FROM mimiciv_echo.imaging_wma_new
UNION ALL
SELECT 'Imaging: Positive (EF or WMA)' AS metric, COUNT(DISTINCT stay_id)::text FROM (
    SELECT stay_id FROM mimiciv_echo.imaging_ef_drop
    UNION SELECT stay_id FROM mimiciv_echo.imaging_wma_new
) t
UNION ALL
SELECT 'Excluded for ACS' AS metric, COUNT(*)::text FROM mimiciv_echo.acs_exclusion
UNION ALL
SELECT 'Final SICM (Imaging) after ACS exclusion' AS metric, SUM(sicm_imaging)::text FROM mimiciv_echo.sicm_imaging_criteria;

-- ============================================================
-- MIMIC-IV Echo 影像学标准 (SICM) 构建脚本
-- 标准: 新发左室射血分数下降 或 新发室壁运动异常
-- ============================================================

-- 1. 将 echo 测量与 icustays 做时间关联
DROP TABLE IF EXISTS mimiciv_echo.echo_icu_link CASCADE;
CREATE TABLE mimiciv_echo.echo_icu_link AS
SELECT e.subject_id, e.measurement_id, e.measurement_datetime, e.test_type,
       e.measurement, e.result, e.unit,
       i.stay_id, i.hadm_id, i.intime, i.outtime,
       EXTRACT(EPOCH FROM (e.measurement_datetime - i.intime))/3600.0 AS hours_from_icu_intime
FROM mimiciv_echo.lvef e
INNER JOIN mimiciv_icu.icustays i
    ON e.subject_id = i.subject_id
    AND e.measurement_datetime >= i.intime
    AND e.measurement_datetime <= i.outtime
UNION ALL
SELECT e.subject_id, e.measurement_id, e.measurement_datetime, e.test_type,
       e.measurement, e.result, e.unit,
       i.stay_id, i.hadm_id, i.intime, i.outtime,
       EXTRACT(EPOCH FROM (e.measurement_datetime - i.intime))/3600.0
FROM mimiciv_echo.lv_function e
INNER JOIN mimiciv_icu.icustays i
    ON e.subject_id = i.subject_id
    AND e.measurement_datetime >= i.intime
    AND e.measurement_datetime <= i.outtime;

CREATE INDEX idx_echo_icu_link_stay ON mimiciv_echo.echo_icu_link(stay_id);
CREATE INDEX idx_echo_icu_link_subject ON mimiciv_echo.echo_icu_link(subject_id);

-- 2. 提取数值型 EF
DROP TABLE IF EXISTS mimiciv_echo.echo_lvef_numeric CASCADE;
CREATE TABLE mimiciv_echo.echo_lvef_numeric AS
SELECT DISTINCT ON (stay_id, measurement_id)
    stay_id, hadm_id, subject_id, measurement_id, measurement_datetime, test_type,
    CASE WHEN result ~ '^[0-9]+(\.[0-9]+)?$' THEN result::numeric ELSE NULL END AS lvef_value,
    hours_from_icu_intime
FROM mimiciv_echo.echo_icu_link
WHERE measurement = 'lvef' AND result IS NOT NULL AND result <> '';

CREATE INDEX idx_echo_lvef_stay ON mimiciv_echo.echo_lvef_numeric(stay_id);
CREATE INDEX idx_echo_lvef_time ON mimiciv_echo.echo_lvef_numeric(measurement_datetime);

-- 3. 基线 EF (同一 stay 最早的一次)
DROP TABLE IF EXISTS mimiciv_echo.echo_lvef_baseline CASCADE;
CREATE TABLE mimiciv_echo.echo_lvef_baseline AS
SELECT stay_id,
    MIN(measurement_datetime) AS baseline_datetime,
    MIN(lvef_value) FILTER (WHERE measurement_datetime = (SELECT MIN(measurement_datetime) FROM mimiciv_echo.echo_lvef_numeric b WHERE b.stay_id = a.stay_id)) AS baseline_lvef
FROM mimiciv_echo.echo_lvef_numeric a
GROUP BY stay_id;

-- 4. 后续 EF (非基线)
DROP TABLE IF EXISTS mimiciv_echo.echo_lvef_followup CASCADE;
CREATE TABLE mimiciv_echo.echo_lvef_followup AS
SELECT a.stay_id, a.measurement_id, a.measurement_datetime AS followup_datetime,
    a.lvef_value AS followup_lvef, a.hours_from_icu_intime,
    b.baseline_lvef, b.baseline_datetime,

-- 5. 新发 EF 下降判定 (基线>=50%, 后续<50% 或下降>=10个百分点)
DROP TABLE IF EXISTS mimiciv_echo.imaging_ef_drop CASCADE;
CREATE TABLE mimiciv_echo.imaging_ef_drop AS
SELECT DISTINCT ON (stay_id)
    stay_id, baseline_lvef, followup_lvef, lvef_drop_absolute,
    lvef_drop_percent, followup_datetime, hours_from_icu_intime
FROM mimiciv_echo.echo_lvef_followup
WHERE baseline_lvef >= 50
  AND ( followup_lvef < 50 OR lvef_drop_absolute >= 10 )
ORDER BY stay_id, lvef_drop_absolute DESC;

-- 6. 室壁运动分类映射
DROP TABLE IF EXISTS mimiciv_echo.lvfunc_mapping CASCADE;
CREATE TABLE mimiciv_echo.lvfunc_mapping AS
SELECT stay_id, hadm_id, subject_id, measurement_id, measurement_datetime, result,
    hours_from_icu_intime,
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

-- 7. 新发室壁运动异常 (基线 normal -> 后续 abnormal)
DROP TABLE IF EXISTS mimiciv_echo.imaging_wma_new CASCADE;
CREATE TABLE mimiciv_echo.imaging_wma_new AS
WITH baseline_func AS (
    SELECT DISTINCT ON (stay_id) stay_id, measurement_datetime AS baseline_dt, func_category AS baseline_cat
    FROM mimiciv_echo.lvfunc_mapping
    WHERE func_category = 'normal'
    ORDER BY stay_id, measurement_datetime
),
followup_func AS (
    SELECT DISTINCT ON (stay_id) stay_id, measurement_datetime AS followup_dt, func_category AS followup_cat
    FROM mimiciv_echo.lvfunc_mapping
    WHERE func_category = 'abnormal'
    ORDER BY stay_id, measurement_datetime
)
SELECT b.stay_id, b.baseline_dt, b.baseline_cat, f.followup_dt, f.followup_cat
FROM baseline_func b
INNER JOIN followup_func f ON b.stay_id = f.stay_id AND f.followup_dt > b.baseline_dt;

-- 8. 影像学标准阳性合并
DROP TABLE IF EXISTS mimiciv_echo.imaging_positive CASCADE;
CREATE TABLE mimiciv_echo.imaging_positive AS
SELECT stay_id, 1 AS imaging_positive FROM mimiciv_echo.imaging_ef_drop
UNION
SELECT stay_id, 1 AS imaging_positive FROM mimiciv_echo.imaging_wma_new;

-- 9. ACS 排除 (ICD-10 I21/I22)
DROP TABLE IF EXISTS mimiciv_echo.acs_exclusion CASCADE;
CREATE TABLE mimiciv_echo.acs_exclusion AS
SELECT DISTINCT i.stay_id
FROM mimiciv_icu.icustays i
INNER JOIN mimiciv_hosp.diagnoses_icd d ON i.hadm_id = d.hadm_id
WHERE d.icd_code LIKE 'I21%' OR d.icd_code LIKE 'I22%';

CREATE INDEX idx_acs_stay ON mimiciv_echo.acs_exclusion(stay_id);

-- 10. 最终影像学标准结果表
DROP TABLE IF EXISTS mimiciv_echo.sicm_imaging_criteria CASCADE;
CREATE TABLE mimiciv_echo.sicm_imaging_criteria AS
SELECT ic.stay_id,
    CASE WHEN img.stay_id IS NOT NULL THEN 1 ELSE 0 END AS imaging_positive,
    CASE WHEN acs.stay_id IS NOT NULL THEN 1 ELSE 0 END AS has_acs,
    CASE WHEN img.stay_id IS NOT NULL AND acs.stay_id IS NULL THEN 1 ELSE 0 END AS sicm_imaging
FROM mimiciv_icu.icustays ic
LEFT JOIN mimiciv_echo.imaging_positive img ON ic.stay_id = img.stay_id
LEFT JOIN mimiciv_echo.acs_exclusion acs ON ic.stay_id = acs.stay_id;

CREATE INDEX idx_sicm_img_stay ON mimiciv_echo.sicm_imaging_criteria(stay_id);

-- 11. 统计
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
SELECT 'Imaging: Positive (EF or WMA)' AS metric, COUNT(*)::text FROM mimiciv_echo.imaging_positive
UNION ALL
SELECT 'Excluded for ACS' AS metric, COUNT(*)::text FROM mimiciv_echo.acs_exclusion
UNION ALL
SELECT 'Final SICM (Imaging) after ACS exclusion' AS metric, SUM(sicm_imaging)::text FROM mimiciv_echo.sicm_imaging_criteria;

    b.baseline_lvef - a.lvef_value AS lvef_drop_absolute,
    CASE WHEN b.baseline_lvef > 0 THEN (b.baseline_lvef - a.lvef_value)/b.baseline_lvef * 100 ELSE NULL END AS lvef_drop_percent
FROM mimiciv_echo.echo_lvef_numeric a
INNER JOIN mimiciv_echo.echo_lvef_baseline b
    ON a.stay_id = b.stay_id AND a.measurement_datetime > b.baseline_datetime;

CREATE INDEX idx_echo_lvef_fu_stay ON mimiciv_echo.echo_lvef_followup(stay_id);

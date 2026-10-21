-- ============================================================
-- SICM 影像学标准 敏感性分析
-- 从严格到宽松的多层级定义
-- ============================================================

-- Tier 0: 当前标准 (基线EF>=50 + 后续下降>=10 或 <50)
-- 已存在: mimiciv_echo.imaging_ef_drop

-- Tier 1: 放宽基线到 >=40 (允许轻中度受损基线)
DROP TABLE IF EXISTS mimiciv_echo.img_tier1_ef_drop CASCADE;
CREATE TABLE mimiciv_echo.img_tier1_ef_drop AS
SELECT DISTINCT ON (f.stay_id)
    f.stay_id, b.baseline_lvef, f.followup_lvef, f.lvef_drop_absolute
FROM mimiciv_echo.echo_lvef_followup f
INNER JOIN mimiciv_echo.echo_lvef_baseline b ON f.stay_id = b.stay_id
INNER JOIN mimiciv_echo.sepsis_icu_cohort s ON f.stay_id = s.stay_id
WHERE b.baseline_lvef >= 40
  AND ( f.followup_lvef < 40 OR f.lvef_drop_absolute >= 10 )
ORDER BY f.stay_id, f.lvef_drop_absolute DESC;

-- Tier 2: 不限制基线正常，任何下降>=10个百分点即算
DROP TABLE IF EXISTS mimiciv_echo.img_tier2_ef_drop CASCADE;
CREATE TABLE mimiciv_echo.img_tier2_ef_drop AS
SELECT DISTINCT ON (f.stay_id)
    f.stay_id, b.baseline_lvef, f.followup_lvef, f.lvef_drop_absolute
FROM mimiciv_echo.echo_lvef_followup f
INNER JOIN mimiciv_echo.echo_lvef_baseline b ON f.stay_id = b.stay_id
INNER JOIN mimiciv_echo.sepsis_icu_cohort s ON f.stay_id = s.stay_id
WHERE f.lvef_drop_absolute >= 10
ORDER BY f.stay_id, f.lvef_drop_absolute DESC;

-- Tier 3: 任何ICU期间EF<40（不强调"新发"，仅反映严重收缩功能障碍）
DROP TABLE IF EXISTS mimiciv_echo.img_tier3_ef_low CASCADE;
CREATE TABLE mimiciv_echo.img_tier3_ef_low AS
SELECT DISTINCT ON (el.stay_id)
    el.stay_id, el.lvef_value AS lowest_lvef
FROM mimiciv_echo.echo_lvef_numeric el
INNER JOIN mimiciv_echo.sepsis_icu_cohort s ON el.stay_id = s.stay_id
WHERE el.lvef_value < 40
ORDER BY el.stay_id, el.lvef_value;

-- Tier 4: 任何ICU期间EF<50（更敏感的收缩功能异常）
DROP TABLE IF EXISTS mimiciv_echo.img_tier4_ef_low CASCADE;
CREATE TABLE mimiciv_echo.img_tier4_ef_low AS
SELECT DISTINCT ON (el.stay_id)
    el.stay_id, el.lvef_value AS lowest_lvef
FROM mimiciv_echo.echo_lvef_numeric el
INNER JOIN mimiciv_echo.sepsis_icu_cohort s ON el.stay_id = s.stay_id
WHERE el.lvef_value < 50
ORDER BY el.stay_id, el.lvef_value;

-- Tier A: WMA 新发 (当前)
-- 已存在: mimiciv_echo.imaging_wma_new

-- Tier B: WMA 任何时间存在（不强调"新发"）
DROP TABLE IF EXISTS mimiciv_echo.img_tierb_wma_any CASCADE;
CREATE TABLE mimiciv_echo.img_tierb_wma_any AS
SELECT DISTINCT lm.stay_id
FROM mimiciv_echo.lvfunc_mapping lm
INNER JOIN mimiciv_echo.sepsis_icu_cohort s ON lm.stay_id = s.stay_id
WHERE lm.func_category = 'abnormal';

-- 合并各层级影像学阳性 (排除ACS)
DROP TABLE IF EXISTS mimiciv_echo.img_tier_combined CASCADE;
CREATE TABLE mimiciv_echo.img_tier_combined AS
SELECT
    s.stay_id,
    CASE WHEN t0.stay_id IS NOT NULL AND acs.stay_id IS NULL THEN 1 ELSE 0 END AS img_tier0,  -- current
    CASE WHEN t1.stay_id IS NOT NULL AND acs.stay_id IS NULL THEN 1 ELSE 0 END AS img_tier1,  -- baseline>=40
    CASE WHEN t2.stay_id IS NOT NULL AND acs.stay_id IS NULL THEN 1 ELSE 0 END AS img_tier2,  -- any drop>=10
    CASE WHEN t3.stay_id IS NOT NULL AND acs.stay_id IS NULL THEN 1 ELSE 0 END AS img_tier3,  -- any EF<40
    CASE WHEN t4.stay_id IS NOT NULL AND acs.stay_id IS NULL THEN 1 ELSE 0 END AS img_tier4,  -- any EF<50
    CASE WHEN w0.stay_id IS NOT NULL AND acs.stay_id IS NULL THEN 1 ELSE 0 END AS wma_tier0,  -- WMA new
    CASE WHEN wb.stay_id IS NOT NULL AND acs.stay_id IS NULL THEN 1 ELSE 0 END AS wma_tierb   -- WMA any
FROM mimiciv_echo.sepsis_icu_cohort s
LEFT JOIN mimiciv_echo.imaging_ef_drop t0 ON s.stay_id = t0.stay_id
LEFT JOIN mimiciv_echo.img_tier1_ef_drop t1 ON s.stay_id = t1.stay_id
LEFT JOIN mimiciv_echo.img_tier2_ef_drop t2 ON s.stay_id = t2.stay_id
LEFT JOIN mimiciv_echo.img_tier3_ef_low t3 ON s.stay_id = t3.stay_id
LEFT JOIN mimiciv_echo.img_tier4_ef_low t4 ON s.stay_id = t4.stay_id
LEFT JOIN mimiciv_echo.imaging_wma_new w0 ON s.stay_id = w0.stay_id
LEFT JOIN mimiciv_echo.img_tierb_wma_any wb ON s.stay_id = wb.stay_id
LEFT JOIN mimiciv_echo.acs_exclusion acs ON s.stay_id = acs.stay_id;

CREATE INDEX idx_img_tier_stay ON mimiciv_echo.img_tier_combined(stay_id);

-- 与临床标准合并做交叉表
DROP TABLE IF EXISTS mimiciv_echo.sicm_sensitivity CASCADE;
CREATE TABLE mimiciv_echo.sicm_sensitivity AS
SELECT
    c.stay_id,
    c.sicm_clinical,
    i.img_tier0, i.img_tier1, i.img_tier2, i.img_tier3, i.img_tier4,
    i.wma_tier0, i.wma_tierb,
    -- 影像学阳性 = EF标准 或 WMA标准 (各层级)
    CASE WHEN i.img_tier0 = 1 OR i.wma_tier0 = 1 THEN 1 ELSE 0 END AS img_current,
    CASE WHEN i.img_tier1 = 1 OR i.wma_tier0 = 1 THEN 1 ELSE 0 END AS img_tier1_comb,
    CASE WHEN i.img_tier2 = 1 OR i.wma_tier0 = 1 THEN 1 ELSE 0 END AS img_tier2_comb,
    CASE WHEN i.img_tier3 = 1 OR i.wma_tierb = 1 THEN 1 ELSE 0 END AS img_tier3_comb,
    CASE WHEN i.img_tier4 = 1 OR i.wma_tierb = 1 THEN 1 ELSE 0 END AS img_tier4_comb
FROM mimiciv_echo.sicm_clinical_criteria c
FULL OUTER JOIN mimiciv_echo.img_tier_combined i ON c.stay_id = i.stay_id
WHERE c.stay_id IN (SELECT stay_id FROM mimiciv_echo.sepsis_icu_cohort)
   OR i.stay_id IN (SELECT stay_id FROM mimiciv_echo.sepsis_icu_cohort);

CREATE INDEX idx_sicm_sens_stay ON mimiciv_echo.sicm_sensitivity(stay_id);

-- ============================================================
-- MIMIC-IV 临床复合标准 (SICM) 构建脚本
-- 文献依据: Sepsis-induced cardiomyopathy (SICM) 临床定义
-- 标准: 肌钙蛋白 > URL第99百分位 + 新发/升级血管活性药物 + 排除ACS
-- ============================================================

-- 1. 脓毒症 ICU 队列 (suspected_infection_time 落在 ICU 期间)
DROP TABLE IF EXISTS mimiciv_echo.sepsis_icu_cohort CASCADE;
CREATE TABLE mimiciv_echo.sepsis_icu_cohort AS
SELECT
    s.subject_id,
    s.stay_id,
    s.hadm_id,
    s.suspected_infection_time,
    s.sofa_time,
    s.sofa_score,
    i.intime,
    i.outtime
FROM mimiciv_derived.sepsis3 s
INNER JOIN mimiciv_icu.icustays i ON s.stay_id = i.stay_id
WHERE s.suspected_infection_time BETWEEN i.intime AND i.outtime;

CREATE INDEX idx_sepsis_icu_stay ON mimiciv_echo.sepsis_icu_cohort(stay_id);
CREATE INDEX idx_sepsis_icu_subj ON mimiciv_echo.sepsis_icu_cohort(subject_id);

SELECT 'Sepsis ICU cohort' AS metric, COUNT(*)::text AS value FROM mimiciv_echo.sepsis_icu_cohort;

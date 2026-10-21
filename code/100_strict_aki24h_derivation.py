# 100_strict_aki24h_derivation.py — database-side derivation checks
# 1) strict 0-24h AKI stage (KDIGO creatinine criteria) vs whole-stay aki_stage_max
# 2) window characterisation of vaso_new_upgrade / post_max_ned
# 3) eICU n=6,006流程数字
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, psycopg2
import sys, traceback
pd.set_option('display.width', 200)

LOG = open('<PROJECT_ROOT>/exec/results/v4_verify_db.txt', 'w', encoding='utf-8')
def print(*a, **kw):  # noqa: A001
    sys.stdout.write(' '.join(str(x) for x in a) + '\n')
    LOG.write(' '.join(str(x) for x in a) + '\n'); LOG.flush()

def _hook(exc_type, exc, tb):
    LOG.write(''.join(traceback.format_exception(exc_type, exc, tb))); LOG.flush()
    sys.__excepthook__(exc_type, exc, tb)
sys.excepthook = _hook

D = '<PROJECT_ROOT>/exec'
co = pd.read_csv(f'{D}/data/cohort_main.csv')
conn = psycopg2.connect(dbname='mimic', host='127.0.0.1', port=5431,
                        user='postgres', password='<SET-IN-ENVIRONMENT>')

# ---------- 1. 严格0-24h AKI分期（kdigo_stages按charttime限窗） ----------
aki24 = pd.read_sql("""
    SELECT k.stay_id, MAX(k.aki_stage) AS aki_stage_24h
    FROM mimiciv_derived.kdigo_stages k
    JOIN mimiciv_icu.icustays i ON k.stay_id = i.stay_id
    WHERE k.charttime BETWEEN i.intime AND i.intime + INTERVAL '24 hours'
    GROUP BY k.stay_id""", conn)
print('aki24 rows:', len(aki24))

d = co.merge(aki24, on='stay_id', how='left')
d['aki_stage_24h'] = d['aki_stage_24h'].fillna(0).astype(int)

print('\n=== AKI分期对比（全队列n=%d）===' % len(d))
print('whole-stay aki_stage_max:'); print(d.aki_stage_max.value_counts(dropna=False).sort_index())
print('strict 0-24h aki_stage_24h:'); print(d.aki_stage_24h.value_counts(dropna=False).sort_index())
both = d.dropna(subset=['aki_stage_max', 'aki_stage_24h'])
print('一致率:', (both.aki_stage_max == both.aki_stage_24h).mean().round(3),
      '| 24h低估(whole>24h):', (both.aki_stage_max > both.aki_stage_24h).mean().round(3))
print('spearman:', both[['aki_stage_max', 'aki_stage_24h']].corr(method='spearman').iloc[0, 1].round(3))
print('按SICM分层的24h分期:'); print(d.groupby('sicm').aki_stage_24h.value_counts().unstack(fill_value=0))

# 修正新发队列子集
dn = d[(d.has_hf == 0) & (d.congestive_heart_failure == 0)]
print('\n新发队列(n=%d): 24h分期分布:' % len(dn))
print(dn.aki_stage_24h.value_counts(dropna=False).sort_index())
d[['stay_id', 'aki_stage_24h']].to_csv(f'{D}/data/aki_stage_24h.csv', index=False)
print('[saved] aki_stage_24h.csv')

# ---------- 2. vaso_new_upgrade / post_max_ned 窗口刻画 ----------
try:
    v = pd.read_sql("SELECT * FROM mimiciv_echo.vaso_new_upgrade LIMIT 5", conn)
    print('\n=== vaso_new_upgrade表结构 ==='); print(v.columns.tolist()); print(v.head())
except Exception as e:
    print('vaso table err:', e)
cnt = pd.read_sql("""SELECT COUNT(*) total, SUM(vaso_new_upgrade) n_upgrade,
    AVG(post_max_ned) avg_ned, MAX(post_max_ned) max_ned
    FROM mimiciv_echo.vaso_new_upgrade""", conn)
print(cnt)
# 与NE启用时间关系（inputevents）: 随机核对是否有24h后启用者被标记
chk = pd.read_sql("""
    SELECT v.stay_id, v.vaso_new_upgrade, v.post_max_ned,
           MIN(EXTRACT(EPOCH FROM (ie.starttime - i.intime))/3600.0) AS ne_first_h,
           MAX(EXTRACT(EPOCH FROM (ie.endtime - i.intime))/3600.0) AS ne_last_h
    FROM mimiciv_echo.vaso_new_upgrade v
    JOIN mimiciv_icu.icustays i ON v.stay_id = i.stay_id
    JOIN mimiciv_icu.inputevents ie ON v.stay_id = ie.stay_id
    JOIN mimiciv_icu.d_items di ON ie.itemid = di.itemid
    WHERE di.label ILIKE 'norepinephrine%' AND ie.rate > 0
    GROUP BY v.stay_id, v.vaso_new_upgrade, v.post_max_ned
    LIMIT 100000""", conn)
print('\nNE时序（n=%d stays with NE）:' % len(chk))
print('ne_first_h 分位数:', chk.ne_first_h.quantile([0.05, 0.5, 0.95]).round(1).tolist())
print('ne_last_h>24h 比例:', (chk.ne_last_h > 24).mean().round(3))
print('post_max_ned>0 但 ne_first_h>24h 的比例:',
      (chk[chk.post_max_ned > 0].ne_first_h > 24).mean().round(3))

# ---------- 3. eICU流程 ----------
conn2 = psycopg2.connect(dbname='eicu', host='127.0.0.1', port=5431,
                         user='postgres', password='<SET-IN-ENVIRONMENT>')
ei0 = pd.read_sql("SELECT COUNT(*) n FROM eicu_crd.temp_final_export", conn2)
ei = pd.read_sql("SELECT age, los_icu_days FROM eicu_crd.temp_final_export", conn2)
ei['age_num'] = pd.to_numeric(ei['age'].astype(str).str.replace('> 89', '90'), errors='coerce')
n1 = len(ei)
ei2 = ei[(ei.age_num >= 18) & (ei.age_num <= 90) & (ei.los_icu_days >= 1)]
n2 = len(ei2)
trop_cols = pd.read_sql("""SELECT column_name FROM information_schema.columns
    WHERE table_schema='eicu_crd' AND table_name='temp_final_export'
    AND column_name LIKE 'troponin_i%'""", conn2).column_name.tolist()
t = pd.read_sql("SELECT %s FROM eicu_crd.temp_final_export" % ','.join(trop_cols), conn2)
t['age_num'] = ei['age_num']; t['los'] = ei['los_icu_days']
t = t[(t.age_num >= 18) & (t.age_num <= 90) & (t.los >= 1)]
tc = [c for c in trop_cols if c != 'troponin_i_first']
n3 = t[t[tc + ['troponin_i_first']].max(axis=1).notna()].shape[0]
print(f'\n=== eICU流程: 导出 {n1} -> 年龄18-90&LOS>=1d {n2} -> 有cTnI测定 {n3} ===')
conn2.close(); conn.close()
print('[done]')

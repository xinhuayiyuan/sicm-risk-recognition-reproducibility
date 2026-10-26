# 62_prepare_comprisk.py - S5竞争风险数据准备
# 事件: 1=SICM(首次EF<50超声时刻), 2=28天内死亡(早于SICM), 0=删失(672h)
# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, psycopg2

D = '<PROJECT_ROOT>/exec'
co = pd.read_csv(f'{D}/data/cohort_main.csv')
dn = co[co['has_hf'] == 0][['stay_id', 'sicm', 'mort_28d', 'lactate_max_24h',
                            'sofa_24h', 'age', 'post_max_ned']].copy()

conn = psycopg2.connect(dbname='mimic', host='127.0.0.1', port=5431,
                        user='postgres', password='<SET-IN-ENVIRONMENT>')
ef = pd.read_sql("""
    SELECT stay_id, MIN(hours_from_icu_intime) AS t_ef50
    FROM mimiciv_echo.echo_lvef_numeric
    WHERE lvef_value < 50 AND hours_from_icu_intime >= 0
    GROUP BY stay_id""", conn)
dea = pd.read_sql("""
    SELECT i.stay_id,
           EXTRACT(EPOCH FROM (p.dod - i.intime))/3600.0 AS t_death
    FROM mimiciv_icu.icustays i
    JOIN mimiciv_hosp.patients p ON i.subject_id = p.subject_id
    WHERE p.dod IS NOT NULL AND p.dod <= i.intime + INTERVAL '28 days'""", conn)
conn.close()

dn = dn.merge(ef, on='stay_id', how='left').merge(dea, on='stay_id', how='left')
CENS = 672.0  # 28天
# 只有 sicm==1 者才可能有SICM事件（表型需同时满足cTn）
dn['t_sicm'] = np.where(dn['sicm'] == 1, dn['t_ef50'], np.nan)
# 表型一的EF标准含"降≥10pp"，此处以EF<50首次时刻近似（文档声明）
def resolve(r):
    ts, td = r['t_sicm'], r['t_death']
    if pd.notna(ts) and (pd.isna(td) or ts <= td):
        return min(ts, CENS), 1
    if pd.notna(td):
        return min(td, CENS), 2
    return CENS, 0
dn[['time', 'status']] = dn.apply(lambda r: pd.Series(resolve(r)), axis=1)
print(dn['status'].value_counts().to_string())
print(f"SICM事件中位时间: {dn.loc[dn.status==1,'time'].median():.1f}h")
dn.to_csv(f'{D}/data/comprisk_dn.csv', index=False)
print("[done] comprisk_dn.csv")

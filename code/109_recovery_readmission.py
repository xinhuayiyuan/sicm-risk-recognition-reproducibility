# 109_recovery_readmission.py — LVEF recovery after SICM onset; repeat-stay accounting
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, psycopg2

D = 'D:/MIMICIV/exec'
OUT = []
def log(s):
    print(s); OUT.append(str(s))

conn = psycopg2.connect(dbname='mimic', host='127.0.0.1', port=5431,
                        user='postgres', password='123456')
co = pd.read_csv(f'{D}/data/cohort_main.csv')
dn = co[(co['has_hf'] == 0) & (co['congestive_heart_failure'] == 0)].copy()

# ---- A7: 再入院口径 ----
log(f'[stays] cohort stays={len(co)}, unique patients={co.subject_id.nunique()}')
multi = co.groupby('subject_id').size()
log(f'[stays] patients with >1 stay in cohort: {(multi>1).sum()}, extra stays: {len(co)-co.subject_id.nunique()}')
rk = pd.read_sql("""SELECT stay_id, subject_id,
                    ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime) AS rn
                    FROM mimiciv_icu.icustays""", conn)
m = co.merge(rk, on=['stay_id', 'subject_id'], how='left')

# ---- LVEF recovery rate ----
lv = pd.read_sql("""SELECT stay_id, measurement_datetime, lvef_value, hours_from_icu_intime
                    FROM mimiciv_echo.echo_lvef_numeric""", conn)
s = dn[dn['sicm'] == 1][['stay_id', 'mort_28d']].copy()
e = lv.merge(s, on='stay_id', how='inner').sort_values(['stay_id', 'hours_from_icu_intime'])
log(f'\n[recovery] SICM patients with any LVEF record: {e.stay_id.nunique()} / {len(s)}')
per = e.groupby('stay_id').size()
log(f'[recovery] LVEF records per SICM patient: median {per.median():.0f}, >=2 records: {(per>=2).sum()}')

rec = []
for sid, gdf in e.groupby('stay_id'):
    gdf = gdf.sort_values('hours_from_icu_intime')
    vals = gdf['lvef_value'].values
    hrs = gdf['hours_from_icu_intime'].values
    idx = np.where(vals < 50)[0]
    if len(idx) == 0:
        continue  # decline-only cases (EF never <50)
    t0 = hrs[idx[0]]
    after = gdf[(gdf['hours_from_icu_intime'] > t0)]
    recov = after[after['lvef_value'] >= 50]
    rec.append(dict(stay_id=sid, n_records=len(gdf), t_first_abnormal=t0,
                    recovered=len(recov) > 0,
                    t_recovery=(recov['hours_from_icu_intime'].iloc[0] - t0) if len(recov) > 0 else np.nan,
                    has_followup=len(after) > 0))
r = pd.DataFrame(rec)
fu = r[r['has_followup']]
log(f"[recovery] EF<50 cases with follow-up echo after first abnormal: {len(fu)}")
log(f"[recovery] recovered to LVEF>=50%: {fu['recovered'].sum()} / {len(fu)} ({fu['recovered'].mean()*100:.1f}%)")
log(f"[recovery] median time from first abnormal to recovery: {fu.loc[fu.recovered,'t_recovery'].median():.1f} h "
    f"(IQR {fu.loc[fu.recovered,'t_recovery'].quantile(.25):.1f}-{fu.loc[fu.recovered,'t_recovery'].quantile(.75):.1f})")
# 7天内恢复
log(f"[recovery] recovery within 168h: {(fu.loc[fu.recovered,'t_recovery']<=168).sum()} of {fu['recovered'].sum()} recoveries")
# 恢复与死亡的关系（描述性）
fu2 = fu.merge(s, on='stay_id')
log(f"[recovery] 28d mortality: recovered {fu2.loc[fu2.recovered,'mort_28d'].mean()*100:.1f}% vs not-recovered {fu2.loc[~fu2.recovered,'mort_28d'].mean()*100:.1f}% (descriptive)")
r.to_csv(f'{D}/results/v6_lvef_recovery_detail.csv', index=False)

with open(f'{D}/results/v6_recovery_readmission_log.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(OUT))
conn.close()
print('[done 109]')

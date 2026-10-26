# 107_ipw_recovery.py — IPW balance diagnostics (SMD, max weight, ESS); LVEF recovery rate
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

D = '<PROJECT_ROOT>/exec'
OUT = []
def log(s):
    print(s); OUT.append(str(s))

conn = psycopg2.connect(dbname='mimic', host='127.0.0.1', port=5431,
                        user='postgres', password='<SET-IN-ENVIRONMENT>')

# ---- IPW balance diagnostics (same propensity model as 101, part 4) ----
full = pd.read_sql("""SELECT stay_id, age, gender, sofa_24h, heart_rate_max, mbp_min,
                      resp_rate_max, spo2_min, lactate_max_24h, wbc_max, creatinine_max,
                      invasive_vent, cci, evaluable
                      FROM mimiciv_echo.sicm_analysis_dataset
                      WHERE age>=18 AND age<=90 AND has_acs=0""", conn)
PSF = ['age','sofa_24h','heart_rate_max','mbp_min','resp_rate_max','spo2_min',
       'lactate_max_24h','wbc_max','creatinine_max','cci']
Xps = full[PSF].copy()
for c in PSF:
    Xps[c] = Xps[c].fillna(Xps[c].median())
Xps['invasive_vent'] = full['invasive_vent'].fillna(0)
Xps['gender_f'] = (full['gender'] == 'F').astype(int)
scp = StandardScaler().fit(Xps)
pm = LogisticRegression(max_iter=1500).fit(scp.transform(Xps), full['evaluable'])
ps = np.clip(pm.predict_proba(scp.transform(Xps))[:, 1], 0.02, 0.98)
prevf = full['evaluable'].mean()
w = np.where(full['evaluable'] == 1, prevf / ps, 1.0)
we_raw = w[full['evaluable'] == 1].copy()
lo_w, hi_w = np.percentile(we_raw, [1, 99])
w[full['evaluable'] == 1] = np.clip(we_raw, lo_w, hi_w)
full['sw'] = w
log(f'[ipw] propensity model AUC={roc_auc_score(full["evaluable"], ps):.3f}, n={len(full)}, evaluable={full.evaluable.sum()}')
log(f'[ipw] weight range after 1-99pct truncation: [{w[full.evaluable==1].min():.2f}, {w[full.evaluable==1].max():.2f}]; pre-truncation max={we_raw.max():.2f}')
ev1 = (full['evaluable'] == 1).values
ess = w[ev1].sum() ** 2 / (w[ev1] ** 2).sum()
log(f'[ipw] ESS(evaluable)={ess:.0f} / {ev1.sum()} ({ess/ev1.sum()*100:.1f}%)')

def smd(x, g, wt=None):
    x1, x0 = x[g == 1].astype(float), x[g == 0].astype(float)
    if wt is None:
        m1, m0, v1, v0 = x1.mean(), x0.mean(), x1.var(), x0.var()
    else:
        w1, w0 = wt[g == 1], wt[g == 0]
        m1 = np.average(x1, weights=w1); m0 = np.average(x0, weights=w0)
        v1 = np.average((x1 - m1) ** 2, weights=w1); v0 = np.average((x0 - m0) ** 2, weights=w0)
    sp = np.sqrt((v1 + v0) / 2)
    return (m1 - m0) / sp if sp > 0 else 0.0

g = full['evaluable'].values
rows = [dict(covariate=c,
             SMD_unweighted=round(smd(Xps[c].values, g), 3),
             SMD_weighted=round(smd(Xps[c].values, g, full['sw'].values), 3))
        for c in Xps.columns]
smdf = pd.DataFrame(rows)
smdf.to_csv(f'{D}/results/v6_ipw_balance_smd.csv', index=False)
log('[ipw] SMD before/after weighting:')
log(smdf.to_string(index=False))
log(f"[ipw] max |SMD| weighted={smdf['SMD_weighted'].abs().max():.3f}; positivity: min PS among evaluable={ps[ev1].min():.3f}")

# ---- LVEF recovery rate ----
tables = pd.read_sql("SELECT table_name FROM information_schema.tables WHERE table_schema='mimiciv_echo'", conn)
log(f'\n[recovery] mimiciv_echo tables: {list(tables.table_name)}')
lvef = pd.read_sql("SELECT * FROM mimiciv_echo.echo_lvef_numeric LIMIT 3", conn)
log(f'[recovery] echo_lvef_numeric columns: {list(lvef.columns)}')

with open(f'{D}/results/v6_ipw_recovery_log.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(OUT))
conn.close()
print('[done 107]')

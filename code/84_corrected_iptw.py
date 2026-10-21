# 84_corrected_iptw.py - 修正表型队列的超声选择偏倚IPTW（HC1稳健SE）
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

D = '<PROJECT_ROOT>/exec'
conn = psycopg2.connect(dbname='mimic', host='127.0.0.1', port=5431,
                        user='postgres', password='<SET-IN-ENVIRONMENT>')
full = pd.read_sql("""SELECT stay_id, age, gender, sofa_24h, heart_rate_max, mbp_min,
                      resp_rate_max, spo2_min, lactate_max_24h, wbc_max, creatinine_max,
                      invasive_vent, cci, evaluable
                      FROM mimiciv_echo.sicm_analysis_dataset
                      WHERE age>=18 AND age<=90 AND has_acs=0""", conn)
conn.close()
PSF = ['age','sofa_24h','heart_rate_max','mbp_min','resp_rate_max','spo2_min',
       'lactate_max_24h','wbc_max','creatinine_max','cci']
Xps = full[PSF].copy()
for c in PSF:
    Xps[c] = Xps[c].fillna(Xps[c].median())
Xps['invasive_vent'] = full['invasive_vent'].fillna(0)
Xps['gender_f'] = (full['gender'] == 'F').astype(int)
sc = StandardScaler().fit(Xps)
pm = LogisticRegression(max_iter=1500).fit(sc.transform(Xps), full['evaluable'])
ps = np.clip(pm.predict_proba(sc.transform(Xps))[:, 1], 0.02, 0.98)
prev = full['evaluable'].mean()
w = np.where(full['evaluable'] == 1, prev / ps, 1.0)
we = w[full['evaluable'] == 1]
lo, hi = np.percentile(we, [1, 99])
w[full['evaluable'] == 1] = np.clip(we, lo, hi)
full['sw'] = w

co = pd.read_csv(f'{D}/data/cohort_main.csv')
dn2 = co[(co['has_hf'] == 0) & (co['congestive_heart_failure'] == 0)].copy()
dn2 = dn2.merge(full[['stay_id', 'sw']], on='stay_id', how='left')
dn2['lact_f'] = dn2['lactate_max_24h'].fillna(dn2['lactate_max_24h'].median())
CONF = ['age', 'sofa_24h', 'cci', 'lact_f', 'invasive_vent', 'aki_stage_max']
for c in ['invasive_vent', 'aki_stage_max']:
    dn2[c] = dn2[c].fillna(0)

def w_or(dsub, xcol, lab):
    d2 = dsub.dropna(subset=['sw', 'mort_28d', xcol] + CONF)
    d2 = d2[d2['sw'] > 0]
    X = sm.add_constant(pd.concat([d2[xcol], d2[CONF]], axis=1).astype(float))
    mu = sm.Logit(d2['mort_28d'].astype(float), X).fit(disp=0)
    mw = sm.GLM(d2['mort_28d'].astype(float), X, family=sm.families.Binomial(),
                freq_weights=d2['sw']).fit(cov_type='HC1')
    cu = np.exp(mu.conf_int().loc[xcol]); cw = np.exp(mw.conf_int().loc[xcol])
    print(f"{lab}: 未加权OR={np.exp(mu.params[xcol]):.3f} ({cu[0]:.3f}-{cu[1]:.3f}) | "
          f"IPTW OR={np.exp(mw.params[xcol]):.3f} (稳健95%CI {cw[0]:.3f}-{cw[1]:.3f}), "
          f"P={mw.pvalues[xcol]:.4f}")
    return dict(exposure=lab, OR_unw=round(np.exp(mu.params[xcol]), 3),
                OR_iptw=round(np.exp(mw.params[xcol]), 3),
                lo=round(cw[0], 3), hi=round(cw[1], 3), p=float(f"{mw.pvalues[xcol]:.2e}"))

rows = [w_or(dn2, 'sicm', '修正新发SICM(表型一)'),
        w_or(dn2, 'sicm_defB', '修正新发SICM失代偿(定义B)')]
pd.DataFrame(rows).to_csv(f'{D}/results/v4_corrected_iptw.csv', index=False)
print("[done]")

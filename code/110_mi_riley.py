# 110_mi_riley.py — export dataset for multiple imputation (mice); Riley minimum-sample-size criteria
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats

D = 'D:/MIMICIV/exec'
co = pd.read_csv(f'{D}/data/cohort_main.csv')
aki = pd.read_csv(f'{D}/data/aki_stage_24h.csv')
st = pd.read_csv(f'{D}/data/cohort_dn_strict24h.csv', usecols=['stay_id', 'vent_24h'])
dn = co[(co['has_hf'] == 0) & (co['congestive_heart_failure'] == 0)].copy().reset_index(drop=True)
dn = dn.merge(aki, on='stay_id', how='left').merge(st, on='stay_id', how='left')
dn['aki_stage_24h'] = dn['aki_stage_24h'].fillna(0).astype(int)
dn['vent_24h'] = dn['vent_24h'].fillna(0).astype(int)

# ---- export MI dataset (all variables of the primary association model, raw values) ----
mi = dn[['stay_id', 'mort_28d', 'sicm', 'age', 'sofa_24h', 'cci',
         'lactate_max_24h', 'vent_24h', 'aki_stage_24h']].copy()
mi.to_csv(f'{D}/data/v6_mi_dataset.csv', index=False)
print('[B12] MI dataset exported:', mi.shape, '| lactate missing:', mi['lactate_max_24h'].isna().sum())

# ---- Riley minimum-sample-size criteria ----
# 候选参数 P=47 (完整候选集) 与 P=21 (lambda.min保留变量数)
y = dn['sicm'].astype(float)
FEATS21 = ['age', 'sofa_24h', 'heart_rate_max', 'resp_rate_max', 'temperature_max',
           'lactate_max_24h', 'wbc_max', 'platelets_min', 'creatinine_max', 'bun_max',
           'bilirubin_total_max', 'sodium_min', 'potassium_max', 'glucose_max', 'inr_max',
           'ptt_max', 'aniongap_max', 'hemoglobin_min', 'ne_max_24h_strict',
           'myocardial_infarct', 'peripheral_vascular_disease']
st2 = pd.read_csv(f'{D}/data/cohort_dn_strict24h.csv',
                  usecols=['stay_id', 'ne_max_24h_strict'])
dn = dn.drop(columns=['ne_max_24h_strict'], errors='ignore').merge(st2, on='stay_id', how='left')
X = dn[FEATS21].copy()
for c in X.columns:
    X[c] = X[c].fillna(X[c].median())
X = sm.add_constant(X.astype(float))
m = sm.Logit(y, X).fit(disp=0)
m0 = sm.Logit(y, np.ones((len(y), 1))).fit(disp=0)
LR = 2 * (m.llf - m0.llf)
n = len(y); p_prev = y.mean()
r2cs_app = 1 - np.exp(-LR / n)
max_r2cs = 1 - np.exp(2 * (p_prev * np.log(p_prev) + (1 - p_prev) * np.log(1 - p_prev)))
S = 0.9
print(f'[B7] n={n}, events={int(y.sum())}, prevalence={p_prev:.3f}')
print(f'[B7] LR={LR:.1f}, R2cs_app={r2cs_app:.3f}, max_R2cs={max_r2cs:.3f}, R2Nag={r2cs_app / max_r2cs:.3f}')
for P in (21, 47):
    n1 = P / ((S - 1) * np.log(1 - r2cs_app / S))
    print(f'[B7] Criterion-1 (shrinkage>=0.9, P={P}): required n={n1:.0f} (have {n}, events {int(y.sum())}) -> {"MET" if n >= n1 else "NOT MET"}')
# 准则4: 总体风险估计精度 (95%CI半宽<=0.05)
n4 = 1.96 ** 2 * p_prev * (1 - p_prev) / 0.05 ** 2
print(f'[B7] Criterion-4 (overall-risk precision ±0.05): required n={n4:.0f} -> {"MET" if n >= n4 else "NOT MET"}')
# 准则2近似: adjusted R2损失<=0.05所需 (Riley 2020 eq for binary, criterion 2)
for P in (21, 47):
    r2cs_adj = r2cs_app * S
    n2 = P / ((r2cs_adj / (max_r2cs * 0.05)) - 1) if (r2cs_adj / (max_r2cs * 0.05)) > 1 else np.inf
    print(f'[B7] Criterion-2 (R2 precision 0.05, P={P}): approx required n={n2:.0f} -> {"MET" if n >= n2 else "NOT MET"}')
print('[done 110]')

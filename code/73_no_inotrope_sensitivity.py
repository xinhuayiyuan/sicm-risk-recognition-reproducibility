# 73_no_inotrope_sensitivity.py
# Sensitivity model without inotrope variables (dobutamine_24h / milrinone_24h)
# Rationale: rule out residual reverse causation (inotropes started within 24 h
# may follow a very early echocardiogram)
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SEED = 20260101
rng = np.random.RandomState(SEED)
D = '<PROJECT_ROOT>/exec'
dn = pd.read_csv(f'{D}/data/cohort_dn_strict24h.csv')

NUM = ['age','cci','sofa_24h','heart_rate_max','mbp_min','resp_rate_max','spo2_min',
       'temperature_max','lactate_max_24h','wbc_max','platelets_min','creatinine_max',
       'bun_max','bilirubin_total_max','sodium_min','potassium_max','glucose_max',
       'inr_max','ptt_max','bicarbonate_min','aniongap_max','hemoglobin_min',
       'ne_max_24h_strict','aki_stage_max']
COMORB = ['myocardial_infarct','peripheral_vascular_disease','cerebrovascular_disease',
          'dementia','chronic_pulmonary_disease','rheumatic_disease','mild_liver_disease',
          'diabetes_without_cc','diabetes_with_cc','renal_disease','malignant_cancer',
          'severe_liver_disease','metastatic_solid_tumor','aids']
BIN = ['gender_f','race_black','race_other'] + COMORB + \
      ['vent_24h','rrt_firstday','lactate_miss','bili_miss']     # 已去除两种正性肌力药
dn['gender_f'] = (dn['gender'] == 'F').astype(int)
rc = dn['race'].astype(str).str.upper()
dn['race_black'] = rc.str.contains('BLACK').astype(int)
dn['race_other'] = (~rc.str.contains('WHITE|BLACK')).astype(int)
for c in COMORB + ['rrt_firstday']:
    dn[c] = dn[c].fillna(0).astype(int)
FEATS = [f for f in (NUM + BIN) if f != 'aids']
y = dn['sicm'].values

def prep(a, b):
    med = a[NUM].median()
    Xa = a[FEATS].copy(); Xb = b[FEATS].copy()
    Xa[NUM] = Xa[NUM].fillna(med); Xb[NUM] = Xb[NUM].fillna(med)
    sc = StandardScaler().fit(Xa[NUM])
    Xa[NUM] = sc.transform(Xa[NUM]); Xb[NUM] = sc.transform(Xb[NUM])
    return Xa, Xb

def las_fit(X, yy):
    m = LogisticRegressionCV(Cs=20, penalty='l1', solver='liblinear', cv=10,
                             scoring='roc_auc', max_iter=3000, random_state=SEED)
    m.fit(X, yy); return m

def boot_ci(yy, pp, n=1000):
    a = []
    for _ in range(n):
        i = rng.randint(0, len(yy), len(yy))
        if len(np.unique(yy[i])) < 2: continue
        a.append(roc_auc_score(yy[i], pp[i]))
    return np.percentile(a, [2.5, 97.5])

tr, te = train_test_split(dn.index, test_size=0.3, stratify=y, random_state=SEED)
Xtr, Xte = prep(dn.loc[tr], dn.loc[te])
m = las_fit(Xtr, y[tr])
p = m.predict_proba(Xte)[:, 1]
ci = boot_ci(y[te], p)
oof = np.zeros(len(dn))
for tr2, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(dn, y):
    Xa, Xb = prep(dn.iloc[tr2], dn.iloc[va])
    oof[va] = las_fit(Xa, y[tr2]).predict_proba(Xb)[:, 1]

print("=== 去除正性肌力药变量的敏感性模型（严格24h特征集） ===")
print(f"测试集 AUC={roc_auc_score(y[te], p):.3f} (95%CI {ci[0]:.3f}-{ci[1]:.3f})")
print(f"OOF    AUC={roc_auc_score(y, oof):.3f}")
print(f"对照(含正性肌力药): 测试集 0.769 / OOF 0.743")
sel = [f for f, c in zip(FEATS, m.coef_[0]) if abs(c) > 1e-8]
print(f"选中{len(sel)}变量")
pd.DataFrame({'模型': ['含正性肌力药(测试)', '去除正性肌力药(测试)', '含正性肌力药(OOF)', '去除正性肌力药(OOF)'],
              'AUC': [0.769, round(roc_auc_score(y[te], p), 3), 0.743, round(roc_auc_score(y, oof), 3)],
              'AUC_lo': [0.721, round(ci[0], 3), np.nan, np.nan],
              'AUC_hi': [0.810, round(ci[1], 3), np.nan, np.nan]}
             ).to_csv(f'{D}/results/v3strict_no_inotrope_sensitivity.csv', index=False)
print("[done]")

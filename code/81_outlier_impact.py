# 81_outlier_impact.py - EF越界值影响量化 + 原始字符串溯源
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SEED = 20260101
D = '<PROJECT_ROOT>/exec'

# 1) 原始字符串溯源
bad_ids = [6535132, 6485615, 5239153, 5636684, 5923043, 5105919]
imp = pd.read_csv('<PROJECT_ROOT>/echo/lvef_import.csv')
print("=== 越界measurement_id的原始导入记录 ===")
print(imp[imp['measurement_id'].isin(bad_ids)][
      ['subject_id','measurement_id','measurement_description','result','unit']].to_string(index=False))

# 2) 主分析队列影响
dn = pd.read_csv(f'{D}/data/cohort_dn_strict24h.csv')
bad_stays = [33127927, 35134659, 35718523, 38259929, 38650305]
in_dn = dn[dn['stay_id'].isin(bad_stays)]
print(f"\n新发建模队列中受影响stay: {len(in_dn)} (stay_id={list(in_dn['stay_id'])})")
print(f"其SICM标记: {list(in_dn['sicm'])}")
dn_clean = dn[~dn['stay_id'].isin(bad_stays)].copy()
print(f"清洗后队列 n={len(dn_clean)} (原{len(dn)}), 事件={dn_clean.sicm.sum()} (原{dn.sicm.sum()})")
dn_clean.to_csv(f'{D}/data/cohort_dn_strict24h_clean.csv', index=False)

# 3) 清洗后OOF AUC（快速验证稳定性）
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
      ['vent_24h','dobutamine_24h','milrinone_24h','rrt_firstday','lactate_miss','bili_miss']
dn_clean['gender_f'] = (dn_clean['gender'] == 'F').astype(int)
rc = dn_clean['race'].astype(str).str.upper()
dn_clean['race_black'] = rc.str.contains('BLACK').astype(int)
dn_clean['race_other'] = (~rc.str.contains('WHITE|BLACK')).astype(int)
for c in COMORB + ['rrt_firstday']:
    dn_clean[c] = dn_clean[c].fillna(0).astype(int)
FEATS = [f for f in (NUM + BIN) if f != 'aids']
y = dn_clean['sicm'].values

oof = np.zeros(len(dn_clean))
for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(dn_clean, y):
    a, b = dn_clean.iloc[tr], dn_clean.iloc[va]
    med = a[NUM].median()
    Xa = a[FEATS].copy(); Xb = b[FEATS].copy()
    Xa[NUM] = Xa[NUM].fillna(med); Xb[NUM] = Xb[NUM].fillna(med)
    sc = StandardScaler().fit(Xa[NUM])
    Xa[NUM] = sc.transform(Xa[NUM]); Xb[NUM] = sc.transform(Xb[NUM])
    m = LogisticRegressionCV(Cs=10, penalty='l1', solver='liblinear', cv=5,
                             scoring='roc_auc', max_iter=3000, random_state=SEED)
    m.fit(Xa, y[tr]); oof[va] = m.predict_proba(Xb)[:, 1]
print(f"\n清洗后 OOF AUC = {roc_auc_score(y, oof):.3f} (清洗前 0.743)")
print("[done]")

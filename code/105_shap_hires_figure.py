# 105_shap_hires_figure.py - 以高分辨率重绘 fig4_shap_corrected.png（与102第⑦节完全同管线/同种子）
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap

SEED = 20260101
D = '<PROJECT_ROOT>/exec'
DPI = 400  # 3200x2400px ≈ 600dpi @13.5cm

co = pd.read_csv(f'{D}/data/cohort_main.csv')
aki = pd.read_csv(f'{D}/data/aki_stage_24h.csv')
st = pd.read_csv(f'{D}/data/cohort_dn_strict24h.csv',
                 usecols=['stay_id', 'ne_max_24h_strict', 'dobutamine_24h',
                          'milrinone_24h', 'vent_24h'])
co = co.merge(aki, on='stay_id', how='left').merge(st, on='stay_id', how='left')
co['aki_stage_24h'] = co['aki_stage_24h'].fillna(0).astype(int)
dn = co[(co['has_hf'] == 0) & (co['congestive_heart_failure'] == 0)].copy().reset_index(drop=True)

NUM = ['age','cci','sofa_24h','heart_rate_max','mbp_min','resp_rate_max','spo2_min',
       'temperature_max','lactate_max_24h','wbc_max','platelets_min','creatinine_max',
       'bun_max','bilirubin_total_max','sodium_min','potassium_max','glucose_max',
       'inr_max','ptt_max','bicarbonate_min','aniongap_max','hemoglobin_min',
       'ne_max_24h_strict','aki_stage_24h']
COMORB = ['myocardial_infarct','peripheral_vascular_disease','cerebrovascular_disease',
          'dementia','chronic_pulmonary_disease','rheumatic_disease','mild_liver_disease',
          'diabetes_without_cc','diabetes_with_cc','renal_disease','malignant_cancer',
          'severe_liver_disease','metastatic_solid_tumor','aids']
BIN = ['gender_f','race_black','race_other'] + COMORB + \
      ['vent_24h','dobutamine_24h','milrinone_24h','rrt_firstday','lactate_miss','bili_miss']

dn['gender_f'] = (dn['gender'] == 'F').astype(int)
rc = dn['race'].astype(str).str.upper()
dn['race_black'] = rc.str.contains('BLACK').astype(int)
dn['race_other'] = (~rc.str.contains('WHITE|BLACK')).astype(int)
for c in COMORB + ['rrt_firstday']:
    dn[c] = dn[c].fillna(0).astype(int)
FEATS = [f for f in (NUM + BIN) if f != 'aids']
y = dn['sicm'].values

FEAT_EN = {'myocardial_infarct': 'Prior myocardial infarction',
           'ptt_max': 'PTT max', 'glucose_max': 'Glucose max',
           'heart_rate_max': 'Heart rate max', 'temperature_max': 'Temperature max',
           'hemoglobin_min': 'Haemoglobin min', 'platelets_min': 'Platelets min',
           'sodium_min': 'Sodium min', 'aniongap_max': 'Anion gap max',
           'resp_rate_max': 'Respiratory rate max', 'age': 'Age',
           'wbc_max': 'WBC max', 'potassium_max': 'Potassium max',
           'inr_max': 'INR max', 'bilirubin_total_max': 'Bilirubin max',
           'creatinine_max': 'Creatinine max', 'bun_max': 'BUN max',
           'lactate_max_24h': 'Lactate max', 'spo2_min': 'SpO2 min',
           'mbp_min': 'MAP min', 'sofa_24h': 'SOFA score', 'cci': 'Charlson index',
           'ne_max_24h_strict': 'Peak norepinephrine (24 h)',
           'aki_stage_24h': 'AKI stage (0–24 h)', 'bicarbonate_min': 'Bicarbonate min',
           'dobutamine_24h': 'Dobutamine (24 h)', 'milrinone_24h': 'Milrinone (24 h)',
           'vent_24h': 'Invasive ventilation (24 h)', 'rrt_firstday': 'RRT (day 1)',
           'gender_f': 'Female sex', 'race_black': 'Black race', 'race_other': 'Other race',
           'lactate_miss': 'Lactate missing', 'bili_miss': 'Bilirubin missing',
           'peripheral_vascular_disease': 'Peripheral vascular disease',
           'cerebrovascular_disease': 'Cerebrovascular disease', 'dementia': 'Dementia',
           'chronic_pulmonary_disease': 'Chronic pulmonary disease',
           'rheumatic_disease': 'Rheumatic disease', 'mild_liver_disease': 'Mild liver disease',
           'diabetes_without_cc': 'Diabetes', 'diabetes_with_cc': 'Diabetes with complications',
           'renal_disease': 'Renal disease', 'malignant_cancer': 'Malignancy',
           'severe_liver_disease': 'Severe liver disease',
           'metastatic_solid_tumor': 'Metastatic tumour', 'aids': 'AIDS'}

X0 = dn[FEATS].fillna(dn[FEATS].median())
sc0 = StandardScaler().fit(X0[NUM])
Xs0 = X0.copy(); Xs0[NUM] = sc0.transform(Xs0[NUM])
mx2 = xgb.train(dict(objective='binary:logistic', eval_metric='auc', max_depth=3, eta=0.05,
                     subsample=0.8, colsample_bytree=0.8, min_child_weight=20, seed=SEED),
                xgb.DMatrix(Xs0, label=y), num_boost_round=400, verbose_eval=False)
sv = shap.TreeExplainer(mx2).shap_values(Xs0)
labels = [FEAT_EN.get(f, f) for f in FEATS]

plt.figure(figsize=(8, 6))
shap.summary_plot(sv, Xs0, feature_names=labels, show=False, max_display=15)
plt.tight_layout()
plt.savefig(f'{D}/figures/fig4_shap_corrected_hires.png', dpi=DPI, bbox_inches='tight')
plt.close('all')

imp = pd.DataFrame({'feature': FEATS, 'mean_abs_shap': np.abs(sv).mean(axis=0)}
                   ).sort_values('mean_abs_shap', ascending=False)
print('SHAP Top10 (应与v5_shap_importance.csv一致):')
print(imp.head(10).to_string(index=False))
from PIL import Image
im = Image.open(f'{D}/figures/fig4_shap_corrected_hires.png')
print('hires:', im.size, im.info.get('dpi'), '| eff.dpi@13.5cm = %.0f' % (im.size[0] / (13.5 / 2.54)))

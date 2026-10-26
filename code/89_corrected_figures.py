# 89_corrected_figures.py - 修正队列版 Fig3(NE剂量-反应RCS) 与 Fig4(SHAP)
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from patsy import dmatrix
import statsmodels.api as sm
import xgboost as xgb
import shap

SEED = 20260101
D = '<PROJECT_ROOT>/exec'
co = pd.read_csv(f'{D}/data/cohort_main.csv')
dn = co[(co['has_hf'] == 0) & (co['congestive_heart_failure'] == 0)].copy().reset_index(drop=True)
st = pd.read_csv(f'{D}/data/cohort_dn_strict24h.csv',
                 usecols=['stay_id', 'ne_max_24h_strict', 'dobutamine_24h',
                          'milrinone_24h', 'vent_24h'])
dn = dn.merge(st, on='stay_id', how='left')

# ---- Fig3: RCS 剂量-反应（修正队列） ----
fig, ax = plt.subplots(figsize=(7, 5))
for lab, mask, c in [('New-onset SICM', dn['sicm'] == 1, '#d62728'),
                     ('Non-SICM reference', dn['sicm'] == 0, '#1f77b4')]:
    dsub = dn[mask]
    ned = dsub['post_max_ned'].fillna(0).values.astype(float)
    yy = dsub['mort_28d'].values.astype(float)
    Xs = dmatrix("cr(ned, df=4)", {'ned': ned}, return_type='dataframe')
    m = sm.Logit(yy, Xs).fit(disp=0)
    grid = np.linspace(0, np.percentile(ned, 99), 100)
    Xg = dmatrix("cr(ned, df=4)", {'ned': grid}, return_type='dataframe')
    ax.plot(grid, m.predict(Xg), color=c, label=f"{lab} (n={len(dsub)})")
ax.set_xlabel('Peak norepinephrine-equivalent dose (post-onset)')
ax.set_ylabel('Predicted 28-day mortality')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{D}/figures/fig3_ne_dose_response_corrected.png', dpi=200)
print('fig3 saved')

# ---- Fig4: SHAP（修正队列） ----
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
dn['gender_f'] = (dn['gender'] == 'F').astype(int)
rc = dn['race'].astype(str).str.upper()
dn['race_black'] = rc.str.contains('BLACK').astype(int)
dn['race_other'] = (~rc.str.contains('WHITE|BLACK')).astype(int)
for c in COMORB + ['rrt_firstday']:
    dn[c] = dn[c].fillna(0).astype(int)
FEATS = [f for f in (NUM + BIN) if f != 'aids']
X = dn[FEATS].fillna(dn[FEATS].median())
y = dn['sicm'].values
sc_ = StandardScaler().fit(X[NUM])
Xs = X.copy(); Xs[NUM] = sc_.transform(Xs[NUM])
mx = xgb.train(dict(objective='binary:logistic', eval_metric='auc', max_depth=3, eta=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=20, seed=SEED),
               xgb.DMatrix(Xs, label=y), num_boost_round=400, verbose_eval=False)
sv = shap.TreeExplainer(mx).shap_values(Xs)
shap.summary_plot(sv, Xs, feature_names=FEATS, show=False, max_display=15)
plt.tight_layout()
plt.savefig(f'{D}/figures/fig4_shap_corrected.png', dpi=200, bbox_inches='tight')
print('fig4 saved')

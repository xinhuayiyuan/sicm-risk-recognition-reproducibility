# 114_fig2_v6.py — C1: Fig2重绘 (A)补1-SE曲线 (B)校准散点+loess; 模型复现与99e/101完全一致
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb

SEED = 20260101
D = '<PROJECT_ROOT>/exec'

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

def prep(a, b):
    med = a[NUM].median()
    Xa = a[FEATS].copy(); Xb = b[FEATS].copy()
    Xa[NUM] = Xa[NUM].fillna(med); Xb[NUM] = Xb[NUM].fillna(med)
    sc = StandardScaler().fit(Xa[NUM])
    Xa[NUM] = sc.transform(Xa[NUM]); Xb[NUM] = sc.transform(Xb[NUM])
    return Xa, Xb

y = dn['sicm'].values
tr, te = train_test_split(dn.index, test_size=0.3, stratify=y, random_state=SEED)
dtr, dte = dn.loc[tr].copy(), dn.loc[te].copy()
Xtr, Xte = prep(dtr, dte)
ytr, yte = y[tr], y[te]

las = LogisticRegressionCV(Cs=20, penalty='l1', solver='liblinear', cv=10,
                           scoring='roc_auc', max_iter=3000, random_state=SEED).fit(Xtr, ytr)
p1 = las.predict_proba(Xte)[:, 1]
means = las.scores_[1].mean(axis=0); ses = las.scores_[1].std(axis=0) / np.sqrt(10)
c1se = las.Cs_[np.where(means >= means.max() - ses[means.argmax()])[0][0]]
las1 = LogisticRegression(C=c1se, penalty='l1', solver='liblinear',
                          max_iter=3000, random_state=SEED).fit(Xtr, ytr)
p2 = las1.predict_proba(Xte)[:, 1]
momed = dtr[['age', 'sofa_24h']].median()
mb = LogisticRegression(max_iter=1000).fit(dtr[['age', 'sofa_24h']].fillna(momed), ytr)
pb = mb.predict_proba(dte[['age', 'sofa_24h']].fillna(momed))[:, 1]
itr, ival = train_test_split(np.arange(len(Xtr)), test_size=0.2, stratify=ytr, random_state=SEED)
mx = xgb.train(dict(objective='binary:logistic', eval_metric='auc', max_depth=3, eta=0.03,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=20, seed=SEED),
               xgb.DMatrix(Xtr.iloc[itr], label=ytr[itr]), num_boost_round=2000,
               evals=[(xgb.DMatrix(Xtr.iloc[ival], label=ytr[ival]), 'val')],
               early_stopping_rounds=50, verbose_eval=False)
p_val = mx.predict(xgb.DMatrix(Xtr.iloc[ival]), iteration_range=(0, mx.best_iteration + 1))
p_raw = mx.predict(xgb.DMatrix(Xte), iteration_range=(0, mx.best_iteration + 1))
p_xgb = LogisticRegression(max_iter=1000).fit(p_val.reshape(-1, 1), ytr[ival]) \
          .predict_proba(p_raw.reshape(-1, 1))[:, 1]
ml = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, max_depth=3, subsample=0.8,
                        colsample_bytree=0.8, min_child_samples=20, random_state=SEED,
                        verbose=-1, n_jobs=-1)
ml.fit(Xtr.iloc[itr], ytr[itr], eval_set=[(Xtr.iloc[ival], ytr[ival])],
       callbacks=[lgb.early_stopping(50, verbose=False)])
p_val_l = ml.predict_proba(Xtr.iloc[ival])[:, 1]
p_raw_l = ml.predict_proba(Xte)[:, 1]
p_lgb = LogisticRegression(max_iter=1000).fit(p_val_l.reshape(-1, 1), ytr[ival]) \
          .predict_proba(p_raw_l.reshape(-1, 1))[:, 1]
print(f'[sanity] AUCs: lambda.min={roc_auc_score(yte,p1):.3f} 1SE={roc_auc_score(yte,p2):.3f} '
      f'xgb={roc_auc_score(yte,p_xgb):.3f} lgb={roc_auc_score(yte,p_lgb):.3f} base={roc_auc_score(yte,pb):.3f}')

MODELS = [('LASSO-logistic (lambda.min, primary)', p1, '#d62728'),
          ('LASSO-logistic (1-SE)', p2, '#9467bd'),
          ('XGBoost (Platt)', p_xgb, '#ff7f0e'),
          ('LightGBM (Platt)', p_lgb, '#2ca02c'),
          ('Baseline (age+SOFA)', pb, '#7f7f7f')]

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.1))
fig.suptitle('MIMIC new-onset cohort | Held-out test set: n = 237 (65 SICM) | Predictors: strict 0–24 h',
             fontsize=12, y=0.995)
ax = axes[0]
for name, p, c in MODELS:
    fpr, tpr, _ = roc_curve(yte, p)
    ax.plot(fpr, tpr, color=c, lw=1.6, label=f'{name} (AUC {roc_auc_score(yte, p):.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
ax.set_title('A. ROC (test set)'); ax.legend(fontsize=7.5, loc='lower right'); ax.grid(alpha=0.3)

ax = axes[1]
for name, p, c in MODELS:
    pc_ = calibration_curve(yte, p, n_bins=10, strategy='quantile')
    ax.plot(pc_[1], pc_[0], 'o', color=c, ms=4.5, alpha=0.85)
    lo = lowess(yte, p, frac=0.75, it=0)
    ax.plot(lo[:, 0], lo[:, 1], color=c, lw=1.5, label=name)
ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed frequency')
ax.set_title('B. Calibration (quantile bins + loess)'); ax.legend(fontsize=7.5, loc='upper left')
ax.grid(alpha=0.3)

ax = axes[2]
dca = pd.read_csv(f'{D}/results/v5_dca.csv')
for name, c, ls in [('LASSO lambda.min', '#d62728', '-'), ('Baseline', '#7f7f7f', '-'),
                    ('Treat all', '#1f77b4', '--'), ('Treat none', '#000000', '--')]:
    dd = dca[dca.model == name]
    lab = 'LASSO-logistic (primary)' if name == 'LASSO lambda.min' else name
    ax.plot(dd.threshold, dd.net_benefit, label=lab, color=c, ls=ls, lw=1.4)
ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit')
ax.set_ylim(-0.05, 0.3); ax.set_title('C. Decision-curve analysis')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(f'{D}/figures/fig2_model_performance.png', dpi=200)
plt.close('all'); print('[done 114: fig2 v6 saved]')

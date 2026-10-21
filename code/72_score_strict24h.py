# 72_score_strict24h.py - 基于严格24h模型的最终评分卡 + SHAP（替代65的旧特征版本）
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import xgboost as xgb
import shap

SEED = 20260101
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
      ['vent_24h','dobutamine_24h','milrinone_24h','rrt_firstday','lactate_miss','bili_miss']
dn['gender_f'] = (dn['gender'] == 'F').astype(int)
rc = dn['race'].astype(str).str.upper()
dn['race_black'] = rc.str.contains('BLACK').astype(int)
dn['race_other'] = (~rc.str.contains('WHITE|BLACK')).astype(int)
for c in COMORB + ['rrt_firstday']:
    dn[c] = dn[c].fillna(0).astype(int)
FEATS = [f for f in (NUM + BIN) if f != 'aids']
y = dn['sicm'].values
tr, te = train_test_split(dn.index, test_size=0.3, stratify=y, random_state=SEED)
dtr, dte = dn.loc[tr].copy(), dn.loc[te].copy()
med = dtr[NUM].median()
dtr[NUM] = dtr[NUM].fillna(med); dte[NUM] = dte[NUM].fillna(med)
ytr, yte = dtr['sicm'].values, dte['sicm'].values

sc = StandardScaler().fit(dtr[NUM])
Xtr = dtr[FEATS].copy(); Xte = dte[FEATS].copy()
Xtr[NUM] = sc.transform(Xtr[NUM]); Xte[NUM] = sc.transform(Xte[NUM])
las = LogisticRegressionCV(Cs=20, penalty='l1', solver='liblinear', cv=10,
                           scoring='roc_auc', max_iter=3000, random_state=SEED)
las.fit(Xtr, ytr)
means = las.scores_[1].mean(axis=0); ses = las.scores_[1].std(axis=0) / np.sqrt(10)
c_1se = las.Cs_[np.where(means >= means.max() - ses[means.argmax()])[0][0]]
las1 = LogisticRegression(C=c_1se, penalty='l1', solver='liblinear',
                          max_iter=3000, random_state=SEED).fit(Xtr, ytr)
sel = [f for f, c in zip(FEATS, las1.coef_[0]) if abs(c) > 1e-8]
auc_las = roc_auc_score(yte, las1.predict_proba(Xte)[:, 1])
print(f"严格24h 1-SE模型({len(sel)}变量): 测试集AUC={auc_las:.3f}")

# 整数评分（z标化）
cont = [v for v in sel if v in NUM]
mu, sd_ = dtr[cont].mean(), dtr[cont].std().replace(0, 1)
Ztr = dtr[sel].astype(float).copy(); Zte = dte[sel].astype(float).copy()
Ztr[cont] = (Ztr[cont] - mu) / sd_; Zte[cont] = (Zte[cont] - mu) / sd_
m = sm.Logit(ytr, sm.add_constant(Ztr)).fit(disp=0)
beta = m.params.drop('const')
unit = np.median(np.abs(beta))
pts = (beta / unit).round().astype(int)
pts[pts == 0] = np.sign(beta[pts == 0]).astype(int)
score_te = (Zte * pts).sum(axis=1)
print(f"整数评分: 测试集AUC={roc_auc_score(yte, score_te):.3f} (衰减{auc_las - roc_auc_score(yte, score_te):+.3f})")
ptab = pd.DataFrame({'变量': sel, '系数_z': beta.round(3).values, '分值': pts.values})
ptab.to_csv(f'{D}/results/v3strict_DN_integer_score.csv', index=False)
print(ptab.to_string(index=False))

# SHAP（XGBoost，严格特征）
itr, ival = train_test_split(np.arange(len(Xtr)), test_size=0.2, stratify=ytr, random_state=SEED)
mx = xgb.train(dict(objective='binary:logistic', eval_metric='auc', max_depth=3, eta=0.03,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=20, seed=SEED),
               xgb.DMatrix(Xtr.iloc[itr], label=ytr[itr]), num_boost_round=2000,
               evals=[(xgb.DMatrix(Xtr.iloc[ival], label=ytr[ival]), 'val')],
               early_stopping_rounds=50, verbose_eval=False)
sv = shap.TreeExplainer(mx).shap_values(Xte)
imp = pd.DataFrame({'feature': FEATS, 'mean_abs_shap': np.abs(sv).mean(axis=0)}
                   ).sort_values('mean_abs_shap', ascending=False)
imp.to_csv(f'{D}/results/v3strict_DN_shap_importance.csv', index=False)
print("\nSHAP Top10 (严格24h):"); print(imp.head(10).to_string(index=False))
print("\n[done]")

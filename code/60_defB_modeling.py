# 60_defB_modeling.py
# V3.0表型二：新发SICM伴血流动力学失代偿（复合定义B）完整建模 + 预后关联
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
import statsmodels.api as sm
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

SEED = 20260101
rng = np.random.RandomState(SEED)
D = '<PROJECT_ROOT>/exec'
df = pd.read_csv(f'{D}/data/cohort_main.csv')

NUM = ['age','cci','sofa_24h','heart_rate_max','mbp_min','resp_rate_max','spo2_min',
       'temperature_max','lactate_max_24h','wbc_max','platelets_min','creatinine_max',
       'bun_max','bilirubin_total_max','sodium_min','potassium_max','glucose_max',
       'inr_max','ptt_max','bicarbonate_min','aniongap_max','hemoglobin_min','aki_stage_max']
# 注意: 定义B含血管活性药 -> post_max_ned/dobutamine/milrinone/vaso_new_upgrade 全部剔除（循环）
COMORB = ['myocardial_infarct','peripheral_vascular_disease','cerebrovascular_disease',
          'dementia','chronic_pulmonary_disease','rheumatic_disease','mild_liver_disease',
          'diabetes_without_cc','diabetes_with_cc','renal_disease','malignant_cancer',
          'severe_liver_disease','metastatic_solid_tumor','aids']
BIN = ['gender_f','race_black','race_other'] + COMORB + \
      ['invasive_vent','rrt_firstday','lactate_miss','bili_miss']
df['gender_f'] = (df['gender'] == 'F').astype(int)
rc = df['race'].astype(str).str.upper()
df['race_black'] = rc.str.contains('BLACK').astype(int)
df['race_other'] = (~rc.str.contains('WHITE|BLACK')).astype(int)
for c in COMORB + ['invasive_vent', 'rrt_firstday']:
    df[c] = df[c].fillna(0).astype(int)
FEATS = NUM + BIN
print(f"定义B特征数(剔除循环变量后): {len(FEATS)}")

dn = df[df['has_hf'] == 0].reset_index(drop=True)
y = dn['sicm_defB'].values
print(f"新发队列 n={len(dn)}, 定义B事件={y.sum()} ({y.mean()*100:.1f}%)")

def prep(a, b):
    med = a[NUM].median()
    Xa = a[FEATS].copy(); Xb = b[FEATS].copy()
    Xa[NUM] = Xa[NUM].fillna(med); Xb[NUM] = Xb[NUM].fillna(med)
    sc = StandardScaler().fit(Xa[NUM])
    Xa[NUM] = sc.transform(Xa[NUM]); Xb[NUM] = sc.transform(Xb[NUM])
    return Xa, Xb

def calib_stats(yy, p):
    p = np.clip(p, 1e-6, 1-1e-6); lp = np.log(p/(1-p))
    try:
        inter = sm.GLM(yy, np.ones((len(yy),1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    except Exception: inter = np.nan
    try:
        prm = sm.Logit(yy, sm.add_constant(lp)).fit(disp=0).params
        slope = prm.iloc[1] if len(prm) > 1 else np.nan
    except Exception: slope = np.nan
    pc = calibration_curve(yy, p, n_bins=10, strategy='quantile')
    return inter, slope, np.mean(np.abs(pc[1]-pc[0]))

def boot_ci(yy, pp, n=1000):
    a = []
    for _ in range(n):
        i = rng.randint(0, len(yy), len(yy))
        if len(np.unique(yy[i])) < 2: continue
        a.append(roc_auc_score(yy[i], pp[i]))
    return np.percentile(a, [2.5, 97.5])

tr, te = train_test_split(np.arange(len(dn)), test_size=0.3, stratify=y, random_state=SEED)
Xtr, Xte = prep(dn.iloc[tr], dn.iloc[te])
ytr, yte = y[tr], y[te]

rows = {}
las = LogisticRegressionCV(Cs=20, penalty='l1', solver='liblinear', cv=10,
                           scoring='roc_auc', max_iter=3000, random_state=SEED)
las.fit(Xtr, ytr)
p_las = las.predict_proba(Xte)[:, 1]
sel = [f for f, c in zip(FEATS, las.coef_[0]) if abs(c) > 1e-8]
print(f"定义B LASSO选中 {len(sel)} 变量: {sel}")

# XGBoost + Platt校准（V3.0预设）
itr, ival = train_test_split(np.arange(len(Xtr)), test_size=0.2, stratify=ytr, random_state=SEED)
mx = xgb.train(dict(objective='binary:logistic', eval_metric='auc', max_depth=3, eta=0.03,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=20, seed=SEED),
               xgb.DMatrix(Xtr.iloc[itr], label=ytr[itr]), num_boost_round=2000,
               evals=[(xgb.DMatrix(Xtr.iloc[ival], label=ytr[ival]), 'val')],
               early_stopping_rounds=50, verbose_eval=False)
p_val = mx.predict(xgb.DMatrix(Xtr.iloc[ival]), iteration_range=(0, mx.best_iteration+1))
p_xgb_raw = mx.predict(xgb.DMatrix(Xte), iteration_range=(0, mx.best_iteration+1))
platt = LogisticRegression(max_iter=1000).fit(p_val.reshape(-1,1), ytr[ival])
p_xgb_cal = platt.predict_proba(p_xgb_raw.reshape(-1,1))[:, 1]

for name, p in [('LASSO-Logistic', p_las), ('XGBoost(未校准)', p_xgb_raw), ('XGBoost(Platt)', p_xgb_cal)]:
    inter, slope, ici = calib_stats(yte, p)
    ci = boot_ci(yte, p)
    rows[name] = dict(AUC=round(roc_auc_score(yte, p), 3), AUC_lo=round(ci[0],3), AUC_hi=round(ci[1],3),
                      PR_AUC=round(average_precision_score(yte, p),3),
                      Brier=round(brier_score_loss(yte, p),3),
                      Cal_slope=round(slope,3), ICI=round(ici,3))
tab = pd.DataFrame(rows).T
print("\n=== 定义B模型性能（测试集） ==="); print(tab.to_string())
tab.to_csv(f'{D}/results/v3_defB_model_performance.csv')

# 预后关联：定义B vs 非SICM vs 定义A-only（EF+但未失代偿）
dn['grp'] = '非SICM'
dn.loc[(dn['sicm']==1)&(dn['sicm_defB']==0), 'grp'] = 'SICM未失代偿'
dn.loc[dn['sicm_defB']==1, 'grp'] = 'SICM失代偿(定义B)'
dn['lact_f'] = dn['lactate_max_24h'].fillna(dn['lactate_max_24h'].median())
CONF = ['age','sofa_24h','cci','lact_f','invasive_vent','aki_stage_max']
G = pd.get_dummies(dn['grp'], drop_first=False)
Xa = pd.concat([G[['SICM未失代偿','SICM失代偿(定义B)']], dn[CONF]], axis=1).astype(float)
m = sm.Logit(dn['mort_28d'], sm.add_constant(Xa)).fit(disp=0)
print("\n=== 28天死亡调整OR（参照=非SICM） ===")
for g in ['SICM未失代偿','SICM失代偿(定义B)']:
    ci = np.exp(m.conf_int().loc[g])
    print(f"{g}: OR={np.exp(m.params[g]):.3f} (95%CI {ci[0]:.3f}-{ci[1]:.3f}), P={m.pvalues[g]:.4f}")
crude = dn.groupby('grp')['mort_28d'].agg(['size','sum','mean'])
print("\n粗死亡率:"); print(crude)
print("[done]")

# 20_aim1_modeling.py
# 阶段B-步骤2：Aim 1 表型分层建模（方案V2.0 §五）
# Model-DN（新发，主模型LASSO-Logistic）+ Model-PD（既存心脏病，简约模型）+ 跨表型迁移
# 种子 20260101；插补仅用训练集统计量；评估: AUC/PR-AUC/Brier/校准/ICI/DCA + bootstrap CI
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
import statsmodels.api as sm
import xgboost as xgb, lightgbm as lgb

SEED = 20260101
rng = np.random.RandomState(SEED)
D = '<PROJECT_ROOT>/exec'

df = pd.read_csv(f'{D}/data/cohort_main.csv')

# ---------- 特征清单（方案4.5，剔除白蛋白/cTn/结局成分/中介） ----------
NUM = ['age', 'cci', 'sofa_24h', 'heart_rate_max', 'mbp_min', 'resp_rate_max',
       'spo2_min', 'temperature_max', 'lactate_max_24h', 'wbc_max', 'platelets_min',
       'creatinine_max', 'bun_max', 'bilirubin_total_max', 'sodium_min', 'potassium_max',
       'glucose_max', 'inr_max', 'ptt_max', 'bicarbonate_min', 'aniongap_max',
       'hemoglobin_min', 'post_max_ned', 'aki_stage_max']
COMORB = ['myocardial_infarct','peripheral_vascular_disease','cerebrovascular_disease',
          'dementia','chronic_pulmonary_disease','rheumatic_disease','mild_liver_disease',
          'diabetes_without_cc','diabetes_with_cc','renal_disease','malignant_cancer',
          'severe_liver_disease','metastatic_solid_tumor','aids']
BIN = ['gender_f', 'race_black', 'race_other'] + COMORB + \
      ['invasive_vent', 'dobutamine', 'milrinone', 'rrt_firstday', 'lactate_miss', 'bili_miss']
df['gender_f'] = (df['gender'] == 'F').astype(int)
rc = df['race'].astype(str).str.upper()
df['race_black'] = rc.str.contains('BLACK').astype(int)
df['race_other'] = (~rc.str.contains('WHITE|BLACK')).astype(int)
for c in COMORB + ['invasive_vent', 'dobutamine', 'milrinone', 'rrt_firstday']:
    df[c] = df[c].fillna(0).astype(int)
FEATS = NUM + BIN
print(f"特征数: {len(FEATS)}")

def prep(dtrain, dtest):
    med = dtrain[NUM].median()
    Xtr = dtrain[FEATS].copy(); Xte = dtest[FEATS].copy()
    Xtr[NUM] = Xtr[NUM].fillna(med); Xte[NUM] = Xte[NUM].fillna(med)
    sc = StandardScaler().fit(Xtr[NUM])
    Xtr[NUM] = sc.transform(Xtr[NUM]); Xte[NUM] = sc.transform(Xte[NUM])
    return Xtr, Xte

def calib_stats(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    try:
        inter = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    except Exception:
        inter = np.nan
    try:
        prm = sm.Logit(y, sm.add_constant(lp)).fit(disp=0).params
        slope = prm.iloc[1] if len(prm) > 1 else np.nan
    except Exception:
        slope = np.nan
    try:
        pc = calibration_curve(y, p, n_bins=10, strategy='quantile')
        ici = np.mean(np.abs(pc[1] - pc[0]))
    except Exception:
        ici = np.nan
    return inter, slope, ici

def boot_auc_ci(y, p, n=1000):
    y = np.asarray(y); p = np.asarray(p); aucs = []
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2: continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(aucs, [2.5, 97.5])

def evaluate(y, p):
    auc = roc_auc_score(y, p); ci = boot_auc_ci(y, p)
    inter, slope, ici = calib_stats(y, p)
    return dict(AUC=round(auc, 3), AUC_lo=round(ci[0], 3), AUC_hi=round(ci[1], 3),
                PR_AUC=round(average_precision_score(y, p), 3),
                Brier=round(brier_score_loss(y, p), 3),
                Cal_intercept=round(inter, 3), Cal_slope=round(slope, 3), ICI=round(ici, 3))


def fit_all(dtr, dte, ytr, yte, tag):
    res, preds, models = {}, {}, {}
    Xtr, Xte = prep(dtr, dte)
    bi = [FEATS.index('age'), FEATS.index('sofa_24h')]
    mb = LogisticRegression(max_iter=1000).fit(Xtr.iloc[:, bi], ytr)
    preds['Baseline(age+SOFA)'] = mb.predict_proba(Xte.iloc[:, bi])[:, 1]
    las = LogisticRegressionCV(Cs=20, penalty='l1', solver='saga', cv=10,
                               scoring='roc_auc', max_iter=5000, random_state=SEED, n_jobs=-1)
    las.fit(Xtr, ytr)
    preds['LASSO-Logistic'] = las.predict_proba(Xte)[:, 1]
    models['lasso'] = las
    sel = [f for f, c in zip(FEATS, las.coef_[0]) if abs(c) > 1e-8]
    print(f"[{tag}] LASSO选中 {len(sel)} 变量: {sel}")
    itr, ival = train_test_split(np.arange(len(Xtr)), test_size=0.2, stratify=ytr, random_state=SEED)
    dxtr = xgb.DMatrix(Xtr.iloc[itr], label=ytr.iloc[itr])
    dxva = xgb.DMatrix(Xtr.iloc[ival], label=ytr.iloc[ival])
    mxgb = xgb.train(dict(objective='binary:logistic', eval_metric='auc', max_depth=3,
                          eta=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                          seed=SEED), dxtr, num_boost_round=2000,
                     evals=[(dxva, 'val')], early_stopping_rounds=50, verbose_eval=False)
    preds['XGBoost'] = mxgb.predict(xgb.DMatrix(Xte), iteration_range=(0, mxgb.best_iteration + 1))
    models['xgb'] = mxgb
    mlgb = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, max_depth=3,
                              subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                              random_state=SEED, verbose=-1, n_jobs=-1)
    mlgb.fit(Xtr.iloc[itr], ytr.iloc[itr], eval_set=[(Xtr.iloc[ival], ytr.iloc[ival])],
             callbacks=[lgb.early_stopping(50, verbose=False)])
    preds['LightGBM'] = mlgb.predict_proba(Xte)[:, 1]
    for name, p in preds.items():
        res[name] = evaluate(yte, p)
    return res, preds, models, (Xtr, Xte), sel

# ================= Model-DN（新发，主分析） =================
dn = df[df['has_hf'] == 0].reset_index(drop=True)
tr, te = train_test_split(dn.index, test_size=0.3, stratify=dn['sicm'], random_state=SEED)
dtr, dte = dn.loc[tr], dn.loc[te]
ytr, yte = dn.loc[tr, 'sicm'], dn.loc[te, 'sicm']
print(f"\n=== Model-DN: train={len(dtr)} (事件{ytr.sum()}) / test={len(dte)} (事件{yte.sum()}) ===")
res_dn, pred_dn, mod_dn, X_dn, sel_dn = fit_all(dtr, dte, ytr, yte, 'DN')
tab_dn = pd.DataFrame(res_dn).T
print(tab_dn.to_string())
tab_dn.to_csv(f'{D}/results/aim1_DN_model_performance.csv')

# LASSO选中变量 OR 表（未标化尺度重拟合，便于解释）
med = dtr[NUM].median()
Rtr = dtr[FEATS].copy()
Rtr[NUM] = Rtr[NUM].fillna(med)
or_rows = []
for v in sel_dn:
    m = sm.Logit(ytr, sm.add_constant(Rtr[[v]])).fit(disp=0)
    or_rows.append(dict(Variable=v, OR=round(np.exp(m.params[v]), 3),
                        OR_lo=round(np.exp(m.conf_int().loc[v, 0]), 3),
                        OR_hi=round(np.exp(m.conf_int().loc[v, 1]), 3),
                        p=float(f"{m.pvalues[v]:.2e}")))
pd.DataFrame(or_rows).to_csv(f'{D}/results/aim1_DN_lasso_OR_table.csv', index=False)

# ================= Model-PD（既存心脏病，简约） =================
pdco = df[df['has_hf'] == 1].reset_index(drop=True)
tr2, te2 = train_test_split(pdco.index, test_size=0.3, stratify=pdco['sicm'], random_state=SEED)
res_pd, pred_pd, mod_pd, X_pd, sel_pd = fit_all(pdco.loc[tr2], pdco.loc[te2],
                                                pdco.loc[tr2, 'sicm'], pdco.loc[te2, 'sicm'], 'PD')
tab_pd = pd.DataFrame(res_pd).T
print("\n=== Model-PD ==="); print(tab_pd.to_string())
tab_pd.to_csv(f'{D}/results/aim1_PD_model_performance.csv')

# ================= 跨表型迁移（方案5.1核心问题） =================
Xdn_all, Xpd_all = prep(dn, pdco)
p_dn2pd = mod_dn['lasso'].predict_proba(Xpd_all)[:, 1]
Xpd_all2, Xdn_all2 = prep(pdco, dn)
p_pd2dn = mod_pd['lasso'].predict_proba(Xdn_all2)[:, 1]
mig = pd.DataFrame({
    '方向': ['Model-DN -> 既存心脏病人群', 'Model-PD -> 新发人群'],
    'AUC_迁移': [round(roc_auc_score(pdco['sicm'], p_dn2pd), 3),
                round(roc_auc_score(dn['sicm'], p_pd2dn), 3)]})
mig.to_csv(f'{D}/results/aim1_cross_phenotype_transport.csv', index=False)
print("\n=== 跨表型迁移 ==="); print(mig.to_string(index=False))
print("[done]")
